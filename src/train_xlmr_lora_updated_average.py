"""
XLM-RoBERTa + LoRA for RTI legal meaning preservation regression.
Run from project root: python src/train_xlmr_lora.py
"""
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.optim import AdamW
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "RTI_Model_Ready_Dataset_Updated_Average.csv"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "xlmr_lora_updated_average_best_model"
MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_LENGTH, BATCH_SIZE, EPOCHS = 256, 4, 15
LEARNING_RATE, WEIGHT_DECAY, PATIENCE = 1e-4, 0.01, 3
LORA_R, LORA_ALPHA, LORA_DROPOUT = 8, 16, 0.1
SEED, NUM_WORKERS = 42, 0

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class LMPDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.originals = frame["Original_Text"].astype(str).tolist()
        self.simplified = frame["Simplified_Text"].astype(str).tolist()
        self.scores = frame["LMP_Score"].astype(float).tolist()
        self.tokenizer = tokenizer
    def __len__(self): return len(self.scores)
    def __getitem__(self, i):
        enc = self.tokenizer(self.originals[i], self.simplified[i],
            truncation=True, max_length=MAX_LENGTH, padding="max_length",
            return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.scores[i], dtype=torch.float32)
        return item

class XLMRLoRAForLMP(nn.Module):
    def __init__(self, model_name, lora_config=None, adapter_path=None):
        super().__init__()
        base = AutoModel.from_pretrained(model_name)
        if adapter_path is not None:
            self.encoder = PeftModel.from_pretrained(base, adapter_path)
        else:
            self.encoder = get_peft_model(base, lora_config)
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, 1)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.regressor(self.dropout(out.last_hidden_state[:, 0, :])).squeeze(-1)

def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        pr = sr = np.nan
    else:
        pr, sr = float(pearsonr(y, p).statistic), float(spearmanr(y, p).statistic)
    return {"N": len(y), "Pearson_r": pr, "Spearman_rho": sr,
            "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p)))}

def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    losses, preds, labels_all = [], [], []
    for batch in tqdm(loader, desc="Training" if training else "Evaluating", leave=False):
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        if training: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            pred = model(ids, mask)
            loss = nn.functional.mse_loss(pred, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                if scheduler: scheduler.step()
        losses.append(float(loss.detach().cpu()))
        preds.extend(pred.detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())
    return float(np.mean(losses)), np.asarray(labels_all), np.asarray(preds)

def main():
    set_seed(SEED); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists(): raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = ["Pair_ID","Language","Original_Text","Simplified_Text","LMP_Score","Split"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    df["LMP_Score"] = pd.to_numeric(df["LMP_Score"], errors="coerce")
    df = df.dropna(subset=required).copy()
    df["Original_Text"] = df["Original_Text"].astype(str).str.strip()
    df["Simplified_Text"] = df["Simplified_Text"].astype(str).str.strip()
    if df[["Original_Text","Simplified_Text"]].eq("").any().any(): raise ValueError("Empty text found.")
    if not df["LMP_Score"].between(1,10).all(): raise ValueError("Scores must be 1-10.")
    train = df[df.Split.eq("train")].reset_index(drop=True)
    val = df[df.Split.eq("validation")].reset_index(drop=True)
    test = df[df.Split.eq("test")].reset_index(drop=True)
    if min(len(train),len(val),len(test)) == 0: raise ValueError("All splits must be non-empty.")

    print("XLM-RoBERTa + LoRA (UPDATED AVERAGE DATASET)")
    print(f"Rows: train={len(train)}, validation={len(val)}, test={len(test)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda": print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(LMPDataset(train,tokenizer), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(LMPDataset(val,tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(LMPDataset(test,tokenizer), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=LORA_R,
        lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=["query","value"], bias="none")
    model = XLMRLoRAForLMP(MODEL_NAME, lora_config=config).to(device)
    model.encoder.print_trainable_parameters()
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader)*EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=max(1,int(.1*total_steps)), num_training_steps=total_steps)

    best_rmse, best_epoch, no_improve, history = float("inf"), 0, 0, []
    print("Starting LoRA training...")
    for epoch in range(1,EPOCHS+1):
        print(f"\\nEpoch {epoch}/{EPOCHS}")
        train_loss,_,_ = run_epoch(model,train_loader,device,optimizer,scheduler)
        val_loss,yv,pv = run_epoch(model,val_loader,device)
        pv = np.clip(pv,1,10); vm = metrics(yv,pv); vrmse = vm["RMSE"]
        print(f"Train MSE: {train_loss:.4f} | Validation MSE: {val_loss:.4f} | "
              f"Validation RMSE: {vrmse:.4f} | Validation MAE: {vm['MAE']:.4f}")
        history.append({"Epoch":epoch,"Train_MSE":train_loss,"Validation_MSE":val_loss,
            "Validation_RMSE":vrmse,"Validation_MAE":vm["MAE"],
            "Validation_Pearson":vm["Pearson_r"],"Validation_Spearman":vm["Spearman_rho"]})
        if vrmse < best_rmse:
            best_rmse,best_epoch,no_improve = vrmse,epoch,0
            MODEL_DIR.mkdir(parents=True,exist_ok=True)
            model.encoder.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            torch.save({"regressor_state_dict":model.regressor.state_dict(),
                "best_epoch":best_epoch,"best_validation_rmse":best_rmse,
                "model_name":MODEL_NAME,"max_length":MAX_LENGTH},
                MODEL_DIR/"regression_head.pt")
            print("Best LoRA checkpoint saved.")
        else:
            no_improve += 1; print(f"No validation improvement ({no_improve}/{PATIENCE}).")
        if no_improve >= PATIENCE:
            print("Early stopping triggered."); break

    if best_epoch == 0: raise RuntimeError("No best checkpoint was saved.")
    print(f"Best epoch: {best_epoch}; validation RMSE: {best_rmse:.4f}")
    ckpt = torch.load(MODEL_DIR/"regression_head.pt",map_location=device,weights_only=False)
    model = XLMRLoRAForLMP(MODEL_NAME,adapter_path=MODEL_DIR).to(device)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"]); model.eval()
    _,yt,pt = run_epoch(model,test_loader,device); pt = np.clip(pt,1,10)
    pred_df = test[["Pair_ID","Language","LMP_Score"]].copy()
    pred_df["Predicted_LMP_Score"] = pt
    if "Provisional_Label" in test.columns: pred_df["Provisional_Label"] = test["Provisional_Label"].values
    pred_path = OUTPUT_DIR/"xlmr_lora_updated_average_test_predictions.csv"
    pred_df.to_csv(pred_path,index=False,encoding="utf-8-sig")
    metric_rows = [{"Split":"test","Language":"All",**metrics(yt,pt)}]
    for lang in sorted(test.Language.unique()):
        m = test.Language.eq(lang).to_numpy()
        metric_rows.append({"Split":"test","Language":lang,**metrics(yt[m],pt[m])})
    metric_path = OUTPUT_DIR/"xlmr_lora_updated_average_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metric_path,index=False,encoding="utf-8-sig")
    hist_path = OUTPUT_DIR/"xlmr_lora_updated_average_training_history.csv"
    pd.DataFrame(history).to_csv(hist_path,index=False,encoding="utf-8-sig")
    run_config = {"model":MODEL_NAME,"method":"LoRA","rank":LORA_R,"alpha":LORA_ALPHA,
        "dropout":LORA_DROPOUT,"target_modules":["query","value"],"max_length":MAX_LENGTH,
        "batch_size":BATCH_SIZE,"epochs_requested":EPOCHS,"learning_rate":LEARNING_RATE,
        "patience":PATIENCE,"seed":SEED,"best_epoch":best_epoch,
        "best_validation_rmse":best_rmse,"test_metrics":metrics(yt,pt),
        "label_caveat":"Provisional weighted consensus scores; inter-annotator agreement was poor/negative."}
    with open(OUTPUT_DIR/"xlmr_lora_updated_average_run_config.json","w",encoding="utf-8") as f: json.dump(run_config,f,indent=2)
    print("\\nXLM-RoBERTa + LoRA TEST RESULTS")
    print(pd.DataFrame(metric_rows).round(4).to_string(index=False))
    print("\\nSaved:", MODEL_DIR, pred_path, metric_path, hist_path, sep="\\n")
    print("Test set was not used for training, early stopping, or checkpoint selection.")

if __name__ == "__main__":
    main()

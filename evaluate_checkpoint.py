from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, confusion_matrix
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "RTI_Model_Ready_Dataset_Updated_Average.csv"
RUN_DIR = ROOT / "output" / "mbert_improved_v1" / "checkpoints" / "cls_huber_seed123"
OUT_DIR = ROOT / "output" / "ui_checkpoint_evaluation"
MAX_LENGTH_DEFAULT = 256
DROPOUT_DEFAULT = 0.2

class MBertForLMP(nn.Module):
    def __init__(self, encoder, pooling="cls", dropout=0.2):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**inputs).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.regressor(self.dropout(pooled)).squeeze(-1)

def label(score):
    if score < 6: return "Not Preserved"
    if score < 8: return "Partially Preserved"
    return "Preserved"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    ckpt_path = RUN_DIR / "best_checkpoint.pt"
    encoder_dir = RUN_DIR / "encoder"
    if not ckpt_path.exists() or not encoder_dir.is_dir():
        raise FileNotFoundError("Run folder must contain best_checkpoint.pt and encoder/")

    frame = pd.read_csv(DATA_PATH)
    test = frame[frame["Split"].astype(str).str.lower().eq("test")].copy().reset_index(drop=True)
    if test.empty:
        raise ValueError("No rows with Split='test' found.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pooling = ckpt.get("pooling", "cls")
    dropout = float(ckpt.get("dropout", DROPOUT_DEFAULT))
    max_length = int(ckpt.get("max_length", MAX_LENGTH_DEFAULT))
    encoder = AutoModel.from_pretrained(str(encoder_dir), local_files_only=True)
    model = MBertForLMP(encoder, pooling, dropout)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir), local_files_only=True)

    predictions = []
    with torch.inference_mode():
        for _, row in test.iterrows():
            encoded = tokenizer(
                str(row["Original_Text"]), str(row["Simplified_Text"]),
                truncation=True, max_length=max_length, padding="max_length",
                return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            pred = model(
                encoded["input_ids"], encoded["attention_mask"],
                token_type_ids=encoded.get("token_type_ids")
            ).item()
            predictions.append(float(np.clip(pred, 1, 10)))

    y = test["LMP_Score"].astype(float).to_numpy()
    p = np.asarray(predictions)
    train_mean = frame.loc[frame["Split"].astype(str).str.lower().eq("train"), "LMP_Score"].astype(float).mean()
    baseline = np.full_like(y, train_mean, dtype=float)
    pred_label = [label(x) for x in p]
    true_label = [label(x) for x in y]
    cm = confusion_matrix(true_label, pred_label, labels=["Not Preserved", "Partially Preserved", "Preserved"])

    def safe_corr(fn, a, b):
        if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
            return float("nan")
        return float(fn(a, b).statistic)

    print(f"Checkpoint: {RUN_DIR}")
    print(f"Run: pooling={ckpt.get('pooling')}, loss={ckpt.get('loss')}, seed={ckpt.get('seed')}, best_epoch={ckpt.get('best_epoch')}")
    print(f"Test rows: {len(test)}")
    print("\nMODEL TEST METRICS")
    print(f"Pearson r:  {safe_corr(pearsonr, y, p):.4f}")
    print(f"Spearman ρ: {safe_corr(spearmanr, y, p):.4f}")
    print(f"MAE:        {mean_absolute_error(y, p):.4f}")
    print(f"RMSE:       {np.sqrt(mean_squared_error(y, p)):.4f}")
    print(f"Prediction SD: {np.std(p, ddof=1):.4f} | Target SD: {np.std(y, ddof=1):.4f}")
    print("\nTRAIN-MEAN BASELINE (predict training mean for every test item)")
    print(f"Train mean: {train_mean:.4f}")
    print(f"Baseline MAE:  {mean_absolute_error(y, baseline):.4f}")
    print(f"Baseline RMSE: {np.sqrt(mean_squared_error(y, baseline)):.4f}")
    print("\nPROJECT LABEL METRICS (thresholds are project-defined)")
    print(f"Accuracy: {accuracy_score(true_label, pred_label):.4f}")
    print(f"Macro F1: {f1_score(true_label, pred_label, labels=['Not Preserved','Partially Preserved','Preserved'], average='macro', zero_division=0):.4f}")
    print("Confusion matrix rows=true, columns=predicted; label order:")
    print(["Not Preserved", "Partially Preserved", "Preserved"])
    print(cm)

    out = test[["Pair_ID", "Language", "LMP_Score"]].copy()
    out["Predicted_LMP_Score"] = p
    out["Absolute_Error"] = np.abs(y-p)
    out["True_Project_Label"] = true_label
    out["Predicted_Project_Label"] = pred_label
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "checkpoint_test_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved detailed predictions to: {OUT_DIR / 'checkpoint_test_predictions.csv'}")

if __name__ == "__main__":
    main()

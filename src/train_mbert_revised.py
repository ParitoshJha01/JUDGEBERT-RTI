from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"dataset"/"RTI_Revised_300_Pairs_Two_Assessors_Third_Adjudication.csv"
OUT=ROOT/"output"/"revised_annotations"/"mbert_baseline"
MODEL_NAME="google-bert/bert-base-multilingual-cased"
MAX_LENGTH=256
BATCH_SIZE=2
EPOCHS=12
LR=1e-5
WEIGHT_DECAY=0.01
PATIENCE=3
SEED=42
NUM_WORKERS=0

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def corr(fn,y,p):
    if len(y)<2 or np.std(y)==0 or np.std(p)==0: return float("nan")
    return float(fn(y,p).statistic)

def metrics(y,p):
    return {"N":len(y),"Pearson_r":corr(pearsonr,y,p),"Spearman_rho":corr(spearmanr,y,p),
            "MAE":mean_absolute_error(y,p),"RMSE":mean_squared_error(y,p)**0.5,
            "Prediction_SD":float(np.std(p)),"Target_SD":float(np.std(y))}

class PairDataset(Dataset):
    def __init__(self,frame,tokenizer):
        self.frame=frame.reset_index(drop=True); self.tok=tokenizer
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]
        z=self.tok(str(r.Original_Text),str(r.Simplified_Text),truncation=True,
                   max_length=MAX_LENGTH,padding="max_length",return_tensors="pt")
        item={k:v.squeeze(0) for k,v in z.items()}
        item["labels"]=torch.tensor(float(r.LMP_Score),dtype=torch.float32)
        return item

class MBertRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(MODEL_NAME)
        self.dropout=nn.Dropout(0.2)
        h=self.encoder.config.hidden_size
        self.head=nn.Sequential(nn.Linear(h,h//2),nn.GELU(),nn.Dropout(0.2),nn.Linear(h//2,1))
    def forward(self,input_ids,attention_mask,token_type_ids=None):
        kw={"input_ids":input_ids,"attention_mask":attention_mask}
        if token_type_ids is not None: kw["token_type_ids"]=token_type_ids
        out=self.encoder(**kw)
        return self.head(self.dropout(out.last_hidden_state[:,0,:])).squeeze(-1)

def run(model,loader,device,optimizer=None,scheduler=None):
    train=optimizer is not None
    model.train() if train else model.eval()
    ys=[]; ps=[]; loss_sum=0; count=0
    for b in tqdm(loader,desc="Train" if train else "Evaluate",leave=False):
        ids=b["input_ids"].to(device); mask=b["attention_mask"].to(device)
        types=b.get("token_type_ids")
        if types is not None: types=types.to(device)
        y=b["labels"].to(device)
        if train: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            p=model(ids,mask,types)
            loss=nn.functional.mse_loss(p,y)
            if train:
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                if scheduler is not None: scheduler.step()
        bs=len(y); loss_sum+=float(loss.detach().cpu())*bs; count+=bs
        ys.extend(y.detach().cpu().numpy()); ps.extend(p.detach().cpu().numpy())
    return loss_sum/max(count,1),np.asarray(ys),np.asarray(ps)

def main():
    seed_all(SEED); OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA)
    req=["Pair_ID","Language","Original_Text","Simplified_Text","LMP_Score","Split"]
    if any(c not in df for c in req): raise ValueError("Missing required columns.")
    if len(df)!=300 or not df.Pair_ID.is_unique or df[req].isna().any().any():
        raise ValueError("Expected 300 complete rows with unique Pair_ID.")
    df["LMP_Score"]=pd.to_numeric(df.LMP_Score,errors="raise")
    if not df.LMP_Score.between(1,10).all(): raise ValueError("LMP_Score outside 1–10.")
    train=df[df.Split.eq("train")].reset_index(drop=True)
    val=df[df.Split.eq("validation")].reset_index(drop=True)
    test=df[df.Split.eq("test")].reset_index(drop=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    if device.type=="cuda": print("GPU:",torch.cuda.get_device_name(0))
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
    loaders=[
        DataLoader(PairDataset(train,tokenizer),batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS),
        DataLoader(PairDataset(val,tokenizer),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS),
        DataLoader(PairDataset(test,tokenizer),batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS)]
    tr_loader,va_loader,te_loader=loaders
    model=MBertRegressor().to(device)
    opt=AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    total=len(tr_loader)*EPOCHS
    sched=get_linear_schedule_with_warmup(opt,num_warmup_steps=max(1,int(.1*total)),num_training_steps=total)
    best=float("inf"); best_epoch=0; wait=0; history=[]
    checkpoint=OUT/"best_model.pt"
    for epoch in range(1,EPOCHS+1):
        train_loss,_,_=run(model,tr_loader,device,opt,sched)
        val_loss,yv,pv=run(model,va_loader,device)
        pv=np.clip(pv,1,10); vm=metrics(yv,pv)
        history.append({"Epoch":epoch,"Train_MSE":train_loss,"Validation_MSE":val_loss,**{f"Validation_{k}":v for k,v in vm.items()}})
        print(f"Epoch {epoch:02d}/{EPOCHS} | train MSE {train_loss:.4f} | val RMSE {vm['RMSE']:.4f} | val MAE {vm['MAE']:.4f}")
        if vm["RMSE"]<best:
            best=vm["RMSE"]; best_epoch=epoch; wait=0
            torch.save({"model_state_dict":model.state_dict(),"model_name":MODEL_NAME,
                        "best_epoch":best_epoch,"best_validation_rmse":best,
                        "max_length":MAX_LENGTH,"seed":SEED},checkpoint)
        else:
            wait+=1
            if wait>=PATIENCE: print("Early stopping."); break
    saved=torch.load(checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    _,yt,pt=run(model,te_loader,device); pt=np.clip(pt,1,10)
    pred=test[["Pair_ID","Language","LMP_Score"]].copy()
    pred["Predicted_LMP_Score"]=pt
    if "Provisional_Label" in test: pred["Provisional_Label"]=test.Provisional_Label.values
    pred.to_csv(OUT/"test_predictions.csv",index=False,encoding="utf-8-sig")
    rows=[{"Split":"test","Language":"All",**metrics(yt,pt)}]
    for lang in ["English","Hindi"]:
        m=test.Language.eq(lang).to_numpy()
        rows.append({"Split":"test","Language":lang,**metrics(yt[m],pt[m])})
    pd.DataFrame(rows).to_csv(OUT/"test_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(OUT/"training_history.csv",index=False,encoding="utf-8-sig")
    tokenizer.save_pretrained(OUT/"tokenizer")
    config={"model":MODEL_NAME,"seed":SEED,"max_length":MAX_LENGTH,"batch_size":BATCH_SIZE,
            "epochs_requested":EPOCHS,"learning_rate":LR,"weight_decay":WEIGHT_DECAY,
            "patience":PATIENCE,"best_epoch":best_epoch,"best_validation_rmse":best,
            "test_metrics":rows,"target":"Revised LMP_Score on original 1-10 scale",
            "selection":"Checkpoint selected using validation RMSE; test evaluated once."}
    (OUT/"run_config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print("\nTEST RESULTS\n",pd.DataFrame(rows).round(4).to_string(index=False))
    print("Saved outputs:",OUT)

if __name__=="__main__": main()

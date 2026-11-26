"""
Controlled revised-annotation mBERT experiments.
Default is deliberately limited to CLS + Huber with 3 seeds.
After reviewing these results, add further pooling/loss combinations if justified.
"""
from pathlib import Path
import json, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,AutoModel,get_linear_schedule_with_warmup
from torch.optim import AdamW
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_absolute_error,mean_squared_error
from tqdm import tqdm

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"dataset"/"RTI_Revised_300_Pairs_Two_Assessors_Third_Adjudication.csv"
OUT=ROOT/"output"/"revised_annotations"/"mbert_controlled"
MODEL_NAME="google-bert/bert-base-multilingual-cased"
MAX_LENGTH=256
BATCH_SIZE=2
EPOCHS=12
LR=1e-5
WEIGHT_DECAY=.01
PATIENCE=3
DROPOUT=.2
HUBER_DELTA=1.0
SEEDS=[42,123,2026]
# Start with the most promising prior configuration only.
EXPERIMENTS=[{"pooling":"cls","loss":"huber"}]

def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)
def corr(fn,y,p):
    if len(y)<2 or np.std(y)==0 or np.std(p)==0:return float("nan")
    return float(fn(y,p).statistic)
def metrics(y,p):
    return {"N":len(y),"Pearson_r":corr(pearsonr,y,p),"Spearman_rho":corr(spearmanr,y,p),
            "MAE":mean_absolute_error(y,p),"RMSE":mean_squared_error(y,p)**.5,
            "Prediction_SD":float(np.std(p)),"Target_SD":float(np.std(y))}
class DS(Dataset):
    def __init__(self,f,t):self.f=f.reset_index(drop=True);self.t=t
    def __len__(self):return len(self.f)
    def __getitem__(self,i):
        r=self.f.iloc[i];z=self.t(str(r.Original_Text),str(r.Simplified_Text),truncation=True,
            max_length=MAX_LENGTH,padding="max_length",return_tensors="pt")
        o={k:v.squeeze(0) for k,v in z.items()};o["labels"]=torch.tensor(float(r.LMP_Score),dtype=torch.float32);return o
class Net(nn.Module):
    def __init__(self,pooling):
        super().__init__();self.encoder=AutoModel.from_pretrained(MODEL_NAME);self.pooling=pooling
        h=self.encoder.config.hidden_size
        self.head=nn.Sequential(nn.Dropout(DROPOUT),nn.Linear(h,h//2),nn.GELU(),
                                nn.Dropout(DROPOUT),nn.Linear(h//2,1))
    def forward(self,ids,mask,types=None):
        kw={"input_ids":ids,"attention_mask":mask}
        if types is not None:kw["token_type_ids"]=types
        hidden=self.encoder(**kw).last_hidden_state
        if self.pooling=="cls": pooled=hidden[:,0,:]
        else:
            m=mask.unsqueeze(-1).to(hidden.dtype);pooled=(hidden*m).sum(1)/m.sum(1).clamp(min=1e-9)
        return self.head(pooled).squeeze(-1)
def epoch(model,loader,device,loss_name,opt=None,sched=None):
    training=opt is not None;model.train() if training else model.eval()
    ys=[];ps=[];ls=[];n=0
    for b in tqdm(loader,leave=False,desc="Train" if training else "Eval"):
        ids=b["input_ids"].to(device);mask=b["attention_mask"].to(device)
        types=b.get("token_type_ids")
        if types is not None:types=types.to(device)
        y=b["labels"].to(device)
        if training:opt.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            p=model(ids,mask,types)
            loss=nn.functional.huber_loss(p,y,delta=HUBER_DELTA) if loss_name=="huber" else nn.functional.mse_loss(p,y)
            if training:
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                if sched is not None:sched.step()
        ls.append(float(loss.detach().cpu())*len(y));n+=len(y)
        ys.extend(y.detach().cpu().numpy());ps.extend(p.detach().cpu().numpy())
    return sum(ls)/max(n,1),np.array(ys),np.array(ps)
def one_run(tr,va,te,tok,device,pooling,loss_name,seed):
    set_seed(seed);name=f"{pooling}_{loss_name}_seed{seed}";rd=OUT/"runs"/name;rd.mkdir(parents=True,exist_ok=True)
    tl=DataLoader(DS(tr,tok),batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    vl=DataLoader(DS(va,tok),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    el=DataLoader(DS(te,tok),batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
    model=Net(pooling).to(device);opt=AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    total=len(tl)*EPOCHS;sched=get_linear_schedule_with_warmup(opt,max(1,int(.1*total)),total)
    best=float("inf");wait=0;best_ep=0;hist=[];ck=rd/"best_model.pt"
    for ep in range(1,EPOCHS+1):
        trloss,_,_=epoch(model,tl,device,loss_name,opt,sched)
        vlss,y,p=epoch(model,vl,device,loss_name);p=np.clip(p,1,10);vm=metrics(y,p)
        hist.append({"Run":name,"Epoch":ep,"Train_Loss":trloss,"Validation_Loss":vlss,**{f"Validation_{k}":v for k,v in vm.items()}})
        print(f"{name} | epoch {ep}/{EPOCHS} | val RMSE {vm['RMSE']:.4f} | val MAE {vm['MAE']:.4f}")
        if vm["RMSE"]<best:
            best=vm["RMSE"];best_ep=ep;wait=0
            torch.save({"model_state_dict":model.state_dict(),"best_epoch":ep,"pooling":pooling,"loss":loss_name,"seed":seed},ck)
        else:
            wait+=1
            if wait>=PATIENCE:break
    model.load_state_dict(torch.load(ck,map_location=device,weights_only=False)["model_state_dict"])
    _,y,p=epoch(model,el,device,loss_name);p=np.clip(p,1,10)
    pred=te[["Pair_ID","Language","LMP_Score"]].copy();pred["Predicted_LMP_Score"]=p
    pred.to_csv(rd/"test_predictions.csv",index=False,encoding="utf-8-sig")
    row={"Run":name,"Pooling":pooling,"Loss":loss_name,"Seed":seed,"Best_Epoch":best_ep,
         "Best_Validation_RMSE":best,**metrics(y,p)}
    langs=[]
    for lang in ["English","Hindi"]:
        m=te.Language.eq(lang).to_numpy();langs.append({"Run":name,"Language":lang,**metrics(y[m],p[m])})
    pd.DataFrame(hist).to_csv(rd/"history.csv",index=False,encoding="utf-8-sig")
    return row,langs,hist
def main():
    OUT.mkdir(parents=True,exist_ok=True);df=pd.read_csv(DATA)
    req=["Pair_ID","Language","Original_Text","Simplified_Text","LMP_Score","Split"]
    if any(c not in df for c in req) or len(df)!=300 or not df.Pair_ID.is_unique:raise ValueError("Dataset schema/row/ID check failed.")
    if df[req].isna().any().any():raise ValueError("Missing required values.")
    df["LMP_Score"]=pd.to_numeric(df.LMP_Score,errors="raise")
    if not df.LMP_Score.between(1,10).all():raise ValueError("Invalid score range.")
    tr=df[df.Split.eq("train")].reset_index(drop=True);va=df[df.Split.eq("validation")].reset_index(drop=True);te=df[df.Split.eq("test")].reset_index(drop=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    if device.type=="cuda":print("GPU:",torch.cuda.get_device_name(0))
    tok=AutoTokenizer.from_pretrained(MODEL_NAME);rows=[];langs=[];histories=[]
    for exp in EXPERIMENTS:
        for seed in SEEDS:
            r,l,h=one_run(tr,va,te,tok,device,exp["pooling"],exp["loss"],seed);rows.append(r);langs.extend(l);histories.extend(h)
            pd.DataFrame(rows).to_csv(OUT/"run_metrics.csv",index=False,encoding="utf-8-sig")
            pd.DataFrame(langs).to_csv(OUT/"language_metrics.csv",index=False,encoding="utf-8-sig")
            pd.DataFrame(histories).to_csv(OUT/"training_histories.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT/"run_metrics.csv",index=False,encoding="utf-8-sig")
    (OUT/"config.json").write_text(json.dumps({"model":MODEL_NAME,"seeds":SEEDS,"experiments":EXPERIMENTS,
        "batch_size":BATCH_SIZE,"epochs":EPOCHS,"learning_rate":LR,"selection":"Validation RMSE only",
        "test_note":"Test evaluated once per seed; do not select a seed/configuration by test performance."},indent=2),encoding="utf-8")
    print(pd.DataFrame(rows).round(4).to_string(index=False));print("Saved:",OUT)
if __name__=="__main__":main()

from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import AutoModel, AutoTokenizer

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"dataset"/"RTI_Model_Ready_Dataset_Updated_Average.csv"
RUN=ROOT/"output"/"mbert_improved_v1"/"checkpoints"/"cls_huber_seed123" # set to app's checkpoint

class MBertForLMP(nn.Module):
    def __init__(self, encoder, pooling="cls", dropout=0.2):
        super().__init__(); self.encoder=encoder; self.pooling=pooling
        h=encoder.config.hidden_size
        self.dropout=nn.Dropout(dropout)
        self.regressor=nn.Sequential(nn.Linear(h,h//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(h//2,1))
    def forward(self,input_ids,attention_mask,token_type_ids=None):
        kw={"input_ids":input_ids,"attention_mask":attention_mask}
        if token_type_ids is not None: kw["token_type_ids"]=token_type_ids
        hidden=self.encoder(**kw).last_hidden_state
        if self.pooling=="cls": pooled=hidden[:,0,:]
        else:
            mask=attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled=(hidden*mask).sum(1)/mask.sum(1).clamp(min=1e-9)
        return self.regressor(self.dropout(pooled)).squeeze(-1)

def corr(fn,y,p):
    if len(y)<2 or np.std(y)==0 or np.std(p)==0:return float("nan")
    return float(fn(y,p).statistic)

def main():
    if not DATA.exists(): raise FileNotFoundError(f"Dataset not found: {DATA}. Put CSV beside this script or edit DATA.")
    cp=RUN/"best_checkpoint.pt"; encdir=RUN/"encoder"
    if not cp.exists() or not encdir.is_dir(): raise FileNotFoundError(f"Expected best_checkpoint.pt and encoder/ in {RUN}")
    df=pd.read_csv(DATA)
    ck=torch.load(cp,map_location="cpu",weights_only=False)
    pooling=ck.get("pooling","cls"); dropout=float(ck.get("dropout",.2)); maxlen=int(ck.get("max_length",256))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder=AutoModel.from_pretrained(str(encdir),local_files_only=True)
    model=MBertForLMP(encoder,pooling,dropout); model.load_state_dict(ck["model_state_dict"],strict=True); model.to(device).eval()
    tok=AutoTokenizer.from_pretrained(str(encdir),local_files_only=True)
    trainmean=pd.to_numeric(df.loc[df.Split.astype(str).str.lower()=="train","LMP_Score"]).mean()
    outputs=[]; summaries=[]
    print(f"Checkpoint: {RUN}\\nPooling={pooling}, loss={ck.get('loss')}, seed={ck.get('seed')}\\nTrain-mean baseline={trainmean:.4f}\\n")
    for split in ["train","validation","test"]:
        part=df[df.Split.astype(str).str.lower()==split].copy().reset_index(drop=True)
        if part.empty: continue
        preds=[]
        with torch.inference_mode():
            for _,r in part.iterrows():
                x=tok(str(r.Original_Text),str(r.Simplified_Text),truncation=True,max_length=maxlen,padding="max_length",return_tensors="pt")
                x={k:v.to(device) for k,v in x.items()}
                preds.append(float(np.clip(model(x["input_ids"],x["attention_mask"],token_type_ids=x.get("token_type_ids")).item(),1,10)))
        y=pd.to_numeric(part.LMP_Score).to_numpy(float); pred=np.array(preds); base=np.full_like(y,trainmean)
        print(f"=== {split.upper()} n={len(y)} ===")
        print(f"Human mean/SD/range: {y.mean():.3f}/{y.std(ddof=1):.3f}/{y.min():.2f}-{y.max():.2f}")
        print(f"Pred mean/SD/range:  {pred.mean():.3f}/{pred.std(ddof=1):.3f}/{pred.min():.2f}-{pred.max():.2f}")
        print(f"Pearson {corr(pearsonr,y,pred):.3f} | Spearman {corr(spearmanr,y,pred):.3f} | MAE {mean_absolute_error(y,pred):.3f} | RMSE {np.sqrt(mean_squared_error(y,pred)):.3f}")
        print(f"Train-mean baseline MAE {mean_absolute_error(y,base):.3f} | RMSE {np.sqrt(mean_squared_error(y,base)):.3f}\\n")
        part["Predicted_LMP_Score"]=pred; part["Absolute_Error"]=np.abs(y-pred); part["Prediction_Split"]=split; outputs.append(part)
        summaries.append({"split":split,"n":len(y),"human_mean":y.mean(),"human_sd":y.std(ddof=1),"pred_mean":pred.mean(),"pred_sd":pred.std(ddof=1),"pearson":corr(pearsonr,y,pred),"spearman":corr(spearmanr,y,pred),"mae":mean_absolute_error(y,pred),"rmse":np.sqrt(mean_squared_error(y,pred)),"baseline_mae":mean_absolute_error(y,base),"baseline_rmse":np.sqrt(mean_squared_error(y,base))})
    out=ROOT/"output"/"ui_checkpoint_diagnostic"; out.mkdir(parents=True,exist_ok=True)
    pd.concat(outputs,ignore_index=True).to_csv(out/"all_split_predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(out/"split_metrics.csv",index=False)
    print("Saved to",out)

if __name__=="__main__": main()

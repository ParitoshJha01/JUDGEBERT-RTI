from pathlib import Path
import numpy as np
import pandas as pd
import torch
from bert_score import score as bertscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "RTI_Revised_300_Pairs_Two_Assessors_Third_Adjudication.csv"
OUT = ROOT / "output" / "revised_annotations" / "bertscore"
MODEL_TYPE = "xlm-roberta-base"
NUM_LAYERS = 12
BATCH_SIZE = 4

def corr(fn,y,p):
    if len(y)<2 or np.std(y)==0 or np.std(p)==0: return float("nan")
    return float(fn(y,p).statistic)

def get_metrics(frame,split,language):
    y=frame.LMP_Score.to_numpy(float); p=frame.Predicted_LMP_Score.to_numpy(float)
    return {"Split":split,"Language":language,"N":len(frame),
            "Pearson_r":corr(pearsonr,y,p),"Spearman_rho":corr(spearmanr,y,p),
            "MAE":mean_absolute_error(y,p),"RMSE":mean_squared_error(y,p)**0.5,
            "Prediction_SD":float(np.std(p)),"Target_SD":float(np.std(y))}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(DATA)
    req=["Pair_ID","Language","Original_Text","Simplified_Text","LMP_Score","Split"]
    if any(c not in df for c in req): raise ValueError("Required dataset columns are missing.")
    if df[req].isna().any().any() or len(df)!=300 or not df.Pair_ID.is_unique:
        raise ValueError("Dataset has missing required values or unexpected row/ID count.")
    df["LMP_Score"]=pd.to_numeric(df.LMP_Score,errors="raise")
    if not df.LMP_Score.between(1,10).all(): raise ValueError("Scores must be in 1–10.")
    device="cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device=="cuda": print("GPU:",torch.cuda.get_device_name(0))
    # Simplified text is candidate; original legal text is reference.
    _,_,f1=bertscore(df.Simplified_Text.astype(str).tolist(),
                     df.Original_Text.astype(str).tolist(),
                     model_type=MODEL_TYPE,num_layers=NUM_LAYERS,
                     batch_size=BATCH_SIZE,device=device,verbose=True,
                     rescale_with_baseline=False)
    df["BERTScore_F1"]=f1.cpu().numpy()
    train=df.Split.eq("train")
    calibrator=LinearRegression().fit(df.loc[train,["BERTScore_F1"]],df.loc[train,"LMP_Score"])
    df["Predicted_LMP_Score"]=calibrator.predict(df[["BERTScore_F1"]]).clip(1,10)
    rows=[]
    for split in ["train","validation","test"]:
        part=df[df.Split.eq(split)]
        rows.append(get_metrics(part,split,"All"))
        for lang in ["English","Hindi"]:
            sub=part[part.Language.eq(lang)]
            rows.append(get_metrics(sub,split,lang))
    cols=["Pair_ID","Language","Split","LMP_Score","BERTScore_F1","Predicted_LMP_Score"]
    if "Provisional_Label" in df: cols.append("Provisional_Label")
    df[cols].to_csv(OUT/"predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT/"metrics.csv",index=False,encoding="utf-8-sig")
    print("Calibration fitted on training split only.")
    print(pd.DataFrame(rows).query("Split == 'test'").round(4).to_string(index=False))
    print("Saved to",OUT)

if __name__=="__main__": main()

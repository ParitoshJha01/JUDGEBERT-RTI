from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "RTI_Revised_300_Pairs_Two_Assessors_Third_Adjudication.csv"
OUT = ROOT / "output" / "revised_annotations" / "sentence_transformer"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 8

def corr(fn, y, p):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(fn(y, p).statistic)

def metrics(frame, split, language):
    y = frame.LMP_Score.to_numpy(float)
    p = frame.Predicted_LMP_Score.to_numpy(float)
    return {"Split": split, "Language": language, "N": len(frame),
            "Pearson_r": corr(pearsonr,y,p), "Spearman_rho": corr(spearmanr,y,p),
            "MAE": mean_absolute_error(y,p),
            "RMSE": mean_squared_error(y,p)**0.5,
            "Prediction_SD": float(np.std(p)), "Target_SD": float(np.std(y))}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    required = ["Pair_ID","Language","Original_Text","Simplified_Text","LMP_Score","Split"]
    if any(c not in df for c in required): raise ValueError("Required dataset columns are missing.")
    if df[required].isna().any().any(): raise ValueError("Missing required values.")
    if len(df) != 300 or not df.Pair_ID.is_unique: raise ValueError("Expected 300 unique pairs.")
    if not df.LMP_Score.between(1,10).all(): raise ValueError("Scores must be in 1–10.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda": print("GPU:", torch.cuda.get_device_name(0))
    model = SentenceTransformer(MODEL_NAME, device=device)
    n = len(df)
    texts = df.Original_Text.astype(str).tolist() + df.Simplified_Text.astype(str).tolist()
    emb = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    df["Cosine_Similarity"] = np.sum(emb[:n] * emb[n:], axis=1)
    train = df.Split.eq("train")
    calibrator = LinearRegression().fit(df.loc[train,["Cosine_Similarity"]],df.loc[train,"LMP_Score"])
    df["Predicted_LMP_Score"] = calibrator.predict(df[["Cosine_Similarity"]]).clip(1,10)
    rows=[]
    for split in ["train","validation","test"]:
        part=df[df.Split.eq(split)]
        rows.append(metrics(part,split,"All"))
        for lang in ["English","Hindi"]:
            sub=part[part.Language.eq(lang)]
            rows.append(metrics(sub,split,lang))
    pred_cols=["Pair_ID","Language","Split","LMP_Score","Cosine_Similarity","Predicted_LMP_Score"]
    if "Provisional_Label" in df: pred_cols.append("Provisional_Label")
    df[pred_cols].to_csv(OUT/"predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(OUT/"metrics.csv",index=False,encoding="utf-8-sig")
    print(pd.DataFrame(rows).query("Split == 'test'").round(4).to_string(index=False))
    print("Saved to",OUT)

if __name__=="__main__": main()

"""Generate validation-set predictions and a threshold sweep for JUDGEBERT v4.

Run from the project root:
python src/evaluate_v4_validation.py --data dataset/RTI_v4_Critical_Change_Training_Ready.csv
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from train_mbert_v4_multitask_ready import PairDataset, MultiTaskModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/RTI_v4_Critical_Change_Training_Ready.csv")
    ap.add_argument("--output", default="output/mbert_v4_multitask")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output)
    ckpt_path = out_dir / "best_v4_multitask.pt"
    encoder_dir = out_dir / "encoder"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not encoder_dir.exists():
        raise FileNotFoundError(f"Saved encoder not found: {encoder_dir}")

    df = pd.read_csv(data_path)
    df["Critical_Change"] = pd.to_numeric(df["Critical_Change"], errors="coerce")
    val = df[df["Split"].astype(str).str.strip().str.lower().eq("validation")].copy()
    if "Annotation_Status" in val.columns:
        unresolved = val["Annotation_Status"].astype(str).str.strip().str.lower().eq("needs review")
        val.loc[unresolved, "Critical_Change"] = np.nan
    val = val.reset_index(drop=True)
    val_known = val[val["Critical_Change"].isin([0, 1])].copy()
    if val_known.empty or val_known["Critical_Change"].nunique() < 2:
        raise ValueError("Validation set needs reviewed examples from both classes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
    model = MultiTaskModel(AutoModel.from_pretrained(encoder_dir)).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(PairDataset(val_known, tokenizer), batch_size=args.batch_size, shuffle=False)
    probs, lmp_preds = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            token_types = batch.get("token_type_ids")
            token_types = token_types.to(device) if token_types is not None else None
            lmp, logits = model(ids, mask, token_types)
            lmp_preds.extend(np.clip(lmp.cpu().numpy(), 1, 10))
            probs.extend(torch.sigmoid(logits).cpu().numpy())

    val_known["Predicted_LMP_Score"] = lmp_preds
    val_known["Critical_Change_Probability"] = probs
    val_known["Actual_Critical_Change"] = val_known["Critical_Change"].astype(int)
    val_known.to_csv(out_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")

    y = val_known["Actual_Critical_Change"].to_numpy()
    p = val_known["Critical_Change_Probability"].to_numpy()
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.025), 3):
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": float(threshold),
            "N": int(len(y)),
            "Accuracy": float(accuracy_score(y, pred)),
            "Precision": float(precision_score(y, pred, zero_division=0)),
            "Recall": float(recall_score(y, pred, zero_division=0)),
            "F1": float(f1_score(y, pred, zero_division=0)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
        })
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / "validation_threshold_sweep.csv", index=False, encoding="utf-8-sig")

    best = sweep.sort_values(["F1", "Precision", "threshold"], ascending=[False, False, True]).iloc[0]
    report = (
        "VALIDATION-ONLY THRESHOLD ANALYSIS\n"
        f"Validation labeled N: {len(y)}; positives: {int(y.sum())}; negatives: {int((y==0).sum())}\n"
        f"Best validation F1 threshold (exploratory): {best['threshold']:.3f}\n"
        f"At this threshold: F1={best['F1']:.3f}, Precision={best['Precision']:.3f}, "
        f"Recall={best['Recall']:.3f}, Accuracy={best['Accuracy']:.3f}\n\n"
        "Do not use the validation-selected threshold to claim test performance until evaluating the held-out test set once.\n"
        "Threshold selection is unstable with a small validation set; report the threshold and validation metrics transparently.\n"
    )
    (out_dir / "validation_threshold_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print("Saved validation predictions, threshold sweep, and report to", out_dir.resolve())

if __name__ == "__main__":
    main()

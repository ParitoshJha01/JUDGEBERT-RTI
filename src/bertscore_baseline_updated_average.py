"""
BERTScore baseline for the RTI JUDGEBERT project.

Run from the project root:
    python src/bertscore_baseline.py

Expected dataset:
    dataset/RTI_Model_Ready_Dataset_Updated_Average.csv

Outputs:
    output/bertscore_updated_average_predictions.csv
    output/bertscore_updated_average_metrics.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from bert_score import score as bertscore
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "RTI_Model_Ready_Dataset_Updated_Average.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

# XLM-RoBERTa-base is multilingual and more manageable than the large variant.
MODEL_TYPE = "xlm-roberta-base"
NUM_LAYERS = 12
BATCH_SIZE = 8

REQUIRED_COLUMNS = [
    "Pair_ID", "Language", "Original_Text",
    "Simplified_Text", "LMP_Score", "Split"
]


def safe_correlation(function, y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(function(y_true, y_pred).statistic)


def calculate_metrics(frame, split_name, language_name):
    y_true = frame["LMP_Score"].to_numpy(dtype=float)
    y_pred = frame["Predicted_LMP_Score"].to_numpy(dtype=float)

    return {
        "Split": split_name,
        "Language": language_name,
        "N": len(frame),
        "Pearson_r": safe_correlation(pearsonr, y_true, y_pred),
        "Spearman_rho": safe_correlation(spearmanr, y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place the CSV inside the project's dataset folder."
        )

    df = pd.read_csv(DATA_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["LMP_Score"] = pd.to_numeric(df["LMP_Score"], errors="coerce")
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    df["Original_Text"] = df["Original_Text"].astype(str).str.strip()
    df["Simplified_Text"] = df["Simplified_Text"].astype(str).str.strip()

    if (df["Original_Text"] == "").any() or (df["Simplified_Text"] == "").any():
        raise ValueError("Empty original/simplified text found.")
    if not df["LMP_Score"].between(1, 10).all():
        raise ValueError("LMP_Score must be between 1 and 10.")
    if not set(df["Split"].unique()).issubset({"train", "validation", "test"}):
        raise ValueError("Unexpected split value found.")

    print("=" * 65)
    print("BERTSCORE BASELINE — UPDATED AVERAGE DATASET")
    print("=" * 65)
    print(f"Rows loaded: {len(df)}")
    print(df.groupby(["Language", "Split"]).size().to_string())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # BERTScore compares candidate text to reference text.
    # Here, simplified legal clauses are candidates and original clauses
    # are references.
    candidates = df["Simplified_Text"].tolist()
    references = df["Original_Text"].tolist()

    print(f"\nCalculating BERTScore with {MODEL_TYPE}...")
    print("The first run downloads the model and may take several minutes.")

    precision, recall, f1 = bertscore(
        candidates,
        references,
        model_type=MODEL_TYPE,
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE,
        device=device,
        verbose=True,
        rescale_with_baseline=False,
    )

    df["BERTScore_Precision"] = precision.cpu().numpy()
    df["BERTScore_Recall"] = recall.cpu().numpy()
    df["BERTScore_F1"] = f1.cpu().numpy()

    # Fit calibration on training labels only; do not fit on validation/test.
    train_mask = df["Split"].eq("train")
    if train_mask.sum() < 2:
        raise ValueError("At least two training examples are required.")

    calibrator = LinearRegression()
    calibrator.fit(
        df.loc[train_mask, ["BERTScore_F1"]],
        df.loc[train_mask, "LMP_Score"],
    )

    df["Predicted_LMP_Score"] = calibrator.predict(
        df[["BERTScore_F1"]]
    ).clip(1, 10)

    print("\nCalibration fitted using training data only.")
    print(
        f"Linear calibration: score = "
        f"{calibrator.coef_[0]:.4f} * BERTScore_F1 + "
        f"{calibrator.intercept_:.4f}"
    )

    metric_rows = []
    for split_name in ["train", "validation", "test"]:
        split_frame = df[df["Split"] == split_name]

        if len(split_frame) > 0:
            metric_rows.append(
                calculate_metrics(split_frame, split_name, "All")
            )

        for language in sorted(df["Language"].unique()):
            language_frame = split_frame[
                split_frame["Language"] == language
            ]
            if len(language_frame) > 0:
                metric_rows.append(
                    calculate_metrics(
                        language_frame, split_name, language
                    )
                )

    metrics = pd.DataFrame(metric_rows)

    prediction_columns = [
        "Pair_ID", "Language", "Split", "LMP_Score",
        "BERTScore_Precision", "BERTScore_Recall",
        "BERTScore_F1", "Predicted_LMP_Score"
    ]
    if "Provisional_Label" in df.columns:
        prediction_columns.append("Provisional_Label")

    predictions_path = OUTPUT_DIR / "bertscore_updated_average_predictions.csv"
    metrics_path = OUTPUT_DIR / "bertscore_updated_average_metrics.csv"

    df[prediction_columns].to_csv(
        predictions_path, index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print("BERTSCORE RESULTS")
    print("=" * 65)
    print(metrics.round(4).to_string(index=False))

    print("\nFiles saved:")
    print(predictions_path)
    print(metrics_path)

    print("\nImportant:")
    print("- Simplified clauses are candidates; original clauses are references.")
    print("- Calibration was fitted on TRAIN only.")
    print("- Use TEST metrics as the held-out baseline result.")
    print("- Human consensus scores are provisional; interpret results cautiously.")


if __name__ == "__main__":
    main()

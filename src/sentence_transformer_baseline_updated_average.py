"""
Sentence Transformer baseline for the RTI JUDGEBERT project.

Run from the project root:
    python src/sentence_transformer_baseline.py

Expected dataset:
    dataset/RTI_Model_Ready_Dataset_Updated_Average.csv

Outputs:
    output/sentence_transformer_updated_average_predictions.csv
    output/sentence_transformer_updated_average_metrics.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


# -------------------- Configuration --------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "RTI_Model_Ready_Dataset_Updated_Average.csv"
OUTPUT_DIR = PROJECT_ROOT / "output"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BATCH_SIZE = 16
RANDOM_SEED = 42

REQUIRED_COLUMNS = [
    "Pair_ID", "Language", "Original_Text",
    "Simplified_Text", "LMP_Score", "Split"
]


def safe_correlation(function, y_true, y_pred):
    """Return NaN if correlation cannot be calculated."""
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(function(y_true, y_pred).statistic)


def calculate_metrics(frame, score_column, prediction_column, split_name, language_name):
    y_true = frame[score_column].to_numpy(dtype=float)
    y_pred = frame[prediction_column].to_numpy(dtype=float)

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
    print("SENTENCE TRANSFORMER BASELINE - UPDATED AVERAGE DATASET")
    print("=" * 65)
    print(f"Rows loaded: {len(df)}")
    print(df.groupby(["Language", "Split"]).size().to_string())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading model: {MODEL_NAME}")
    print("The first run downloads the model and may take a few minutes.")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # Encode all examples with a fixed pretrained model.
    # No labels are used during embedding generation.
    all_texts = (
        df["Original_Text"].tolist()
        + df["Simplified_Text"].tolist()
    )

    print("\nGenerating sentence embeddings...")
    embeddings = model.encode(
        all_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    n = len(df)
    original_embeddings = embeddings[:n]
    simplified_embeddings = embeddings[n:]

    # Cosine similarity is the dot product because embeddings are normalized.
    cosine_scores = np.sum(
        original_embeddings * simplified_embeddings, axis=1
    )
    df["Cosine_Similarity"] = cosine_scores

    # Fit a simple linear calibration using TRAINING labels only.
    # This maps cosine similarity onto the project's 1-10 LMP score scale.
    train_mask = df["Split"].eq("train")
    if train_mask.sum() < 2:
        raise ValueError("At least two training examples are required.")

    calibrator = LinearRegression()
    calibrator.fit(
        df.loc[train_mask, ["Cosine_Similarity"]],
        df.loc[train_mask, "LMP_Score"],
    )

    df["Predicted_LMP_Score"] = calibrator.predict(
        df[["Cosine_Similarity"]]
    )
    # Keep predictions within the defined score scale.
    df["Predicted_LMP_Score"] = df["Predicted_LMP_Score"].clip(1, 10)

    print("\nCalibration fitted using training data only.")
    print(f"Linear calibration: score = "
          f"{calibrator.coef_[0]:.4f} * cosine + "
          f"{calibrator.intercept_:.4f}")

    # Evaluate validation and test only; training metrics are included
    # as a diagnostic, not as generalization performance.
    metric_rows = []
    for split_name in ["train", "validation", "test"]:
        split_frame = df[df["Split"] == split_name]

        metric_rows.append(
            calculate_metrics(
                split_frame, "LMP_Score", "Predicted_LMP_Score",
                split_name, "All"
            )
        )

        for language in sorted(df["Language"].unique()):
            language_frame = split_frame[
                split_frame["Language"] == language
            ]
            if len(language_frame) > 0:
                metric_rows.append(
                    calculate_metrics(
                        language_frame, "LMP_Score",
                        "Predicted_LMP_Score", split_name, language
                    )
                )

    metrics = pd.DataFrame(metric_rows)

    prediction_columns = [
        "Pair_ID", "Language", "Split", "LMP_Score",
        "Cosine_Similarity", "Predicted_LMP_Score"
    ]
    if "Provisional_Label" in df.columns:
        prediction_columns.append("Provisional_Label")

    predictions_path = OUTPUT_DIR / "sentence_transformer_updated_average_predictions.csv"
    metrics_path = OUTPUT_DIR / "sentence_transformer_updated_average_metrics.csv"

    df[prediction_columns].to_csv(
        predictions_path, index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 65)
    print("BASELINE RESULTS")
    print("=" * 65)
    print(metrics.round(4).to_string(index=False))

    print("\nFiles saved:")
    print(predictions_path)
    print(metrics_path)

    print("\nImportant:")
    print("- Report TEST metrics as the final held-out baseline result.")
    print("- Validation metrics are for model-selection decisions.")
    print("- Calibration was fitted on TRAIN only.")
    print("- Scores are provisional human consensus labels; interpret results cautiously.")


if __name__ == "__main__":
    main()

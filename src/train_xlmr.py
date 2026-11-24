"""
Fine-tune XLM-RoBERTa-base for legal meaning preservation (LMP) regression.

Input: original legal clause + simplified clause.
Target: provisional weighted consensus LMP score (1-10).

Run from the project root:
    python src/train_xlmr.py

Expected:
    dataset/RTI_Model_Ready_Dataset_Grouped_Split.csv

Outputs:
    output/xlmr_best_model/
    output/xlmr_test_predictions.csv
    output/xlmr_metrics.csv
    output/xlmr_training_history.csv
"""

from pathlib import Path
import json
import random

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


# ---------------- Configuration ----------------

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "RTI_Model_Ready_Dataset_Grouped_Split.csv"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "xlmr_best_model"

MODEL_NAME = "FacebookAI/xlm-roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 3
RANDOM_SEED = 42
NUM_WORKERS = 0  # Reliable on Windows


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LMPDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.originals = frame["Original_Text"].astype(str).tolist()
        self.simplified = frame["Simplified_Text"].astype(str).tolist()
        self.scores = frame["LMP_Score"].astype(float).tolist()
        self.pair_ids = frame["Pair_ID"].astype(str).tolist()
        self.languages = frame["Language"].astype(str).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.originals[index],
            self.simplified[index],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.scores[index], dtype=torch.float32)
        return item


class XLMRForLMP(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # XLM-R's first token representation ([CLS]-equivalent).
        pooled = outputs.last_hidden_state[:, 0, :]
        prediction = self.regressor(self.dropout(pooled)).squeeze(-1)
        return prediction


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = np.nan
        spearman = np.nan
    else:
        pearson = float(pearsonr(y_true, y_pred).statistic)
        spearman = float(spearmanr(y_true, y_pred).statistic)

    return {
        "N": len(y_true),
        "Pearson_r": pearson,
        "Spearman_rho": spearman,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    losses = []

    all_predictions = []
    all_labels = []

    progress = tqdm(loader, desc="Training" if training else "Evaluating", leave=False)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(input_ids, attention_mask)
            loss = nn.functional.mse_loss(predictions, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        losses.append(float(loss.detach().cpu()))
        all_predictions.extend(predictions.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    return float(np.mean(losses)), np.array(all_labels), np.array(all_predictions)


def main():
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place the model-ready CSV in the dataset folder."
        )

    df = pd.read_csv(DATA_PATH)
    required = [
        "Pair_ID", "Language", "Original_Text",
        "Simplified_Text", "LMP_Score", "Split"
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["LMP_Score"] = pd.to_numeric(df["LMP_Score"], errors="coerce")
    df = df.dropna(subset=required).copy()
    df["Original_Text"] = df["Original_Text"].astype(str).str.strip()
    df["Simplified_Text"] = df["Simplified_Text"].astype(str).str.strip()

    if df[["Original_Text", "Simplified_Text"]].eq("").any().any():
        raise ValueError("Empty text found.")
    if not df["LMP_Score"].between(1, 10).all():
        raise ValueError("LMP scores must be between 1 and 10.")

    train_df = df[df["Split"] == "train"].reset_index(drop=True)
    val_df = df[df["Split"] == "validation"].reset_index(drop=True)
    test_df = df[df["Split"] == "test"].reset_index(drop=True)

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Train, validation and test splits must all be non-empty.")

    print("=" * 65)
    print("XLM-ROBERTA LMP REGRESSION")
    print("=" * 65)
    print(f"Rows: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    print("Input: original clause paired with simplified clause")
    print("Target: provisional weighted consensus LMP score (1-10)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader = DataLoader(
        LMPDataset(train_df, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        LMPDataset(val_df, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        LMPDataset(test_df, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = XLMRForLMP(MODEL_NAME).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    best_val_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print("\nStarting fine-tuning...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, _, _ = run_epoch(
            model, train_loader, device, optimizer, scheduler
        )
        val_loss, val_y, val_pred_raw = run_epoch(
            model, val_loader, device
        )
        val_pred = np.clip(val_pred_raw, 1, 10)
        val_metrics = calculate_metrics(val_y, val_pred)
        val_rmse = val_metrics["RMSE"]

        print(
            f"Train MSE: {train_loss:.4f} | "
            f"Validation MSE: {val_loss:.4f} | "
            f"Validation RMSE: {val_rmse:.4f} | "
            f"Validation MAE: {val_metrics['MAE']:.4f}"
        )

        history.append({
            "Epoch": epoch,
            "Train_MSE": train_loss,
            "Validation_MSE": val_loss,
            "Validation_RMSE": val_rmse,
            "Validation_MAE": val_metrics["MAE"],
            "Validation_Pearson": val_metrics["Pearson_r"],
            "Validation_Spearman": val_metrics["Spearman_rho"],
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            epochs_without_improvement = 0

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            torch.save(
                {
                    "regressor_state_dict": model.regressor.state_dict(),
                    "dropout_probability": 0.1,
                    "model_name": MODEL_NAME,
                    "max_length": MAX_LENGTH,
                    "best_epoch": best_epoch,
                    "best_validation_rmse": best_val_rmse,
                },
                MODEL_DIR / "regression_head.pt",
            )
            print("Best checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement "
                f"({epochs_without_improvement}/{PATIENCE})."
            )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Restore the checkpoint selected using validation RMSE.
    print(f"\nBest epoch: {best_epoch}; validation RMSE: {best_val_rmse:.4f}")
    checkpoint = torch.load(
        MODEL_DIR / "regression_head.pt",
        map_location=device,
        weights_only=False,
    )
    model.encoder = AutoModel.from_pretrained(MODEL_DIR).to(device)
    model.regressor.load_state_dict(checkpoint["regressor_state_dict"])
    model.eval()

    # Evaluate the held-out test set once after model selection.
    _, test_y, test_pred_raw = run_epoch(model, test_loader, device)
    test_pred = np.clip(test_pred_raw, 1, 10)
    overall = calculate_metrics(test_y, test_pred)

    test_predictions = test_df[
        ["Pair_ID", "Language", "LMP_Score"]
    ].copy()
    test_predictions["Predicted_LMP_Score"] = test_pred

    if "Provisional_Label" in test_df.columns:
        test_predictions["Provisional_Label"] = test_df["Provisional_Label"].values

    prediction_path = OUTPUT_DIR / "xlmr_test_predictions.csv"
    test_predictions.to_csv(
        prediction_path, index=False, encoding="utf-8-sig"
    )

    metric_rows = [{"Split": "test", "Language": "All", **overall}]
    for language in sorted(test_df["Language"].unique()):
        mask = test_df["Language"].eq(language).to_numpy()
        language_metrics = calculate_metrics(test_y[mask], test_pred[mask])
        metric_rows.append({
            "Split": "test",
            "Language": language,
            **language_metrics,
        })

    metrics_path = OUTPUT_DIR / "xlmr_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(
        metrics_path, index=False, encoding="utf-8-sig"
    )

    history_path = OUTPUT_DIR / "xlmr_training_history.csv"
    pd.DataFrame(history).to_csv(
        history_path, index=False, encoding="utf-8-sig"
    )

    config = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "seed": RANDOM_SEED,
        "best_epoch": best_epoch,
        "best_validation_rmse": best_val_rmse,
        "test_metrics": overall,
        "label_caveat": (
            "Targets are provisional weighted consensus scores; "
            "inter-annotator agreement was poor/negative."
        ),
    }
    with open(OUTPUT_DIR / "xlmr_run_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("\n" + "=" * 65)
    print("XLM-ROBERTA TEST RESULTS")
    print("=" * 65)
    print(pd.DataFrame(metric_rows).round(4).to_string(index=False))
    print("\nSaved:")
    print(MODEL_DIR)
    print(prediction_path)
    print(metrics_path)
    print(history_path)
    print(OUTPUT_DIR / "xlmr_run_config.json")
    print("\nThe test set was not used for training, early stopping, or checkpoint selection.")
    print("Interpret results cautiously: only 300 pairs and provisional labels.")


if __name__ == "__main__":
    main()

"""
JUDGEBERT RTI — controlled v2 experiment (CLS pooling + Huber, seed 123)

Run from project root:
    python src/train_mbert_improved_v2.py

Dataset priority:
    dataset/RTI_Model_Ready_Dataset_400_Integrated.csv
    dataset/RTI_Model_Ready_Dataset_Updated_Average.csv

This script trains ONE controlled run. It does not overwrite v1.
Validation RMSE selects the checkpoint. Test is evaluated only after selection.
The new 100 examples include synthetic proposed labels; human-review them before
treating results as final research evidence.
"""

from pathlib import Path
import random
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


# ---------------- Configuration ----------------
ROOT = Path(__file__).resolve().parent.parent

INTEGRATED_DATA = ROOT / "dataset" / "RTI_Model_Ready_Dataset_400_Integrated.csv"
EXPECTED_DATA = ROOT / "dataset" / "RTI_Model_Ready_Dataset_Updated_Average.csv"
DATA_PATH = INTEGRATED_DATA if INTEGRATED_DATA.exists() else EXPECTED_DATA

OUTPUT_DIR = ROOT / "output" / "mbert_improved_v2"
RUN_NAME = "cls_huber_seed123"
RUN_DIR = OUTPUT_DIR / "checkpoints" / RUN_NAME

MODEL_NAME = "google-bert/bert-base-multilingual-cased"
MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 25
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 5
SEED = 123
NUM_WORKERS = 0
HUBER_DELTA = 1.0
DROPOUT = 0.2


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


class MBertForLMP(nn.Module):
    def __init__(self, model_name, pooling="cls"):
        super().__init__()
        if pooling != "cls":
            raise ValueError("This v2 run is configured for CLS pooling.")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(DROPOUT)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**inputs).last_hidden_state
        pooled = hidden[:, 0, :]
        return self.regressor(self.dropout(pooled)).squeeze(-1)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson, spearman = float("nan"), float("nan")
    else:
        pearson = float(pearsonr(y_true, y_pred).statistic)
        spearman = float(spearmanr(y_true, y_pred).statistic)
    return {
        "N": int(len(y_true)),
        "Pearson_r": pearson,
        "Spearman_rho": spearman,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Prediction_SD": float(np.std(y_pred)),
        "Target_SD": float(np.std(y_true)),
    }


def huber_loss(pred, labels):
    return nn.functional.huber_loss(pred, labels, delta=HUBER_DELTA)


def run_epoch(model, loader, device, optimizer=None, scheduler=None, desc="Evaluate"):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, n = 0.0, 0
    predictions, labels_all = [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pred = model(input_ids, attention_mask, token_type_ids)
            loss = huber_loss(pred, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_n = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_n
        n += batch_n
        predictions.extend(pred.detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())

    return total_loss / max(n, 1), np.asarray(labels_all), np.asarray(predictions)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Place the integrated CSV in dataset/ as "
            "'RTI_Model_Ready_Dataset_400_Integrated.csv' or use the expected "
            "'RTI_Model_Ready_Dataset_Updated_Average.csv'."
        )

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    required = ["Pair_ID", "Language", "Original_Text", "Simplified_Text", "LMP_Score", "Split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["LMP_Score"] = pd.to_numeric(df["LMP_Score"], errors="coerce")
    df = df.dropna(subset=required).copy()
    df["Split"] = df["Split"].astype(str).str.strip().str.lower()
    df["Original_Text"] = df["Original_Text"].astype(str).str.strip()
    df["Simplified_Text"] = df["Simplified_Text"].astype(str).str.strip()

    if df["Pair_ID"].duplicated().any():
        duplicates = df.loc[df["Pair_ID"].duplicated(), "Pair_ID"].tolist()[:10]
        raise ValueError(f"Duplicate Pair_ID values found: {duplicates}")
    if not df["LMP_Score"].between(1, 10).all():
        raise ValueError("LMP_Score values must be between 1 and 10.")
    if (df["Original_Text"].eq("") | df["Simplified_Text"].eq("")).any():
        raise ValueError("Empty original or simplified text found.")

    train_df = df[df["Split"] == "train"].reset_index(drop=True)
    val_df = df[df["Split"] == "validation"].reset_index(drop=True)
    test_df = df[df["Split"] == "test"].reset_index(drop=True)
    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Train, validation and test must all contain rows.")

    print("=" * 72)
    print("JUDGEBERT RTI — V2 CLS + HUBER — SEED 123")
    print("=" * 72)
    print(f"Dataset: {DATA_PATH}")
    print(f"Rows: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    print("Training includes proposed synthetic additions; labels need human review.")
    print(f"Config: batch={BATCH_SIZE}, lr={LEARNING_RATE}, epochs={EPOCHS}, patience={PATIENCE}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    set_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader = DataLoader(
        LMPDataset(train_df, tokenizer), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS
    )
    # Evaluation loaders keep deterministic order.
    train_eval_loader = DataLoader(
        LMPDataset(train_df, tokenizer), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        LMPDataset(val_df, tokenizer), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        LMPDataset(test_df, tokenizer), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )

    model = MBertForLMP(MODEL_NAME, pooling="cls").to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    best_path = RUN_DIR / "best_checkpoint.pt"
    best_val_rmse, best_epoch, wait = float("inf"), 0, 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, device, optimizer, scheduler, desc=f"Train {epoch}"
        )
        # Run a separate inference pass on train to measure prediction spread
        # without using dropout or optimizer updates.
        _, train_y, train_raw = run_epoch(
            model, train_eval_loader, device, desc="Train metrics"
        )
        val_loss, val_y, val_raw = run_epoch(
            model, val_loader, device, desc="Validation"
        )

        train_pred = np.clip(train_raw, 1.0, 10.0)
        val_pred = np.clip(val_raw, 1.0, 10.0)
        train_metrics = metrics(train_y, train_pred)
        val_metrics = metrics(val_y, val_pred)

        record = {
            "Run": RUN_NAME,
            "Epoch": epoch,
            "Train_Loss": train_loss,
            "Train_Eval_Loss": train_loss,
            "Validation_Loss": val_loss,
            **{f"Train_{k}": v for k, v in train_metrics.items()},
            **{f"Validation_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        pd.DataFrame(history).to_csv(
            RUN_DIR / "training_history.csv", index=False, encoding="utf-8-sig"
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | train loss={train_loss:.4f} | "
            f"train pred SD={train_metrics['Prediction_SD']:.3f} | "
            f"val RMSE={val_metrics['RMSE']:.4f} | "
            f"val MAE={val_metrics['MAE']:.4f} | "
            f"val r={val_metrics['Pearson_r']:.3f} | "
            f"val pred SD={val_metrics['Prediction_SD']:.3f}"
        )

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": MODEL_NAME,
                    "pooling": "cls",
                    "loss": "huber",
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_validation_rmse": best_val_rmse,
                    "max_length": MAX_LENGTH,
                    "dropout": DROPOUT,
                },
                best_path,
            )
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    # Reload the validation-selected checkpoint; only now evaluate test.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, test_y, test_raw = run_epoch(model, test_loader, device, desc="Final test")
    test_pred = np.clip(test_raw, 1.0, 10.0)

    test_predictions = test_df[["Pair_ID", "Language", "LMP_Score"]].copy()
    test_predictions["Predicted_LMP_Score"] = test_pred
    test_predictions["Run"] = RUN_NAME
    test_predictions.to_csv(
        RUN_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    overall = {
        "Run": RUN_NAME,
        "Dataset": str(DATA_PATH),
        "Train_N": len(train_df),
        "Validation_N": len(val_df),
        "Test_N": len(test_df),
        "Best_Epoch": best_epoch,
        "Best_Validation_RMSE": best_val_rmse,
        **metrics(test_y, test_pred),
    }
    with open(RUN_DIR / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    # Language-wise held-out test metrics.
    language_rows = []
    langs = test_df["Language"].astype(str).to_numpy()
    for language in sorted(set(langs)):
        mask = langs == language
        language_rows.append({"Language": language, **metrics(test_y[mask], test_pred[mask])})
    pd.DataFrame(language_rows).to_csv(
        RUN_DIR / "test_metrics_by_language.csv", index=False, encoding="utf-8-sig"
    )

    model.encoder.save_pretrained(RUN_DIR / "encoder")
    tokenizer.save_pretrained(RUN_DIR / "encoder")

    print("\nBest checkpoint:", best_path)
    print("Best epoch:", best_epoch)
    print("Final held-out test metrics:")
    for key, value in overall.items():
        print(f"  {key}: {value}")
    print("\nSaved outputs in:", RUN_DIR)


if __name__ == "__main__":
    main()

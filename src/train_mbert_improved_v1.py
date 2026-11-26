"""
Controlled mBERT experiments for Legal Meaning Preservation (LMP) regression.

Run from the project root:
    python src/train_mbert_improved_v1.py

Expected dataset:
    dataset/RTI_Model_Ready_Dataset_Updated_Average.csv

Experiments:
    - CLS pooling + MSE loss (reproducible baseline)
    - CLS pooling + Huber loss
    - masked mean pooling + MSE loss
    - masked mean pooling + Huber loss
    Each configuration is trained with three seeds.

Important:
    * Validation RMSE selects the best epoch.
    * The test split is never used for training, early stopping, or model selection.
    * Test results are reported for every run; do not choose a model based on test
      performance. Compare configurations using validation results.
    * Labels are provisional consensus scores and should be interpreted cautiously.
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
DATA_PATH = ROOT / "dataset" / "RTI_Model_Ready_Dataset_Updated_Average.csv"
OUTPUT_DIR = ROOT / "output" / "mbert_improved_v1"

MODEL_NAME = "google-bert/bert-base-multilingual-cased"
MAX_LENGTH = 256
BATCH_SIZE = 1
EPOCHS = 15
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
PATIENCE = 4
SEEDS = [42, 123, 2026]
NUM_WORKERS = 0
HUBER_DELTA = 1.0
DROPOUT = 0.2

EXPERIMENTS = [
    {"pooling": "cls", "loss": "mse"},
    {"pooling": "cls", "loss": "huber"},
    {"pooling": "mean", "loss": "mse"},
    {"pooling": "mean", "loss": "huber"},
]


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
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
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
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_inputs)
        hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts

        return self.regressor(self.dropout(pooled)).squeeze(-1)


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
        "N": int(len(y_true)),
        "Pearson_r": pearson,
        "Spearman_rho": spearman,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Prediction_SD": float(np.std(y_pred)),
        "Target_SD": float(np.std(y_true)),
    }


def get_loss(loss_name, predictions, labels):
    if loss_name == "mse":
        return nn.functional.mse_loss(predictions, labels)
    if loss_name == "huber":
        return nn.functional.huber_loss(
            predictions, labels, delta=HUBER_DELTA
        )
    raise ValueError(f"Unknown loss: {loss_name}")


def run_epoch(model, loader, device, loss_name, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    total_examples = 0
    all_predictions = []
    all_labels = []

    progress = tqdm(
        loader,
        desc="Training" if training else "Evaluating",
        leave=False,
    )

    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(
                input_ids,
                attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = get_loss(loss_name, predictions, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        all_predictions.extend(predictions.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    return (
        total_loss / max(total_examples, 1),
        np.asarray(all_labels),
        np.asarray(all_predictions),
    )


def evaluate_by_language(frame, y_true, y_pred):
    rows = []
    languages = frame["Language"].astype(str).to_numpy()
    for language in sorted(set(languages)):
        mask = languages == language
        rows.append({
            "Language": language,
            **calculate_metrics(y_true[mask], y_pred[mask]),
        })
    return rows


def train_one_run(train_df, val_df, test_df, tokenizer, device,
                  pooling, loss_name, seed):
    set_seed(seed)
    run_name = f"{pooling}_{loss_name}_seed{seed}"
    run_dir = OUTPUT_DIR / "checkpoints" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

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

    model = MBertForLMP(MODEL_NAME, pooling=pooling).to(device)
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
    wait = 0
    history = []
    best_path = run_dir / "best_checkpoint.pt"

    print(f"\n--- {run_name} ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, device, loss_name, optimizer, scheduler
        )
        val_loss, val_y, val_raw = run_epoch(
            model, val_loader, device, loss_name
        )
        val_pred = np.clip(val_raw, 1.0, 10.0)
        val_metrics = calculate_metrics(val_y, val_pred)

        history.append({
            "Run": run_name,
            "Epoch": epoch,
            "Train_Loss": train_loss,
            "Validation_Loss": val_loss,
            **{f"Validation_{k}": v for k, v in val_metrics.items()},
        })

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} | "
            f"val RMSE {val_metrics['RMSE']:.4f} | "
            f"val MAE {val_metrics['MAE']:.4f} | "
            f"val Pearson {val_metrics['Pearson_r']:.4f}"
        )

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": MODEL_NAME,
                    "pooling": pooling,
                    "loss": loss_name,
                    "seed": seed,
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

    # Reload the validation-selected checkpoint before test evaluation.
    checkpoint = torch.load(
        best_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, test_y, test_raw = run_epoch(
        model, test_loader, device, loss_name
    )
    test_pred = np.clip(test_raw, 1.0, 10.0)

    predictions = test_df[
        ["Pair_ID", "Language", "LMP_Score"]
    ].copy()
    predictions["Predicted_LMP_Score"] = test_pred
    predictions["Run"] = run_name
    predictions.to_csv(
        run_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    overall = {
        "Run": run_name,
        "Pooling": pooling,
        "Loss": loss_name,
        "Seed": seed,
        "Best_Epoch": best_epoch,
        "Best_Validation_RMSE": best_val_rmse,
        **calculate_metrics(test_y, test_pred),
    }

    language_rows = []
    for row in evaluate_by_language(test_df, test_y, test_pred):
        language_rows.append({
            "Run": run_name,
            "Pooling": pooling,
            "Loss": loss_name,
            "Seed": seed,
            **row,
        })

    pd.DataFrame(history).to_csv(
        run_dir / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Save encoder and tokenizer so the checkpoint can be reconstructed later.
    model.encoder.save_pretrained(run_dir / "encoder")
    tokenizer.save_pretrained(run_dir / "encoder")

    return overall, language_rows, history


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place the model-ready CSV in the project's dataset folder."
        )

    df = pd.read_csv(DATA_PATH)
    required = [
        "Pair_ID", "Language", "Original_Text",
        "Simplified_Text", "LMP_Score", "Split",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["LMP_Score"] = pd.to_numeric(df["LMP_Score"], errors="coerce")
    df = df.dropna(subset=required).copy()
    df["Original_Text"] = df["Original_Text"].astype(str).str.strip()
    df["Simplified_Text"] = df["Simplified_Text"].astype(str).str.strip()

    if (df["Original_Text"].eq("") | df["Simplified_Text"].eq("")).any():
        raise ValueError("Empty original or simplified text found.")
    if not df["LMP_Score"].between(1, 10).all():
        raise ValueError("LMP scores must be between 1 and 10.")

    train_df = df[df["Split"].eq("train")].reset_index(drop=True)
    val_df = df[df["Split"].eq("validation")].reset_index(drop=True)
    test_df = df[df["Split"].eq("test")].reset_index(drop=True)

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Train, validation and test splits must be non-empty.")

    print("=" * 72)
    print("mBERT CONTROLLED LMP EXPERIMENTS — IMPROVED V1")
    print("=" * 72)
    print(
        f"Rows: train={len(train_df)}, validation={len(val_df)}, "
        f"test={len(test_df)}"
    )
    print(f"Seeds: {SEEDS}")
    print(f"Configurations: {len(EXPERIMENTS)}")
    print("Targets: provisional consensus LMP scores on the original 1–10 scale.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    overall_rows = []
    language_rows = []
    all_history = []

    for experiment in EXPERIMENTS:
        for seed in SEEDS:
            overall, by_language, history = train_one_run(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                tokenizer=tokenizer,
                device=device,
                pooling=experiment["pooling"],
                loss_name=experiment["loss"],
                seed=seed,
            )
            overall_rows.append(overall)
            language_rows.extend(by_language)
            all_history.extend(history)

            pd.DataFrame(overall_rows).to_csv(
                OUTPUT_DIR / "all_run_metrics.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(language_rows).to_csv(
                OUTPUT_DIR / "language_metrics_by_run.csv",
                index=False,
                encoding="utf-8-sig",
            )
            pd.DataFrame(all_history).to_csv(
                OUTPUT_DIR / "all_training_histories.csv",
                index=False,
                encoding="utf-8-sig",
            )

    results = pd.DataFrame(overall_rows)
    summary = (
        results.groupby(["Pooling", "Loss"], as_index=False)
        .agg(
            Runs=("Run", "count"),
            Mean_Validation_RMSE=("Best_Validation_RMSE", "mean"),
            SD_Validation_RMSE=("Best_Validation_RMSE", "std"),
            Mean_Test_Pearson=("Pearson_r", "mean"),
            SD_Test_Pearson=("Pearson_r", "std"),
            Mean_Test_Spearman=("Spearman_rho", "mean"),
            SD_Test_Spearman=("Spearman_rho", "std"),
            Mean_Test_MAE=("MAE", "mean"),
            SD_Test_MAE=("MAE", "std"),
            Mean_Test_RMSE=("RMSE", "mean"),
            SD_Test_RMSE=("RMSE", "std"),
            Mean_Prediction_SD=("Prediction_SD", "mean"),
        )
    )

    summary.to_csv(
        OUTPUT_DIR / "configuration_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "seeds": SEEDS,
        "experiments": EXPERIMENTS,
        "huber_delta": HUBER_DELTA,
        "dropout": DROPOUT,
        "label_caveat": (
            "Targets are provisional weighted consensus scores. "
            "The dataset has substantial annotator disagreement; "
            "results should be interpreted cautiously."
        ),
        "selection_rule": (
            "Best epoch selected by validation RMSE. Do not select a "
            "configuration based on test metrics."
        ),
    }
    with open(OUTPUT_DIR / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 72)
    print("CONFIGURATION SUMMARY")
    print("=" * 72)
    print(summary.round(4).to_string(index=False))
    print(f"\nAll outputs saved under: {OUTPUT_DIR}")
    print(
        "\nCompare configurations using mean validation RMSE and its "
        "standard deviation. Treat test metrics as final reporting only."
    )


if __name__ == "__main__":
    main()

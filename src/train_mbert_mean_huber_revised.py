"""
Controlled mBERT experiment: masked mean pooling + Huber regression.
Uses the revised 300-pair dataset, original fixed splits, and three seeds.
Outputs are isolated from all previous experiments.
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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "RTI_Revised_300_Pairs_Two_Assessors_Third_Adjudication.csv"
OUT = ROOT / "output" / "revised_annotations" / "mbert_mean_huber"

MODEL_NAME = "google-bert/bert-base-multilingual-cased"
MAX_LENGTH = 256
BATCH_SIZE = 2
EPOCHS = 12
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
PATIENCE = 3
DROPOUT = 0.2
HUBER_DELTA = 1.0
SEEDS = [42, 123, 2026]
NUM_WORKERS = 0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_corr(fn, y, pred):
    if len(y) < 2 or np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(fn(y, pred).statistic)


def calculate_metrics(y, pred):
    return {
        "N": int(len(y)),
        "Pearson_r": safe_corr(pearsonr, y, pred),
        "Spearman_rho": safe_corr(spearmanr, y, pred),
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5),
        "Prediction_SD": float(np.std(pred)),
        "Target_SD": float(np.std(y)),
    }


class LegalPairDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            str(row["Original_Text"]),
            str(row["Simplified_Text"]),
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(float(row["LMP_Score"]), dtype=torch.float32)
        return item


class MeanPoolMBertRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        token_embeddings = outputs.last_hidden_state

        # Masked mean pooling: padding tokens are excluded from the average.
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

        return self.regression_head(pooled).squeeze(-1)


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    all_targets, all_predictions = [], []
    weighted_loss, total_examples = 0.0, 0

    for batch in tqdm(loader, leave=False, desc="Train" if training else "Evaluate"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        targets = batch["labels"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(input_ids, attention_mask, token_type_ids)
            loss = nn.functional.huber_loss(
                predictions, targets, delta=HUBER_DELTA
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        batch_size = len(targets)
        weighted_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        all_targets.extend(targets.detach().cpu().numpy())
        all_predictions.extend(predictions.detach().cpu().numpy())

    return (
        weighted_loss / max(total_examples, 1),
        np.asarray(all_targets, dtype=float),
        np.asarray(all_predictions, dtype=float),
    )


def train_one_seed(train_df, val_df, test_df, tokenizer, device, seed):
    set_seed(seed)
    run_name = f"mean_huber_seed{seed}"
    run_dir = OUT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        LegalPairDataset(train_df, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        LegalPairDataset(val_df, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        LegalPairDataset(test_df, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = MeanPoolMBertRegressor().to(device)
    optimizer = AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    best_val_rmse = float("inf")
    best_epoch, patience_counter = 0, 0
    history = []
    checkpoint = run_dir / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, device, optimizer, scheduler
        )
        val_loss, val_y, val_pred = run_epoch(model, val_loader, device)
        val_pred = np.clip(val_pred, 1, 10)
        val_metrics = calculate_metrics(val_y, val_pred)

        history.append({
            "Run": run_name,
            "Epoch": epoch,
            "Train_Huber_Loss": train_loss,
            "Validation_Huber_Loss": val_loss,
            **{f"Validation_{k}": v for k, v in val_metrics.items()},
        })

        print(
            f"{run_name} | Epoch {epoch:02d}/{EPOCHS} "
            f"| train loss {train_loss:.4f} "
            f"| val RMSE {val_metrics['RMSE']:.4f} "
            f"| val MAE {val_metrics['MAE']:.4f}"
        )

        if val_metrics["RMSE"] < best_val_rmse:
            best_val_rmse = val_metrics["RMSE"]
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "pooling": "masked_mean",
                "loss": "huber",
                "seed": seed,
                "best_epoch": best_epoch,
                "best_validation_rmse": best_val_rmse,
                "max_length": MAX_LENGTH,
            }, checkpoint)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"{run_name}: early stopping.")
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    _, test_y, test_pred = run_epoch(model, test_loader, device)
    test_pred = np.clip(test_pred, 1, 10)

    predictions = test_df[
        ["Pair_ID", "Language", "Split", "LMP_Score"]
    ].copy()
    predictions["Predicted_LMP_Score"] = test_pred
    predictions.to_csv(
        run_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    overall = {
        "Run": run_name,
        "Pooling": "masked_mean",
        "Loss": "huber",
        "Seed": seed,
        "Best_Epoch": best_epoch,
        "Best_Validation_RMSE": best_val_rmse,
        **calculate_metrics(test_y, test_pred),
    }

    language_rows = []
    for language in ["English", "Hindi"]:
        mask = test_df["Language"].eq(language).to_numpy()
        language_rows.append({
            "Run": run_name,
            "Language": language,
            **calculate_metrics(test_y[mask], test_pred[mask]),
        })

    pd.DataFrame(history).to_csv(
        run_dir / "training_history.csv", index=False, encoding="utf-8-sig"
    )
    return overall, language_rows, history


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)

    required = [
        "Pair_ID", "Language", "Original_Text",
        "Simplified_Text", "LMP_Score", "Split"
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")
    if len(df) != 300 or not df["Pair_ID"].is_unique:
        raise ValueError("Expected 300 rows with unique Pair_ID values.")
    if df[required].isna().any().any():
        raise ValueError("Missing values in required dataset columns.")

    df["LMP_Score"] = pd.to_numeric(df["LMP_Score"], errors="raise")
    if not df["LMP_Score"].between(1, 10).all():
        raise ValueError("LMP_Score must be between 1 and 10.")

    train_df = df[df["Split"].eq("train")].reset_index(drop=True)
    val_df = df[df["Split"].eq("validation")].reset_index(drop=True)
    test_df = df[df["Split"].eq("test")].reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    all_runs, all_language_rows, all_histories = [], [], []

    for seed in SEEDS:
        run_result, language_results, history = train_one_seed(
            train_df, val_df, test_df, tokenizer, device, seed
        )
        all_runs.append(run_result)
        all_language_rows.extend(language_results)
        all_histories.extend(history)

        pd.DataFrame(all_runs).to_csv(
            OUT / "run_metrics.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(all_language_rows).to_csv(
            OUT / "language_metrics.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(all_histories).to_csv(
            OUT / "training_histories.csv", index=False, encoding="utf-8-sig"
        )

    config = {
        "model": MODEL_NAME,
        "pooling": "masked mean pooling over non-padding token embeddings",
        "loss": "Huber",
        "seeds": SEEDS,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "target": "LMP_Score on original 1-10 scale",
        "checkpoint_selection": "Lowest validation RMSE",
        "test_evaluation": "Each seed's selected checkpoint evaluated once on the fixed test split",
    }
    (OUT / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nTEST RESULTS")
    print(pd.DataFrame(all_runs).round(4).to_string(index=False))
    print("\nLanguage-wise results")
    print(pd.DataFrame(all_language_rows).round(4).to_string(index=False))
    print("\nSaved outputs to:", OUT)


if __name__ == "__main__":
    main()

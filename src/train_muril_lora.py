"""
MuRIL + LoRA for RTI legal meaning preservation (LMP) regression.

Run from the project root:
    python src/train_muril_lora.py

Input: original clause paired with simplified clause.
Target: provisional weighted-consensus LMP score (1-10).

Uses the existing train/validation/test split without changing it.
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
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.optim import AdamW
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dataset" / "RTI_Model_Ready_Dataset_Grouped_Split.csv"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "muril_lora_best_model"

MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
PATIENCE = 3
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
RANDOM_SEED = 42
NUM_WORKERS = 0


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


class MuRILLoRAForLMP(nn.Module):
    def __init__(self, model_name, lora_config=None, adapter_path=None):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_name)

        if adapter_path is not None:
            self.encoder = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            self.encoder = get_peft_model(base_model, lora_config)

        self.dropout = nn.Dropout(0.1)
        self.regressor = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = outputs.last_hidden_state[:, 0, :]
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
        "N": len(y_true),
        "Pearson_r": pearson,
        "Spearman_rho": spearman,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    losses, predictions_all, labels_all = [], [], []

    for batch in tqdm(
        loader,
        desc="Training" if training else "Evaluating",
        leave=False,
    ):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            predictions = model(input_ids, attention_mask, token_type_ids)
            loss = nn.functional.mse_loss(predictions, labels)

            if training:
                loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        losses.append(float(loss.detach().cpu()))
        predictions_all.extend(predictions.detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())

    return (
        float(np.mean(losses)),
        np.asarray(labels_all),
        np.asarray(predictions_all),
    )


def main():
    set_seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

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

    if df[["Original_Text", "Simplified_Text"]].eq("").any().any():
        raise ValueError("Empty original/simplified text found.")
    if not df["LMP_Score"].between(1, 10).all():
        raise ValueError("LMP scores must be between 1 and 10.")

    train_df = df[df["Split"] == "train"].reset_index(drop=True)
    val_df = df[df["Split"] == "validation"].reset_index(drop=True)
    test_df = df[df["Split"] == "test"].reset_index(drop=True)

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Train, validation and test splits must all be non-empty.")

    print("=" * 65)
    print("MuRIL + LoRA LMP REGRESSION")
    print("=" * 65)
    print(f"Rows: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    print("Input: original clause paired with simplified clause")
    print("Target: provisional weighted-consensus LMP score (1-10)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"\nLoading MuRIL: {MODEL_NAME}")
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

    # MuRIL uses BERT-style attention projections named query and value.
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"],
        bias="none",
    )

    model = MuRILLoRAForLMP(
        MODEL_NAME, lora_config=lora_config
    )
    model = model.to(device)
    model.encoder.print_trainable_parameters()

    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
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
    no_improvement = 0
    history = []

    print("\nStarting MuRIL LoRA training...")
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
            no_improvement = 0

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            torch.save(
                {
                    "regressor_state_dict": model.regressor.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_rmse": best_val_rmse,
                    "model_name": MODEL_NAME,
                    "max_length": MAX_LENGTH,
                },
                MODEL_DIR / "regression_head.pt",
            )
            print("Best MuRIL LoRA checkpoint saved.")
        else:
            no_improvement += 1
            print(f"No validation improvement ({no_improvement}/{PATIENCE}).")

        if no_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_epoch == 0:
        raise RuntimeError("No best checkpoint was saved.")

    print(f"\nBest epoch: {best_epoch}; validation RMSE: {best_val_rmse:.4f}")

    # Reload selected adapter and regression head for test evaluation.
    checkpoint = torch.load(
        MODEL_DIR / "regression_head.pt",
        map_location=device,
        weights_only=False,
    )
    model = MuRILLoRAForLMP(
        MODEL_NAME, adapter_path=MODEL_DIR
    ).to(device)
    model.regressor.load_state_dict(checkpoint["regressor_state_dict"])
    model.eval()

    _, test_y, test_pred_raw = run_epoch(model, test_loader, device)
    test_pred = np.clip(test_pred_raw, 1, 10)

    prediction_df = test_df[
        ["Pair_ID", "Language", "LMP_Score"]
    ].copy()
    prediction_df["Predicted_LMP_Score"] = test_pred
    if "Provisional_Label" in test_df.columns:
        prediction_df["Provisional_Label"] = test_df["Provisional_Label"].values

    prediction_path = OUTPUT_DIR / "muril_lora_test_predictions.csv"
    prediction_df.to_csv(
        prediction_path, index=False, encoding="utf-8-sig"
    )

    metric_rows = [
        {"Split": "test", "Language": "All", **calculate_metrics(test_y, test_pred)}
    ]
    for language in sorted(test_df["Language"].unique()):
        mask = test_df["Language"].eq(language).to_numpy()
        metric_rows.append({
            "Split": "test",
            "Language": language,
            **calculate_metrics(test_y[mask], test_pred[mask]),
        })

    metrics_path = OUTPUT_DIR / "muril_lora_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(
        metrics_path, index=False, encoding="utf-8-sig"
    )

    history_path = OUTPUT_DIR / "muril_lora_training_history.csv"
    pd.DataFrame(history).to_csv(
        history_path, index=False, encoding="utf-8-sig"
    )

    config = {
        "model_name": MODEL_NAME,
        "method": "LoRA",
        "rank": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "target_modules": ["query", "value"],
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "seed": RANDOM_SEED,
        "best_epoch": best_epoch,
        "best_validation_rmse": best_val_rmse,
        "test_metrics": calculate_metrics(test_y, test_pred),
        "label_caveat": (
            "Targets are provisional weighted consensus scores; "
            "inter-annotator agreement was poor/negative."
        ),
    }
    with open(
        OUTPUT_DIR / "muril_lora_run_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    print("\n" + "=" * 65)
    print("MuRIL + LoRA TEST RESULTS")
    print("=" * 65)
    print(pd.DataFrame(metric_rows).round(4).to_string(index=False))
    print("\nSaved:")
    print(MODEL_DIR)
    print(prediction_path)
    print(metrics_path)
    print(history_path)
    print(OUTPUT_DIR / "muril_lora_run_config.json")
    print("\nTest set was not used for training, early stopping, or checkpoint selection.")
    print("Targets are provisional; interpret results cautiously.")


if __name__ == "__main__":
    main()

"""
MuRIL + LoRA v2: stable LMP regression.

Run from project root:
    python src/train_muril_lora_v2.py

Uses the existing grouped train/validation/test split.
Input: original clause + simplified clause.
Target: LMP_Score on the 1-10 scale.

Changes from v1:
- masked mean pooling over token representations
- sigmoid-bounded output mapped to 1-10 (no post-hoc clipping)
- normalized target loss on [0, 1]
- raw and scaled prediction diagnostics
- robust extraction of last hidden states from PEFT output
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
DATA_PATH = ROOT / "dataset" / "RTI_Model_Ready_Dataset_Updated_Average.csv"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "muril_lora_v2_updated_average_best_model"

MODEL_NAME = "google/muril-base-cased"
MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 15
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
PATIENCE = 4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
SEED = 42
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

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.originals[idx],
            self.simplified[idx],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        # Normalize original 1-10 target to [0, 1] for stable regression.
        item["labels"] = torch.tensor(
            (self.scores[idx] - 1.0) / 9.0, dtype=torch.float32
        )
        return item


class MuRILLoRARegressor(nn.Module):
    def __init__(self, model_name, lora_config=None, adapter_path=None):
        super().__init__()
        base = AutoModel.from_pretrained(model_name)
        if adapter_path is None:
            self.encoder = get_peft_model(base, lora_config)
        else:
            self.encoder = PeftModel.from_pretrained(base, adapter_path)

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(hidden_size, 1)
        # Start near the middle of the target range rather than below 1.
        nn.init.xavier_uniform_(self.regressor.weight, gain=0.1)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)

        # BaseModelOutput and tuple-style outputs are both handled.
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs[0]

        # Masked mean pooling ignores padding tokens.
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / denominator

        logits = self.regressor(self.dropout(pooled)).squeeze(-1)
        normalized_prediction = torch.sigmoid(logits)
        return normalized_prediction


def metrics(y_true, y_pred):
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
    }


def run_epoch(model, loader, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    losses, preds_norm, labels_norm = [], [], []

    for batch in tqdm(loader, desc="Training" if training else "Evaluating", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        token_types = batch.get("token_type_ids")
        if token_types is not None:
            token_types = token_types.to(device)
        labels = batch["labels"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pred = model(ids, mask, token_types)
            loss = nn.functional.mse_loss(pred, labels)
            if training:
                loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        losses.append(float(loss.detach().cpu()))
        preds_norm.extend(pred.detach().cpu().numpy().tolist())
        labels_norm.extend(labels.detach().cpu().numpy().tolist())

    return float(np.mean(losses)), np.asarray(labels_norm), np.asarray(preds_norm)


def to_score(normalized):
    return 1.0 + 9.0 * np.asarray(normalized, dtype=float)


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required = [
        "Pair_ID", "Language", "Original_Text",
        "Simplified_Text", "LMP_Score", "Split",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["LMP_Score"] = pd.to_numeric(df["LMP_Score"], errors="coerce")
    df = df.dropna(subset=required).copy()
    df["Original_Text"] = df["Original_Text"].astype(str).str.strip()
    df["Simplified_Text"] = df["Simplified_Text"].astype(str).str.strip()
    if (df["Original_Text"].eq("") | df["Simplified_Text"].eq("")).any():
        raise ValueError("Empty original or simplified text found.")
    if not df["LMP_Score"].between(1, 10).all():
        raise ValueError("LMP_Score must be between 1 and 10.")

    train = df[df["Split"].eq("train")].reset_index(drop=True)
    val = df[df["Split"].eq("validation")].reset_index(drop=True)
    test = df[df["Split"].eq("test")].reset_index(drop=True)
    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("Train, validation and test splits must all be non-empty.")

    print("=" * 65)
    print("MuRIL + LoRA v2: UPDATED AVERAGE DATASET")
    print("=" * 65)
    print(f"Rows: train={len(train)}, validation={len(val)}, test={len(test)}")
    print(f"Training score range: {train.LMP_Score.min():.3f} to {train.LMP_Score.max():.3f}")
    print("Input: original clause + simplified clause")
    print("Output: sigmoid-bounded LMP score mapped to 1-10")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(
        LMPDataset(train, tokenizer), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        LMPDataset(val, tokenizer), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        LMPDataset(test, tokenizer), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
    )

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"],
        bias="none",
    )
    model = MuRILLoRARegressor(MODEL_NAME, lora_config=lora_config).to(device)
    model.encoder.print_trainable_parameters()

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
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
    no_improve = 0
    history = []

    print("\nStarting MuRIL LoRA v2 training...")
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, _, train_pred_norm = run_epoch(
            model, train_loader, device, optimizer, scheduler
        )
        val_loss, val_y_norm, val_pred_norm = run_epoch(model, val_loader, device)

        val_y = to_score(val_y_norm)
        val_pred = to_score(val_pred_norm)
        val_metrics = metrics(val_y, val_pred)
        val_rmse = val_metrics["RMSE"]

        print(
            f"Train normalized MSE: {train_loss:.5f} | "
            f"Validation normalized MSE: {val_loss:.5f} | "
            f"Validation RMSE: {val_rmse:.4f} | "
            f"Validation MAE: {val_metrics['MAE']:.4f} | "
            f"Prediction range: {val_pred.min():.3f}-{val_pred.max():.3f}"
        )

        history.append({
            "Epoch": epoch,
            "Train_Normalized_MSE": train_loss,
            "Validation_Normalized_MSE": val_loss,
            "Validation_RMSE": val_rmse,
            "Validation_MAE": val_metrics["MAE"],
            "Validation_Pearson": val_metrics["Pearson_r"],
            "Validation_Spearman": val_metrics["Spearman_rho"],
            "Validation_Prediction_Min": float(val_pred.min()),
            "Validation_Prediction_Max": float(val_pred.max()),
            "Validation_Prediction_Mean": float(val_pred.mean()),
        })

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            no_improve = 0
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
            print("Best v2 checkpoint saved.")
        else:
            no_improve += 1
            print(f"No validation improvement ({no_improve}/{PATIENCE}).")

        if no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    if best_epoch == 0:
        raise RuntimeError("No best checkpoint was saved.")

    print(f"\nBest epoch: {best_epoch}; validation RMSE: {best_val_rmse:.4f}")

    checkpoint = torch.load(
        MODEL_DIR / "regression_head.pt",
        map_location=device,
        weights_only=False,
    )
    best_model = MuRILLoRARegressor(MODEL_NAME, adapter_path=MODEL_DIR).to(device)
    best_model.regressor.load_state_dict(checkpoint["regressor_state_dict"])
    best_model.eval()

    _, test_y_norm, test_pred_norm = run_epoch(best_model, test_loader, device)
    test_y = to_score(test_y_norm)
    test_pred = to_score(test_pred_norm)
    overall = metrics(test_y, test_pred)

    prediction_df = test[["Pair_ID", "Language", "LMP_Score"]].copy()
    prediction_df["Predicted_LMP_Score"] = test_pred
    prediction_df["Raw_Normalized_Prediction"] = test_pred_norm
    if "Provisional_Label" in test.columns:
        prediction_df["Provisional_Label"] = test["Provisional_Label"].values

    prediction_path = OUTPUT_DIR / "muril_lora_v2_updated_average_test_predictions.csv"
    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    metric_rows = [{"Split": "test", "Language": "All", **overall}]
    for language in sorted(test.Language.unique()):
        mask = test["Language"].eq(language).to_numpy()
        metric_rows.append({
            "Split": "test",
            "Language": language,
            **metrics(test_y[mask], test_pred[mask]),
        })

    metrics_path = OUTPUT_DIR / "muril_lora_v2_updated_average_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    history_path = OUTPUT_DIR / "muril_lora_v2_updated_average_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")

    run_config = {
        "model_name": MODEL_NAME,
        "method": "LoRA",
        "version": "v2",
        "pooling": "attention-mask-aware mean pooling",
        "output_mapping": "1 + 9 * sigmoid(logit)",
        "target_modules": ["query", "value"],
        "rank": LORA_R,
        "alpha": LORA_ALPHA,
        "dropout": LORA_DROPOUT,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs_requested": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_rmse": best_val_rmse,
        "test_metrics": overall,
        "label_caveat": "Targets are provisional weighted consensus scores; prior inter-annotator agreement was poor/negative.",
    }
    with open(OUTPUT_DIR / "muril_lora_v2_updated_average_run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\n" + "=" * 65)
    print("MuRIL + LoRA v2 UPDATED AVERAGE TEST RESULTS")
    print("=" * 65)
    print(pd.DataFrame(metric_rows).round(4).to_string(index=False))
    print("\nRaw normalized test prediction range:",
          f"{test_pred_norm.min():.5f} to {test_pred_norm.max():.5f}")
    print("Test prediction score range:",
          f"{test_pred.min():.4f} to {test_pred.max():.4f}")
    print("\nSaved:")
    print(MODEL_DIR)
    print(prediction_path)
    print(metrics_path)
    print(history_path)
    print(OUTPUT_DIR / "muril_lora_v2_updated_average_run_config.json")
    print("Test set was not used for training, early stopping, or checkpoint selection.")


if __name__ == "__main__":
    main()

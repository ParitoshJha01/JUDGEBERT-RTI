from pathlib import Path
import csv
import io

import streamlit as st
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

# Run app.py from the project root. Expected model outputs:
# output/mbert_improved_v1/checkpoints/cls_mse_seed42/
PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = PROJECT_ROOT / "output" / "mbert_improved_v1" / "checkpoints"
DEFAULT_RUN = CHECKPOINT_ROOT / "cls_huber_seed123"
MAX_LENGTH = 256
DROPOUT = 0.2


class MBertForLMP(nn.Module):
    """Architecture copied from train_mbert_improved_v1.py."""
    def __init__(self, encoder, pooling="cls"):
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.encoder = encoder
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


def resolve_run_path(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@st.cache_resource(show_spinner=False)
def load_model(run_path_text, device_name):
    run_dir = resolve_run_path(run_path_text)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    encoder_dir = run_dir / "encoder"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not encoder_dir.is_dir():
        raise FileNotFoundError(f"Encoder/tokenizer folder not found: {encoder_dir}")

    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pooling = checkpoint.get("pooling", "cls")
    model_name = checkpoint.get("model_name", "google-bert/bert-base-multilingual-cased")
    max_length = int(checkpoint.get("max_length", MAX_LENGTH))
    dropout = float(checkpoint.get("dropout", DROPOUT))

    # This training architecture uses dropout=0.2. The checkpoint's dropout
    # value is retained as a safeguard if the training config changes.
    encoder = AutoModel.from_pretrained(str(encoder_dir), local_files_only=True)
    model = MBertForLMP(encoder, pooling=pooling)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir), local_files_only=True)
    return tokenizer, model, device, max_length, checkpoint


def predict(original, simplified, tokenizer, model, device, max_length):
    encoded = tokenizer(
        original,
        simplified,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        raw_score = model(
            encoded["input_ids"],
            encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids"),
        ).item()
    return max(1.0, min(10.0, float(raw_score)))


def label_for(score):
    if score < 6:
        return "Not Preserved"
    if score < 8:
        return "Partially Preserved"
    return "Preserved"


st.set_page_config(
    page_title="JUDGEBERT-RTI | LMP Analyzer",
    page_icon="⚖️",
    layout="wide",
)

st.title("⚖️ JUDGEBERT-RTI")
st.caption("Legal Meaning Preservation Analyzer · English and Hindi · RTI Act research prototype")

with st.sidebar:
    st.header("Model settings")
    default_path = str(DEFAULT_RUN.relative_to(PROJECT_ROOT))
    run_path = st.text_input(
        "Checkpoint run folder",
        value=default_path,
        help="Folder containing best_checkpoint.pt and encoder/. Use a relative path from the project root or an absolute path.",
    )
    st.caption("Validation-selected run: CLS + Huber, seed 123 (lowest validation RMSE in the recorded runs).")
    st.divider()
    st.header("About")
    st.write("Estimates how well a simplified sentence preserves the meaning of an original legal sentence.")
    st.warning("Research prototype only. Not legal advice, and not a guarantee of legal correctness.")

with st.form("lmp_form"):
    language = st.selectbox("Language pair", ["English", "Hindi"])
    col1, col2 = st.columns(2)
    with col1:
        original = st.text_area(
            "Original legal sentence",
            height=220,
            placeholder="Paste the original legal sentence…",
        )
    with col2:
        simplified = st.text_area(
            "Simplified sentence",
            height=220,
            placeholder="Paste the simplified sentence…",
        )
    submitted = st.form_submit_button(
        "Analyze meaning preservation",
        type="primary",
        use_container_width=True,
    )

if submitted:
    original = original.strip()
    simplified = simplified.strip()
    if not original or not simplified:
        st.error("Please enter both the original and simplified sentences.")
    else:
        try:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
            with st.spinner("Loading checkpoint and analyzing the sentence pair…"):
                tokenizer, model, device, max_length, checkpoint = load_model(run_path, device_name)
                score = predict(original, simplified, tokenizer, model, device, max_length)
            st.session_state["lmp_result"] = {
                "language": language,
                "original": original,
                "simplified": simplified,
                "score": score,
                "run_path": run_path,
                "pooling": checkpoint.get("pooling", "cls"),
                "loss": checkpoint.get("loss", "unknown"),
                "seed": checkpoint.get("seed", "unknown"),
                "best_epoch": checkpoint.get("best_epoch", "unknown"),
            }
        except Exception as exc:
            st.error(f"Could not load or run this checkpoint: {exc}")
            st.info("Check that the selected run folder contains best_checkpoint.pt and encoder/.")

result = st.session_state.get("lmp_result")
if result:
    score = result["score"]
    label = label_for(score)
    st.divider()
    st.header("Prediction")
    a, b, c = st.columns(3)
    a.metric("Predicted LMP score", f"{score:.2f} / 10")
    b.metric("Project label", label)
    c.metric("Model run", f"{result['pooling'].upper()} + {result['loss'].upper()}")

    st.caption(
        f"Seed: {result['seed']} · Best epoch: {result['best_epoch']} · "
        f"Checkpoint folder: {result['run_path']}"
    )
    st.progress((score - 1.0) / 9.0)
    st.caption("Project-defined thresholds: score <6 = Not Preserved; 6–<8 = Partially Preserved; ≥8 = Preserved.")

    with st.expander("Original sentence", expanded=True):
        st.write(result["original"])
    with st.expander("Simplified sentence", expanded=True):
        st.write(result["simplified"])

    st.subheader("Manual legal review checklist")
    st.caption("These are review prompts—not automatically detected errors or model explanations.")
    review_items = [
        "Obligation or permission changed (shall/must/may)",
        "Negation or contradiction introduced",
        "Condition or exception omitted",
        "Scope or responsible party changed",
        "Date, deadline, amount, or number changed",
        "Important information omitted or added",
    ]
    flags = []
    for item in review_items:
        if st.checkbox(item, key="review_" + item):
            flags.append(item)
    if flags:
        st.warning("Review flags: " + "; ".join(flags))
    else:
        st.info("No checklist flags selected. This does not certify legal correctness.")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Language", "Original_Text", "Simplified_Text",
        "Predicted_LMP_Score", "Predicted_Label", "Pooling",
        "Loss", "Seed", "Best_Epoch", "Checkpoint_Folder",
        "Manual_Review_Flags",
    ])
    writer.writerow([
        result["language"], result["original"], result["simplified"],
        round(score, 4), label, result["pooling"], result["loss"],
        result["seed"], result["best_epoch"], result["run_path"],
        " | ".join(flags),
    ])
    st.download_button(
        "Download result CSV",
        output.getvalue().encode("utf-8-sig"),
        "judgebert_lmp_result.csv",
        "text/csv",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Scores are model estimates based on project data and provisional consensus labels. "
    "The thresholds are project-defined, not official legal standards. Verify against "
    "the authoritative law and obtain qualified human review."
)

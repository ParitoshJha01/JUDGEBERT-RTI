from pathlib import Path
import csv
import io

import streamlit as st
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

# Run from the project root. v4.1 multitask checkpoint is kept separate from v3/v4.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = PROJECT_ROOT / "output" / "mbert_v4_1_multitask"
MAX_LENGTH = 256
DROPOUT = 0.1
CRITICAL_THRESHOLD = 0.5


class MultiTaskModel(nn.Module):
    """Architecture matching train_mbert_v4_1_multitask.py."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(DROPOUT)
        self.regression_head = nn.Linear(hidden_size, 1)
        self.critical_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        encoder_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_inputs)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        lmp_score = self.regression_head(cls).squeeze(-1)
        critical_logit = self.critical_head(cls).squeeze(-1)
        return lmp_score, critical_logit


def resolve_run_path(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@st.cache_resource(show_spinner=False)
def load_model(run_path_text, device_name):
    run_dir = resolve_run_path(run_path_text)
    checkpoint_path = run_dir / "best_v4_1_multitask.pt"
    encoder_dir = run_dir / "encoder"

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"v4.1 checkpoint not found: {checkpoint_path}")
    if not encoder_dir.is_dir():
        raise FileNotFoundError(f"Encoder/tokenizer folder not found: {encoder_dir}")

    device = torch.device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = checkpoint.get("model_name", "google-bert/bert-base-multilingual-cased")
    max_length = int(checkpoint.get("max_length", MAX_LENGTH))

    encoder = AutoModel.from_pretrained(str(encoder_dir), local_files_only=True)
    model = MultiTaskModel(encoder)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(encoder_dir), local_files_only=True)
    return tokenizer, model, device, max_length, checkpoint, model_name


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
        raw_score, critical_logit = model(
            encoded["input_ids"],
            encoded["attention_mask"],
            token_type_ids=encoded.get("token_type_ids"),
        )
        score = max(1.0, min(10.0, float(raw_score.item())))
        critical_probability = float(torch.sigmoid(critical_logit).item())
    return score, critical_probability


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
st.caption("Legal Meaning Preservation Analyzer · English and Hindi · Multitask research prototype")

with st.sidebar:
    st.header("Model settings")
    default_path = str(DEFAULT_RUN.relative_to(PROJECT_ROOT))
    run_path = st.text_input(
        "v4.1 checkpoint folder",
        value=default_path,
        help="Folder containing best_v4_1_multitask.pt and encoder/. Relative to project root or absolute.",
    )
    st.caption("Frozen v4.1 multitask model · best checkpoint selected by validation RMSE.")
    st.divider()
    st.header("About")
    st.write("Estimates LMP score and the probability of a critical legal meaning change.")
    st.warning("Research prototype only. Not legal advice or a guarantee of legal correctness.")

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
            with st.spinner("Loading v4.1 checkpoint and analyzing the sentence pair…"):
                tokenizer, model, device, max_length, checkpoint, model_name = load_model(
                    run_path, device_name
                )
                score, critical_probability = predict(
                    original, simplified, tokenizer, model, device, max_length
                )
            st.session_state["lmp_result"] = {
                "language": language,
                "original": original,
                "simplified": simplified,
                "score": score,
                "critical_probability": critical_probability,
                "critical_threshold": CRITICAL_THRESHOLD,
                "run_path": run_path,
                "best_epoch": checkpoint.get("best_epoch", "unknown"),
                "model_name": model_name,
            }
        except Exception as exc:
            st.error(f"Could not load or run the v4.1 checkpoint: {exc}")
            st.info(
                "Check that the selected folder contains best_v4_1_multitask.pt "
                "and an encoder/ folder."
            )

result = st.session_state.get("lmp_result")
if result:
    score = result["score"]
    critical_probability = result["critical_probability"]
    label = label_for(score)
    critical_pred = critical_probability >= result["critical_threshold"]

    st.divider()
    st.header("Prediction")
    a, b, c = st.columns(3)
    a.metric("Predicted LMP score", f"{score:.2f} / 10")
    b.metric("Project label", label)
    c.metric("Critical-change probability", f"{critical_probability * 100:.1f}%")

    if critical_pred:
        st.warning("The model flags a possible critical legal meaning change. Review the pair carefully.")
    else:
        st.info("The model did not flag a critical change at the current 0.50 threshold. This does not mean the pair is legally equivalent.")

    st.caption(
        f"v4.1 · Best epoch: {result['best_epoch']} · "
        f"Threshold: {result['critical_threshold']:.2f} · Checkpoint: {result['run_path']}"
    )
    st.progress((score - 1.0) / 9.0)
    st.caption(
        "Project-defined LMP thresholds: score <6 = Not Preserved; "
        "6–<8 = Partially Preserved; ≥8 = Preserved."
    )
    st.caption(
        "Critical-change test recall was 25% on the held-out test set; "
        "the model may miss important legal changes. Always perform human review."
    )

    with st.expander("Original sentence", expanded=True):
        st.write(result["original"])
    with st.expander("Simplified sentence", expanded=True):
        st.write(result["simplified"])

    st.subheader("Manual legal review checklist")
    st.caption("Review prompts only—not automatically detected errors or explanations.")
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
        "Language", "Original_Text", "Simplified_Text", "Predicted_LMP_Score",
        "Predicted_Label", "Critical_Change_Probability",
        "Predicted_Critical_Change", "Threshold", "Best_Epoch",
        "Checkpoint_Folder", "Manual_Review_Flags",
    ])
    writer.writerow([
        result["language"], result["original"], result["simplified"],
        round(score, 4), label, round(critical_probability, 6),
        int(critical_pred), result["critical_threshold"], result["best_epoch"],
        result["run_path"], " | ".join(flags),
    ])
    st.download_button(
        "Download result CSV",
        output.getvalue().encode("utf-8-sig"),
        "judgebert_v4_1_result.csv",
        "text/csv",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Scores and probabilities are model estimates based on project data and provisional labels. "
    "Thresholds are project-defined, not official legal standards. Verify against authoritative "
    "law and obtain qualified human review."
)

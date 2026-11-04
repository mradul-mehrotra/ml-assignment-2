"""Streamlit interface for evaluating saved Online Shoppers classifiers."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

st.set_page_config(
    page_title="Online Shopper Purchase Predictor",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Online Shopper Purchase Intention")
st.caption("Compare five trained classification models on an uploaded labelled test CSV.")

if not METADATA_PATH.exists():
    st.error("Model metadata is missing. Run `python train.py` first and commit the generated artifacts.")
    st.stop()

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

target_col = metadata["target"]
model_names = list(metadata["models"].keys())

with st.sidebar:
    st.header("Evaluation settings")
    selected_model = st.selectbox("Choose a classification model", model_names)
    st.write("**Target column:**", target_col)
    st.write("**Expected features:**", len(metadata["features"]))
    st.divider()
    st.info("Upload labelled test data: it must contain every input feature and the Revenue target column.")

uploaded_file = st.file_uploader("Upload test dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Upload the generated `test_data.csv` to evaluate the saved models.")
    st.subheader("Expected CSV structure")
    st.code(", ".join(metadata["features"][:5]) + ", ..., " + target_col)
    st.stop()

try:
    test_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

missing = [c for c in metadata["features"] + [target_col] if c not in test_df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

if test_df.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

X_test = test_df[metadata["features"]].copy()
raw_y = test_df[target_col]

# Accept the 0/1 target produced by train.py, as well as common boolean labels.
if raw_y.dtype == bool:
    y_test = raw_y.astype(int)
else:
    normalized = raw_y.astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0}
    y_test = normalized.map(mapping)

if y_test.isna().any() or not set(y_test.unique()).issubset({0, 1}):
    st.error("Revenue must contain only 0/1, True/False, or Yes/No values.")
    st.stop()

model_file = MODEL_DIR / metadata["models"][selected_model]
if not model_file.exists():
    st.error(f"Saved model not found: {model_file.name}. Ensure the models folder is committed to GitHub.")
    st.stop()

try:
    model = joblib.load(model_file)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
except Exception as exc:
    st.error(f"Model evaluation failed. Check feature names, data types and dependency versions. Details: {exc}")
    st.stop()

if len(set(y_test)) < 2:
    auc_value = float("nan")
    auc_note = "AUC requires both target classes in the uploaded test data."
else:
    auc_value = roc_auc_score(y_test, probabilities)
    auc_note = ""

metrics = {
    "Accuracy": accuracy_score(y_test, predictions),
    "AUC": auc_value,
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Recall": recall_score(y_test, predictions, zero_division=0),
    "F1 Score": f1_score(y_test, predictions, zero_division=0),
    "MCC": matthews_corrcoef(y_test, predictions),
}

st.subheader(f"Evaluation results — {selected_model}")
st.write(f"**Test records:** {len(test_df):,}")
cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, "N/A" if pd.isna(value) else f"{value:.4f}")

if auc_note:
    st.warning(auc_note)

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots()
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No purchase", "Purchase"]
    )
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(selected_model)
    st.pyplot(fig)
    plt.close(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["No purchase", "Purchase"],
        output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(4), use_container_width=True)

st.divider()
st.subheader("Compare all five models")
comparison_rows = []
for name, filename in metadata["models"].items():
    path = MODEL_DIR / filename
    if not path.exists():
        continue
    candidate = joblib.load(path)
    pred = candidate.predict(X_test)
    proba = candidate.predict_proba(X_test)[:, 1]
    comparison_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba) if len(set(y_test)) == 2 else float("nan"),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })

comparison_df = pd.DataFrame(comparison_rows)
st.dataframe(
    comparison_df.style.format({
        "Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}",
        "Recall": "{:.4f}", "F1": "{:.4f}", "MCC": "{:.4f}"
    }),
    use_container_width=True
)

st.download_button(
    "Download evaluation comparison (CSV)",
    data=comparison_df.to_csv(index=False).encode("utf-8"),
    file_name="model_comparison.csv",
    mime="text/csv"
)

with st.expander("Preview uploaded test data"):
    st.dataframe(test_df.head(20), use_container_width=True)

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from _common import TASK_NAMES, stats
from p3_modellab.external_eval import evaluate_predictions


@st.cache_data(show_spinner=False)
def _read_upload(name: str, content: bytes) -> pd.DataFrame:
    source = BytesIO(content)
    if name.lower().endswith(".csv"):
        return pd.read_csv(source)
    return pd.read_parquet(source)


LABELS = {
    "qlike": "QLIKE (lower is better)",
    "rank_correlation": "Overall rank correlation",
    "mean_period_rank_ic": "Mean period rank IC",
    "mse": "Mean squared error",
    "mae": "Mean absolute error",
    "log_loss": "Log loss (lower is better)",
    "brier_score": "Brier score (lower is better)",
    "pr_auc": "PR-AUC (higher is better)",
    "event_rate": "Event rate",
    "examples": "Examples",
    "periods": "Periods scored",
}


st.title("Evaluate External Predictions")
st.caption("Evaluate forecasts from your model on your dataset. This page scores predictions; it does not execute uploaded model code.")

task = st.segmented_control(
    "Prediction task",
    list(TASK_NAMES),
    default="returns",
    format_func=TASK_NAMES.get,
) or "returns"

with st.expander("How to prepare the file", expanded=False):
    st.markdown(
        "Run your model using the repository workflow, then upload one CSV or Parquet table containing an actual-outcome "
        "column and a prediction column. Return forecasts may also include a month or date column. Refer to the repository "
        "for full benchmark integration and walk-forward fitting."
    )

uploaded = st.file_uploader("Upload your scored dataset", type=["csv", "parquet"])
if uploaded is None:
    st.info("Upload a scored CSV or Parquet file to begin.")
    st.stop()

try:
    data = _read_upload(uploaded.name, uploaded.getvalue())
except Exception as exc:
    st.error(f"Could not read this file: {exc}")
    st.stop()

st.caption(f"{len(data):,} rows · {len(data.columns):,} columns")
numeric = [column for column in data.columns if pd.api.types.is_numeric_dtype(data[column])]
if len(numeric) < 2:
    st.error("The file needs at least two numeric columns: actual outcomes and predictions.")
    st.stop()

c1, c2 = st.columns(2)
target_col = c1.selectbox("Actual outcome column", numeric)
forecast_choices = [column for column in numeric if column != target_col]
forecast_col = c2.selectbox("Model prediction column", forecast_choices)
time_col = None
if task == "returns":
    time_choice = st.selectbox("Month/date column (optional)", ["None", *data.columns.tolist()])
    time_col = None if time_choice == "None" else time_choice

if st.button("Evaluate my predictions", type="primary"):
    try:
        result = evaluate_predictions(data, task, target_col, forecast_col, time_col)
    except ValueError as exc:
        st.error(str(exc))
    else:
        cards = []
        for key, value in result.items():
            if key in {"examples", "periods"}:
                shown = f"{int(value):,}"
            else:
                shown = "Unavailable" if not np.isfinite(value) else f"{value:.5f}"
            cards.append((LABELS[key], shown))
        stats(cards)
        st.success("Evaluation complete. Use the repository workflow to add this model to the full walk-forward benchmark.")

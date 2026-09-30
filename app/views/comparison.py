from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _common import NICE, TASK_NAMES, badge, catalog_label, chart, report_catalog, stats
from p3_modellab import plots
from p3_modellab.uncertainty import block_bootstrap


DISPLAY = {
    "base_rate": "Historical base rate", "logit": "Logistic regression",
    "zero": "Zero forecast", "reversal": "One-month reversal",
    **NICE,
}
COLORS = {"base_rate": "#9E9E9E", "logit": "#009E73", "zero": "#9E9E9E",
          "reversal": "#0072B2", **plots.MODEL_COLORS}


@st.cache_data(show_spinner=False)
def _json(path: str, stamp: float) -> pd.DataFrame:
    return pd.read_json(path)


@st.cache_data(show_spinner=False)
def _parquet(path: str, stamp: float) -> pd.DataFrame:
    return pd.read_parquet(path)


def _load_frame(path: Path) -> pd.DataFrame:
    loader = _json if path.suffix == ".json" else _parquet
    return loader(str(path), path.stat().st_mtime)


def _bar(summary: pd.DataFrame, task: str) -> go.Figure:
    if task == "volatility":
        metric, title, better = "qlike", "Average QLIKE", "Lower is better"
    elif task == "returns":
        metric, title, better = "mean_rank_ic", "Mean monthly rank correlation", "Higher is better"
    else:
        metric, title, better = "mean_monthly_log_loss", "Mean monthly log loss", "Lower is better"
    ordered = summary.sort_values(metric, ascending=task == "returns")
    fig = go.Figure(go.Bar(
        x=[DISPLAY.get(model, model) for model in ordered["model"]], y=ordered[metric],
        marker_color=[COLORS.get(model, "#777777") for model in ordered["model"]],
        text=[f"{value:.4f}" for value in ordered[metric]], textposition="outside",
        hovertemplate="%{x}<br>%{y:.5f}<extra></extra>",
    ))
    fig.update_yaxes(title=f"{title} · {better}")
    return plots.style(fig, f"{title} across saved test months", height=430, legend=False)


def _paired_months(report: Path, task: str, first: str, second: str) -> tuple[float, float, float, int, str]:
    if task == "volatility":
        data, value, direction = _load_frame(report / "monthly_losses.parquet"), "qlike", "lower"
    elif task == "returns":
        data, value, direction = _load_frame(report / "monthly_metrics.parquet"), "rank_ic", "higher"
    else:
        data, value, direction = _load_frame(report / "monthly_metrics.parquet"), "log_loss", "lower"
    wide = data.pivot(index="yyyymm", columns="model", values=value)[[first, second]].dropna()
    difference = (wide[first] - wide[second]).to_numpy()
    interval = block_bootstrap(difference, block=12, seed=0)
    return interval.estimate, interval.lo, interval.hi, len(wide), direction


@st.cache_data(show_spinner=False)
def _all_pairwise(path: str, stamp: float, metric: str, direction: str,
                  selected: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every selected model on the same saved months.

    The matrix is expressed as an advantage: positive values mean that the row
    model beat the column model, regardless of whether the underlying metric is
    minimized or maximized.
    """
    monthly = pd.read_parquet(path)
    wide = monthly.pivot(index="yyyymm", columns="model", values=metric)
    matrix = pd.DataFrame(0.0, index=selected, columns=selected)
    rows: list[dict] = []
    for first, second in combinations(selected, 2):
        paired = wide[[first, second]].dropna()
        interval = block_bootstrap((paired[first] - paired[second]).to_numpy(), block=12, seed=0)
        clear = interval.lo > 0 or interval.hi < 0
        first_better = interval.estimate < 0 if direction == "lower" else interval.estimate > 0
        winner = first if first_better else second
        advantage = -interval.estimate if direction == "lower" else interval.estimate
        matrix.loc[first, second] = advantage
        matrix.loc[second, first] = -advantage
        rows.append({
            "Model A": DISPLAY.get(first, first),
            "Model B": DISPLAY.get(second, second),
            "A minus B": interval.estimate,
            "95% low": interval.lo,
            "95% high": interval.hi,
            "Paired months": len(paired),
            "Result": f"{DISPLAY.get(winner, winner)} better" if clear else "No clear difference",
        })
    return pd.DataFrame(rows), matrix


def _all_model_inputs(report: Path, task: str) -> tuple[Path, str, str, str]:
    if task == "volatility":
        return report / "monthly_losses.parquet", "qlike", "lower", "QLIKE"
    if task == "returns":
        return report / "monthly_metrics.parquet", "rank_ic", "higher", "rank correlation"
    return report / "monthly_metrics.parquet", "log_loss", "lower", "log loss"


def _advantage_heatmap(matrix: pd.DataFrame, unit: str) -> go.Figure:
    labels = [DISPLAY.get(model, model) for model in matrix.index]
    values = matrix.to_numpy(dtype="float64")
    fig = go.Figure(go.Heatmap(
        z=values,
        x=labels,
        y=labels,
        zmid=0,
        colorscale=[[0, "#B2182B"], [0.5, "#F7F7F7"], [1, "#2166AC"]],
        text=np.vectorize(lambda value: f"{value:+.4f}")(values),
        texttemplate="%{text}",
        hovertemplate="Row: %{y}<br>Column: %{x}<br>row advantage: %{z:+.5f}<extra></extra>",
        colorbar=dict(title="row<br>advantage"),
    ))
    fig.update_xaxes(side="top")
    return plots.style(fig, f"All pairwise saved-month comparisons · {unit}",
                       height=max(430, 54 * len(labels)), legend=False)


def _headline(summary: pd.DataFrame, task: str) -> list[tuple]:
    if task == "volatility":
        error = summary.loc[summary["qlike"].idxmin()]
        rank = summary.loc[summary["rank_corr"].idxmax()]
        return [("Lowest average error", DISPLAY.get(error.model, error.model), f"QLIKE {error.qlike:.4f}"),
                ("Best stock ranking", DISPLAY.get(rank.model, rank.model), f"rank correlation {rank.rank_corr:.3f}"),
                ("Models evaluated", str(len(summary)), "same stocks and months")]
    if task == "returns":
        best = summary.loc[summary["mean_rank_ic"].idxmax()]
        return [("Best stock ranking", DISPLAY.get(best.model, best.model), f"rank IC {best.mean_rank_ic:.4f}"),
                ("Models evaluated", str(len(summary)), "same stocks and months")]
    best = summary.loc[summary["mean_monthly_log_loss"].idxmin()]
    return [("Lowest probability loss", DISPLAY.get(best.model, best.model), f"log loss {best.mean_monthly_log_loss:.4f}"),
            ("Event PR-AUC", f"{best.overall_pr_auc:.3f}", "for the lowest-loss model"),
            ("Models evaluated", str(len(summary)), "same stocks and months")]


st.title("Saved model comparison")
st.caption("This page reads compact saved metrics. Changing a selection does not train a model or scan the full forecast file.")
catalog = report_catalog()
if not catalog:
    st.info("No saved reports were found. Build a report after a model run, then reopen this page.")
    st.stop()

choice = st.selectbox("Saved study", [run["run_id"] for run in catalog],
                      format_func=lambda run_id: catalog_label(next(run for run in catalog if run["run_id"] == run_id)))
run = next(run for run in catalog if run["run_id"] == choice)
badge(run)
summary = _load_frame(run["report"] / "model_summary.json")
stats(_headline(summary, run["task"]))
chart(_bar(summary, run["task"]), key="saved_model_summary")

st.subheader("Compare all saved models")
models = summary["model"].tolist()
selected_models = st.multiselect(
    "Models in comparison",
    models,
    default=models,
    format_func=lambda model: DISPLAY.get(model, model),
)
if len(selected_models) < 2:
    st.info("Select at least two models for the pairwise matrix.")
else:
    monthly_path, metric, direction, unit = _all_model_inputs(run["report"], run["task"])
    pairwise, matrix = _all_pairwise(
        str(monthly_path), monthly_path.stat().st_mtime, metric, direction, tuple(selected_models)
    )
    chart(_advantage_heatmap(matrix, unit), key="all_model_pairwise")
    st.caption(
        f"Positive cells mean the row model performed better than the column model on saved {unit}. "
        "The diagonal is zero. No fitting or forecast generation occurs here."
    )
    with st.expander(f"Every pair ({len(pairwise)} comparisons)"):
        st.dataframe(
            pairwise,
            hide_index=True,
            width="stretch",
            column_config={
                "A minus B": st.column_config.NumberColumn(format="%+.5f"),
                "95% low": st.column_config.NumberColumn(format="%+.5f"),
                "95% high": st.column_config.NumberColumn(format="%+.5f"),
            },
        )

st.subheader("Compare any two saved models")
c1, c2 = st.columns(2)
first = c1.selectbox("First model", models, index=0, format_func=lambda model: DISPLAY.get(model, model))
second_options = [model for model in models if model != first]
second = c2.selectbox("Second model", second_options, index=0, format_func=lambda model: DISPLAY.get(model, model))
estimate, lo, hi, n_months, direction = _paired_months(run["report"], run["task"], first, second)
favours_first = estimate < 0 if direction == "lower" else estimate > 0
clear = lo > 0 or hi < 0
if clear:
    winner = first if favours_first else second
    verdict = f"{DISPLAY.get(winner, winner)} has the better saved score"
else:
    verdict = "No clear difference in the saved test months"
unit = {"volatility": "QLIKE", "returns": "rank correlation", "delisting": "log loss"}[run["task"]]
stats([("Head-to-head", verdict, f"{n_months} paired months"),
       (f"First minus second {unit}", f"{estimate:+.4f}", f"95% interval {lo:+.4f} to {hi:+.4f}")])
st.caption("The monthly scores were computed during reporting. This selection only takes their paired difference and confidence interval.")

importance_path = run["report"] / "importance.parquet"
if run["task"] == "volatility" and importance_path.exists():
    st.subheader("Compare saved factor reliance")
    importance = _load_frame(importance_path)
    available_models = sorted(importance["model"].unique(), key=lambda model: models.index(model) if model in models else 999)
    selected = st.selectbox("Model for factor view", available_models, format_func=lambda model: DISPLAY.get(model, model))
    kind = st.segmented_control("Factor level", ["group", "feature"], default="group",
                                format_func={"group": "Groups", "feature": "Individual factors"}.get) or "group"
    factors = importance[(importance["model"] == selected) & (importance["kind"] == kind)].sort_values("mean")
    factor_names = factors["input"].tolist()
    f1, f2 = st.columns(2)
    first_factor = f1.selectbox("First factor", factor_names, index=0, key="first_factor")
    second_factor = f2.selectbox("Second factor", [name for name in factor_names if name != first_factor],
                                 index=0, key="second_factor")
    reliance = factors.set_index("input")["mean"]
    stats([(first_factor, f"{reliance[first_factor]:+.4f}", "QLIKE increase when shuffled"),
           (second_factor, f"{reliance[second_factor]:+.4f}", "QLIKE increase when shuffled")],
          accent=COLORS.get(selected, "#777777"))
    fig = go.Figure(go.Bar(x=factors["mean"], y=factors["input"], orientation="h",
                           marker_color=COLORS.get(selected, "#777777"),
                           error_x=dict(type="data", array=factors["std"])))
    fig.update_xaxes(title="increase in QLIKE when shuffled · larger means more reliance")
    chart(plots.style(fig, f"{DISPLAY.get(selected, selected)} factor reliance", height=max(330, 28 * len(factors)), legend=False),
          key="saved_factor_reliance")

with st.expander("What is stored and what is computed here?"):
    st.markdown(
        f"**Stored run:** `{run['run_id']}`  \n"
        f"**Task:** {TASK_NAMES[run['task']]}  \n"
        f"**Saved forecasts:** {run['forecasts']:,}  \n"
        f"**Report folder:** `{run['report']}`\n\n"
        "Training, forecasts, monthly scores, calibration, and factor reliance are saved. The page performs only small "
        "display calculations on the saved monthly aggregates. A new model, new input set, or paid external model needs "
        "a separate offline/cloud run before it can appear here."
    )

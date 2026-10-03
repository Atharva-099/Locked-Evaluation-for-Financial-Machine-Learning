import streamlit as st

from _common import NICE, aggregate_report_for, badge, chart, select_aggregate_volatility_run, stats
from p3_modellab import plots

st.title("Volatility Model Analysis")
st.caption("Compare all eight saved volatility models using aggregate exploratory or locked-holdout results.")
run = select_aggregate_volatility_run()
badge(run)
data = aggregate_report_for(run)
summary = data["summary"]

lowest = summary.loc[summary["qlike"].idxmin()]
ranked = summary.loc[summary["rank_corr"].idxmax()]
stats([
    ("Lowest average error", NICE[lowest.model], f"QLIKE {lowest.qlike:.4f}"),
    ("Best stock ranking", NICE[ranked.model], f"rank correlation {ranked.rank_corr:.3f}"),
    ("Models evaluated", str(len(summary)), "same stocks and months"),
])

series = data["series"]
models = [model for model in summary["model"] if model in series.columns]
c1, c2 = st.columns([3, 1])
selected = c1.multiselect("Models on chart", models, default=models, format_func=NICE.get)
view = c2.segmented_control("View", ["2D", "3D"], default="2D", key="aggregate_view") or "2D"
if not selected:
    st.info("Select at least one model to draw the chart.")
else:
    start, end = int(series.index.min() // 100), int(series.index.max() // 100)
    years = st.slider("Time frame", start, end, (start, end), key="aggregate_years")
    shown = series[(series.index // 100 >= years[0]) & (series.index // 100 <= years[1])]
    chart(plots.main_chart(
        shown,
        selected,
        view,
        "all",
        "Market-wide realized volatility and saved forecasts",
    ))
    st.caption(
        "Each line is the annualized square root of the cross-sectional mean variance for that month. "
        "The 3D view rotates and zooms; both views use the same saved aggregate series."
    )

with st.expander("Calibration and ranking diagnostics"):
    tabs = st.tabs(["Calibration", "Ranking skill"])
    with tabs[0]:
        chart(plots.calibration(data["calibration"]), key="aggregate_calibration")
    with tabs[1]:
        chart(plots.rank_skill(data["skill"]), key="aggregate_rank_skill")

st.caption(
    "This hosted page contains monthly and report-level aggregates only. Stock search and individual forecast rows remain local to the licensed-data workflow."
)

import streamlit as st

from _common import NICE, aggregate_report_for, badge, chart, select_aggregate_volatility_run, stats
from p3_modellab import misses, plots

st.title("Volatility Error Analysis")
st.caption("See where each saved model gains or loses accuracy across market segments.")
run = select_aggregate_volatility_run()
badge(run)
data = aggregate_report_for(run)
slices = data["slices"]

c1, c2 = st.columns([1, 2])
pairs = [(row.model, row.baseline) for row in slices[["model", "baseline"]].drop_duplicates().itertuples(index=False)]
pair = c1.selectbox("Comparison", pairs, format_func=lambda value: f"{NICE[value[0]]} vs {NICE[value[1]]}")
kinds = c2.multiselect(
    "Break down by",
    list(misses.SLICES),
    default=["year", "size", "price", "industry"],
    format_func=lambda kind: misses.SLICES[kind].split(" (")[0],
)
subset = slices[
    (slices["model"] == pair[0])
    & (slices["baseline"] == pair[1])
    & slices["slice"].isin(kinds)
]
counts = subset["verdict"].value_counts()
stats([
    (verdict.capitalize(), int(counts.get(verdict, 0)), "segments")
    for verdict in ["better", "worse", "no detectable difference", "too little data"]
])
if len(subset):
    chart(plots.slices(subset, *pair))
st.caption("These are exploratory segment checks; testing many segments can produce chance findings.")

st.subheader("Model reliance on input factors")
level = st.segmented_control(
    "Factor level",
    ["group", "feature"],
    default="group",
    format_func={"group": "Groups", "feature": "Individual factors"}.get,
    key="aggregate_factor_level",
) or "group"
chart(plots.importance(data["importance"], level), key="aggregate_importance")

st.info(
    "Individual worst-case stocks are intentionally excluded from the public bundle because those rows contain licensed security-level data."
)

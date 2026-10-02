import streamlit as st

from _common import NICE, badge, chart, need_run, report_for, stats
from p3_modellab import misses, plots

st.title("Volatility Error Analysis")
st.caption("Inspect market segments and individual forecasts where volatility models gain or lose accuracy.")
r = need_run()
badge(r)
d = report_for(r)
sl = d["slices"]

c1, c2 = st.columns([1, 2])
pairs = [(p.model, p.baseline) for p in sl[["model", "baseline"]].drop_duplicates().itertuples(index=False)]
pair = c1.selectbox("Comparison", pairs, format_func=lambda p: f"{NICE[p[0]]} vs {NICE[p[1]]}")
kinds = c2.multiselect("Break down by", list(misses.SLICES), default=["year", "size", "price", "industry"],
                       format_func=lambda k: misses.SLICES[k].split(" (")[0])
sub = sl[(sl["model"] == pair[0]) & (sl["baseline"] == pair[1]) & sl["slice"].isin(kinds)]

counts = sub["verdict"].value_counts()
stats([(v.capitalize(), int(counts.get(v, 0)), "groups") for v in ["better", "worse", "no detectable difference", "too little data"]])
if len(sub):
    chart(plots.slices(sub, *pair))
st.caption("About 1 in 20 verdicts can be luck, because many groups are checked.")

st.subheader("Biggest misses")
w = d["worst"]
if len(w):
    a, b = st.columns([1, 2])
    m = a.selectbox("Model", sorted(w["model"].unique()), format_func=NICE.get)
    k = b.segmented_control("Show", ["model much worse", "model much better"], default="model much worse") or "model much worse"
    show = w[(w["model"] == m) & (w["kind"] == k)]
    fcols = [c for c in show.columns if c.startswith("forecast_")]
    st.dataframe(show[["ticker", "yyyymm", "industry", "target_rv", *fcols, "qlike_gap"]],
                 column_config={"target_rv": st.column_config.NumberColumn("actual", format="%.4f"),
                                **{c: st.column_config.NumberColumn(NICE[c.replace("forecast_", "")], format="%.4f") for c in fcols},
                                "qlike_gap": st.column_config.NumberColumn("gap vs baseline", format="%+.2f"),
                                "yyyymm": st.column_config.NumberColumn("month", format="%d")},
                 hide_index=True, width="stretch")

st.subheader("What each model relies on")
g = st.segmented_control("Level", ["group", "feature"], default="group", format_func={"group": "Input groups", "feature": "Single inputs"}.get) or "group"
chart(plots.importance(d["importance"], g))

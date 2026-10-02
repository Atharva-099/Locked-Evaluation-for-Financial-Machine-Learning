import pandas as pd
import streamlit as st

from _common import NICE, badge, chart, explore_table, names, need_run, report_for, stats
from p3_modellab import explore, plots

st.title("Volatility Model Analysis")
st.caption("Detailed diagnostics for the complete eight-model volatility studies. Use Model Benchmark for cross-task comparisons.")
r = need_run()
badge(r)
d = report_for(r)
t = explore_table(r)
models = explore.models_in(t)

# Headline: the two overall answers.
sc = explore.scorecard(d)
best = sc.index[0]
lowest = sc["average error"].idxmin()
breakdown = "; ".join(f"{NICE[m]}: average rank {row['average rank']:.2f}" for m, row in sc.iterrows())
stats([("Lowest error overall", NICE[lowest], "all stocks, all test months", "Lowest average QLIKE."),
       ("Best overall model", NICE[best], "error, typical month, ranking, consistency",
        "Each model is ranked on average error, error in a typical month, ranking skill and years clearly beating persistence. "
        "Best = lowest average rank. " + breakdown)])

# Choices.
c1, c2, c3 = st.columns([1, 1.3, 1.8])
model = c1.selectbox("Model", models, index=models.index(best), format_func=NICE.get)
scope = c2.segmented_control("Stocks", ["all", "stock"], default="all", format_func={"all": "All stocks", "stock": "One stock"}.get,
                             key="scope") or "all"
y0, y1 = int(t["yyyymm"].min() // 100), int(t["yyyymm"].max() // 100)
years = c3.slider("Time frame", y0, y1, (y0, y1))

permno, label = None, "All stocks"
if scope == "stock":
    s1, s2 = st.columns([1, 2])
    q = s1.text_input("Search", "AAPL", placeholder="Ticker or company name")
    hits = explore.search(names(), q, set(t["permno"].unique()))
    if hits.empty:
        st.info("No stock matches that. Try a ticker like MSFT or part of a name.")
        st.stop()
    pick = s2.selectbox("Stock", hits.itertuples(index=False), format_func=lambda h: f"{h.ticker}  -  {h.comnam}")
    permno, label = int(pick.permno), pick.ticker

sub = explore.filtered(t, years, permno)
if sub.empty:
    st.warning("No forecasts for this choice and time frame.")
    st.stop()

# The selected model on its own.
acc = explore.accuracy(sub, model)
cards = [(f"{NICE[model]} is typically off by", f"{acc['typical_off']:.0%}", "of the actual swing",
          "Half the forecasts are closer than this, half further away (in yearly-volatility terms)."),
         ("Within 25%", f"{acc['within']:.0%}", "of forecasts", "Share of forecasts within 25% of the actual swing."),
         ("Forecast too low", f"{acc['too_low']:.0%}", "of the time", "Share of forecasts below the actual swing.")]
if scope == "all":
    cards.append(("Ranking skill", f"{explore.rank_skill(sub, model):.2f}", "0 = random, 1 = perfect order",
                  "Within each month, how well forecasts put stocks in the right order."))
else:
    cards.append(("Months", f"{acc['n']}", f"{sub['yyyymm'].min() // 100} to {sub['yyyymm'].max() // 100}"))
stats(cards, accent=plots.MODEL_COLORS[model])

# Main chart.
h1, h2 = st.columns([3, 1])
compare = h1.multiselect("Compare with (optional)", [m for m in models if m != model], format_func=NICE.get)
view = h2.segmented_control("View", ["2D", "3D"], default="2D", key="view") or "2D"
shown = [model] + compare
series = explore.market_series(sub, shown) if scope == "all" else explore.stock_series(sub, shown)
what = "typical stock (median)" if scope == "all" else label
chart(plots.main_chart(series, shown, view, scope, f"{what}: actual swing vs forecast"))
st.caption("Each point is the month the forecast was made; the swing is for the following month. Hover to see "
           + ("the share price." if scope == "stock" else "the market swing and number of stocks.")
           + (" Drag to rotate; use the chart toolbar to zoom." if view == "3D" else ""))

# Optional head-to-head for each compared model.
if compare:
    st.subheader("Head to head")
    filt = {"permno": [permno]} if permno is not None else None
    for other in compare:
        res = explore.custom_compare(t, model, other, years, filt)
        v = explore.verdict(res)
        text = {"better": f"{NICE[model]} beats {NICE[other]}", "worse": f"{NICE[other]} beats {NICE[model]}",
                "no detectable difference": "No detectable difference", "too little data": "Too little data to tell",
                "no examples": "No examples"}[v]
        accent = {"better": "#2E7D32", "worse": "#C62828"}.get(v, "#9E9E9E")
        if res.get("n_obs"):
            stats([(f"{NICE[model]} vs {NICE[other]}", text, f"{res['months_a_better']:.0%} of months {NICE[model]} better",
                    "Paired by month on exactly the same stocks."),
                   ("Difference", f"{res['estimate']:+.4f}", f"95% interval {res['ci_lo']:+.3f} to {res['ci_hi']:+.3f}",
                    "Error of the first model minus the second; negative = first model better.")], accent=accent)
            with st.expander(f"Month by month: {NICE[model]} vs {NICE[other]}"):
                chart(plots.custom_monthly(res["monthly"], model, other))

with st.expander("More charts"):
    ml = explore.monthly_losses(sub)[shown]
    chart(plots.loss_timeline(ml, title="Error month by month (lower is better)"))
    if scope == "all":
        tabs = st.tabs(["Year by year vs HAR", "Ranking skill", "Forecast size"])
        with tabs[0]:
            chart(plots.over_time(d["by_year"], "har"))
        with tabs[1]:
            chart(plots.rank_skill(d["skill"]))
        with tabs[2]:
            chart(plots.calibration(d["calibration"]))

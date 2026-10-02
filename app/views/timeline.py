import pandas as pd
import streamlit as st

from _common import NICE, badge, chart, explore_table, need_run, report_for, stats
from p3_modellab import explore, plots

st.title("Volatility Performance Over Time")
st.caption("Track the eight volatility models across locked and exploratory market periods.")
r = need_run()
badge(r)
d = report_for(r)
base = st.segmented_control("Against", ["har", "persistence"], default="har", format_func=NICE.get, key="race_base") or "har"
chart(plots.year_race(d["by_year"], base))

t = explore_table(r)
ml = explore.monthly_losses(t)
months = list(ml.index)
label = lambda ym: pd.Timestamp(year=ym // 100, month=ym % 100, day=1).strftime("%b %Y")
default = 202002 if 202002 in months else months[len(months) // 2]
ym = st.select_slider("Month the forecast was made", months, value=default, format_func=label)
chart(plots.loss_timeline(ml, ym))

row = ml.loc[ym]
mood = t.loc[t["yyyymm"] == ym, "market_regime"].iloc[0]
cards = [("Market mood", mood.split(" (")[0].capitalize(), mood[mood.find("(") + 1:-1] if "(" in mood else "",
          "Market-wide yearly swing at the forecast.")]
cards += [(NICE[m], f"{v:.3f}", "best that month" if i == 0 else "", "Average error that month (lower is better).")
          for i, (m, v) in enumerate(row.sort_values().items())]
stats(cards)

import pandas as pd
import streamlit as st

from _common import NICE, aggregate_report_for, badge, chart, select_aggregate_volatility_run, stats
from p3_modellab import plots

st.title("Volatility Performance Over Time")
st.caption("Track all eight saved models through exploratory years and the untouched locked holdout.")
run = select_aggregate_volatility_run()
badge(run)
data = aggregate_report_for(run)

available_baselines = list(data["by_year"]["baseline"].drop_duplicates())
default = "har" if "har" in available_baselines else available_baselines[0]
baseline = st.segmented_control(
    "Against",
    available_baselines,
    default=default,
    format_func=NICE.get,
    key="aggregate_timeline_baseline",
) or default
chart(plots.year_race(data["by_year"], baseline), key="aggregate_year_race")

monthly = data["monthly"].pivot(index="yyyymm", columns="model", values="qlike").sort_index()
months = list(monthly.index)
label = lambda ym: pd.Timestamp(year=ym // 100, month=ym % 100, day=1).strftime("%b %Y")
default_month = 202002 if 202002 in months else months[len(months) // 2]
month = st.select_slider("Forecast month", months, value=default_month, format_func=label)
chart(plots.loss_timeline(monthly, month), key="aggregate_monthly_losses")

row = monthly.loc[month].sort_values()
stats([
    (NICE.get(model, model), f"{value:.3f}", "best that month" if index == 0 else "", "Monthly average QLIKE; lower is better.")
    for index, (model, value) in enumerate(row.items())
])
st.caption("The slider reads saved monthly scores. It does not rerun forecasts or model fitting.")

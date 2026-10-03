import streamlit as st

from _common import has_aggregate_volatility_pages, has_detailed_pages, has_sweep_pages, sidebar_controls, theme_toggle

st.set_page_config(page_title="Equity ML Forecast Lab", layout="wide")

research = [
    st.Page("views/home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/comparison.py", title="Model Benchmark", icon=":material/compare_arrows:"),
]
pages = {"Research": research}
detailed = has_detailed_pages()
aggregate = not detailed and has_aggregate_volatility_pages()
if detailed:
    research.append(st.Page("views/results.py", title="Volatility Analysis", icon=":material/show_chart:"))
elif aggregate:
    research.append(st.Page("views/aggregate_results.py", title="Volatility Analysis", icon=":material/show_chart:"))
research.append(st.Page("views/custom_test.py", title="Evaluate a Model", icon=":material/upload_file:"))
if detailed:
    pages["Diagnostics"] = [
        st.Page("views/misses.py", title="Error Analysis", icon=":material/troubleshoot:"),
        st.Page("views/timeline.py", title="Performance Over Time", icon=":material/timeline:"),
    ]
elif aggregate:
    pages["Diagnostics"] = [
        st.Page("views/aggregate_misses.py", title="Error Analysis", icon=":material/troubleshoot:"),
        st.Page("views/aggregate_timeline.py", title="Performance Over Time", icon=":material/timeline:"),
    ]
if has_sweep_pages():
    pages["Robustness"] = [st.Page("views/tweaks.py", title="Robustness Lab", icon=":material/science:")]

nav = st.navigation(pages)
theme_toggle()
sidebar_controls()
st.session_state["_chart_n"] = 0  # chart ids restart each time the page is drawn
nav.run()

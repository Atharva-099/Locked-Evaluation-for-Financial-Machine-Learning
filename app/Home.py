import streamlit as st

from _common import sidebar_controls

st.set_page_config(page_title="Stock Swing Forecast", layout="wide")

nav = st.navigation({
    "Main": [st.Page("views/home.py", title="Home", icon=":material/home:", default=True),
             st.Page("views/comparison.py", title="Model comparison", icon=":material/compare_arrows:"),
             st.Page("views/results.py", title="Results", icon=":material/show_chart:")],
    "More": [st.Page("views/misses.py", title="Misses", icon=":material/troubleshoot:"),
             st.Page("views/timeline.py", title="Time slider", icon=":material/timeline:")],
    "Advanced": [st.Page("views/tweaks.py", title="Tweaks (sweeps)", icon=":material/science:")],
})
sidebar_controls()
st.session_state["_chart_n"] = 0  # chart ids restart each time the page is drawn
nav.run()

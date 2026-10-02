import streamlit as st

from _common import MODEL_CARDS, NICE
from p3_modellab import plots

EXPERIMENT = (
    "At each month-end, every model gets the same information available then: past swings, recent returns, "
    "company size, price, trading activity, liquidity and market volatility. It forecasts next month's swing "
    "for roughly 4,000–7,000 U.S. common stocks using about two million stock-months from 1991–2024. "
    "Models retrain yearly on the past, explore 2000–2021, then face the locked 2022–2024 holdout; lower QLIKE wins."
)

models_html = "".join(
    f"<div class='lm'><span class='dot' style='background:{plots.MODEL_COLORS[m]}'></span><b>{NICE[m]}</b>"
    f"<span class='lmtxt'>{c['one']}</span></div>" for m, c in MODEL_CARDS.items())

st.markdown(f"""
<style>
@keyframes rise {{ from {{ opacity:0; transform:translateY(12px) }} to {{ opacity:1; transform:none }} }}
.landing {{ max-width:860px; margin:6vh auto 0 auto; animation:rise .7s ease-out both }}
.brand {{ font-size:3.1em; font-weight:800; line-height:1.1; letter-spacing:-0.02em;
          background:linear-gradient(90deg,#0072B2,#009E73,#E69F00); -webkit-background-clip:text; background-clip:text; color:transparent }}
.tag {{ font-size:1.2em; opacity:0.8; margin:10px 0 28px 0 }}
.sec {{ font-size:0.8em; letter-spacing:.12em; text-transform:uppercase; opacity:0.6; margin:22px 0 8px 0 }}
.lm {{ display:flex; gap:10px; align-items:baseline; padding:7px 0; border-bottom:1px solid rgba(128,128,128,0.15) }}
.lm b {{ min-width:150px }}
.lmtxt {{ opacity:0.8 }}
.dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; flex:none }}
.experiment {{ margin-top:8px; padding:16px 18px; border-radius:12px; line-height:1.55; background:rgba(128,128,128,0.08); border:1px solid rgba(128,128,128,0.18) }}
</style>
<div class="landing">
  <div class="brand">Equity ML Forecast Lab</div>
  <div class="tag">Eight forecasting architectures compete on identical point-in-time equity data; a locked holdout tests which signals survive.</div>
  <div class="sec">Model architectures</div>{models_html}
  <div class="sec">Research design</div><div class="experiment">{EXPERIMENT}</div>
</div>
""", unsafe_allow_html=True)

st.write("")
_, mid, _ = st.columns([1, 1, 1])
if mid.button("Open model benchmark  →", type="primary", width="stretch"):
    st.switch_page("views/comparison.py")

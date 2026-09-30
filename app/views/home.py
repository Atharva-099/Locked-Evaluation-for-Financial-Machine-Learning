import streamlit as st

from _common import MODEL_CARDS, NICE
from p3_modellab import plots

GROUPS = [
    ("Past swings", "yesterday, last week, last month, last quarter, last year"),
    ("Recent price moves", "the last month and the last quarter"),
    ("Size and trading", "company size, share price, trading activity, how hard it is to trade"),
    ("Market mood", "how much the whole market has been swinging"),
]
PARAMS = [
    ("Forecast", "next month's price swing, made at each month end"),
    ("Stocks", "US common stocks, about 4,000 to 7,000 each month"),
    ("Data", "1991 to 2024, about 2 million stock-months"),
    ("Tested on", "2000 to 2021, one year at a time; 2022 to 2024 kept sealed"),
    ("Training", "models retrained every year on past data only"),
    ("Score", "QLIKE, the standard error measure for swings (lower is better)"),
]

models_html = "".join(
    f"<div class='lm'><span class='dot' style='background:{plots.MODEL_COLORS[m]}'></span><b>{NICE[m]}</b>"
    f"<span class='lmtxt'>{c['one']}</span></div>" for m, c in MODEL_CARDS.items())
groups_html = "".join(f"<div class='lg'><b>{g}</b><span>{d}</span></div>" for g, d in GROUPS)
params_html = "".join(f"<div class='lp'><b>{k}</b><span>{v}</span></div>" for k, v in PARAMS)

st.markdown(f"""
<style>
@keyframes rise {{ from {{ opacity:0; transform:translateY(12px) }} to {{ opacity:1; transform:none }} }}
.landing {{ max-width:860px; margin:6vh auto 0 auto; animation:rise .7s ease-out both }}
.brand {{ font-size:3.1em; font-weight:800; line-height:1.1; letter-spacing:-0.02em;
          background:linear-gradient(90deg,#0072B2,#009E73,#E69F00); -webkit-background-clip:text; background-clip:text; color:transparent }}
.tag {{ font-size:1.2em; opacity:0.8; margin:10px 0 28px 0 }}
.sec {{ font-size:0.8em; letter-spacing:.12em; text-transform:uppercase; opacity:0.6; margin:22px 0 8px 0 }}
.lm, .lg, .lp {{ display:flex; gap:10px; align-items:baseline; padding:6px 0; border-bottom:1px solid rgba(128,128,128,0.15) }}
.lm b, .lg b, .lp b {{ min-width:150px }}
.lmtxt, .lg span, .lp span {{ opacity:0.8 }}
.dot {{ width:10px; height:10px; border-radius:50%; display:inline-block; flex:none }}
</style>
<div class="landing">
  <div class="brand">Stock Swing Forecast</div>
  <div class="tag">How well machine learning models forecast next month's stock price swings, tested on 35 years of US stock data.</div>
  <div class="sec">Models</div>{models_html}
  <div class="sec">Factors the models use</div>{groups_html}
  <div class="sec">Setup</div>{params_html}
</div>
""", unsafe_allow_html=True)

st.write("")
_, mid, _ = st.columns([1, 1, 1])
if mid.button("See results  →", type="primary", width="stretch"):
    st.switch_page("views/comparison.py")

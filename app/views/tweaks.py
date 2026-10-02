import streamlit as st

from _common import NICE, chart, confirm, start_job, stats, sweeps
from p3_modellab import sweep_plots

st.title("Robustness Lab")
st.caption("Sensitivity experiments vary one design choice at a time on fixed 2015–2019 data. Model selection remains confined to tuning periods.")

WHAT = {
    "settings": "LightGBM tree size x learning speed",
    "greedy": "Adding inputs one at a time",
    "data": "Years of history x share of firms",
    "noise": "Messy or missing inputs",
}

s = sweeps()
if not s:
    st.info("No sweeps yet. Start one below.")
else:
    pick = st.selectbox("Sweep", s, format_func=lambda x: f"{WHAT[x['kind']]}  ({x['size']}, {x['trials']} trials, {x['created'].replace('T', ' ')})")
    data = sweep_plots.load(pick["path"])
    man, t = data["manifest"], data["trials"]
    body = t[t["role"].isna()]
    refs = t[t["role"] == "reference"].set_index("model")["eval_qlike"]
    swept_models = sorted(body["model"].dropna().unique())
    st.caption(
        "Models saved in this sweep: "
        + ", ".join(NICE.get(model, model) for model in swept_models)
        + ". Sweeps test sensitivity for selected representative models; use Model comparison for the full saved-model leaderboard."
    )
    if man.get("size") == "demo":
        st.warning("Demo size: only checks the calculations run. Too small to read results from.", icon=":material/science:")

    def score_card(rows, label: str) -> None:
        cards = [("Your pick", label, "move the sliders"),
                 ("Score", f"{rows['eval_qlike'].mean():.4f}", "lower is better", "Average over seeds, forecasts 2015 to 2019.")]
        if "diff_vs_har" in rows and rows["diff_vs_har"].notna().any():
            d = rows["diff_vs_har"].mean()
            cards.append(("vs HAR", f"{d:+.4f}", "better than HAR" if d < 0 else "worse than HAR", "Same stocks and months; negative = better."))
        cards.append(("Seeds", len(rows), "repeats"))
        stats(cards)

    kind = man["kind"]
    if kind == "settings":
        lg = body[body["model"] == "lightgbm"]
        c1, c2 = st.columns(2)
        leaves = c1.select_slider("Tree size (leaves)", sorted(lg["num_leaves"].unique()), format_func=lambda v: f"{int(v)}")
        lr = c2.select_slider("Learning speed", sorted(lg["learning_rate"].unique()), format_func=lambda v: f"{v:g}")
        rows = lg[(lg["num_leaves"] == leaves) & (lg["learning_rate"] == lr)]
        score_card(rows, f"{int(leaves)} x {lr:g}")
        chart(sweep_plots.surface(lg, "num_leaves", "learning_rate", title="Score for every setting (yellow = your pick, red = chosen on tuning years)",
                                  log_x=True, log_y=True, highlight=(leaves, lr)))
    elif kind == "data":
        m = st.segmented_control("Model", sorted(body["model"].unique()), default="lightgbm" if "lightgbm" in set(body["model"]) else None,
                                 format_func=NICE.get) or sorted(body["model"].unique())[0]
        sub = body[body["model"] == m]
        c1, c2 = st.columns(2)
        yrs = c1.select_slider("Years of history", sorted(sub["train_years"].unique()), format_func=lambda v: f"{int(v)}")
        pct = c2.select_slider("Share of firms (%)", sorted(sub["firm_percent"].unique()), format_func=lambda v: f"{int(v)}%")
        rows = sub[(sub["train_years"] == yrs) & (sub["firm_percent"] == pct)]
        score_card(rows, f"{int(yrs)}y, {int(pct)}%")
        chart(sweep_plots.surface(sub, "train_years", "firm_percent", title=f"{NICE[m]}: score for every amount of data (yellow = your pick)",
                                  highlight=(yrs, pct)))
    elif kind == "greedy":
        m = st.segmented_control("Model", sorted(body["model"].unique()), default=sorted(body["model"].unique())[0], format_func=NICE.get) \
            or sorted(body["model"].unique())[0]
        sub = body[(body["model"] == m) & (body["seed"] == body["seed"].min())].sort_values("step")
        step = st.slider("Number of inputs", 0, int(sub["step"].max()), min(3, int(sub["step"].max())))
        row = sub[sub["step"] == step]
        score_card(body[(body["model"] == m) & (body["step"] == step)], f"{step} inputs")
        st.markdown("**Inputs so far:** " + (", ".join(f"`{x}`" for x in row["inputs"].iloc[0].split(",")) if step else "none (one forecast for everyone)"))
        chart(sweep_plots.greedy_path(body))
    elif kind == "noise":
        corr = st.segmented_control("Damage", ["noise", "blank"], default="noise", format_func={"noise": "Added noise", "blank": "Blanked values"}.get) or "noise"
        sub = body[body["corruption"] == corr]
        lvl = st.select_slider("How much", sorted(sub["level"].unique()),
                               format_func=(lambda v: f"{v:g} x spread") if corr == "noise" else (lambda v: f"{v:.0%} blanked"))
        c = st.columns(len(sub["model"].unique()) + len(refs))
        for col, (mm, g) in zip(c, sub[sub["level"] == lvl].groupby("model")):
            clean = sub[(sub["model"] == mm) & (sub["level"] == 0)]["eval_qlike"].mean()
            col.metric(NICE[mm], f"{g['eval_qlike'].mean():.4f}", f"{g['eval_qlike'].mean() - clean:+.4f} vs clean", delta_color="inverse")
        for col, (mm, v) in zip(c[len(sub['model'].unique()):], refs.items()):
            col.metric(f"{NICE[mm]} (clean)", f"{v:.4f}")
        chart(sweep_plots.lines(sub, "level", title="Score as the damage grows", references=t[t["role"] == "reference"]))

    with st.expander("More charts"):
        figs = sweep_plots.figures(pick["path"])
        tabs = st.tabs([n.replace("_", " ") for n in figs])
        for tab, fig in zip(tabs, figs.values()):
            with tab:
                chart(fig)

st.divider()
st.subheader("Start a sweep")
c1, c2 = st.columns(2)
kind = c1.selectbox("Kind", list(WHAT), format_func=WHAT.get)
size = c2.segmented_control("Size", ["demo", "standard"], default="demo",
                            format_func={"demo": "Demo (seconds)", "standard": "Standard (real results)"}.get) or "demo"
if confirm("sweep_demo" if size == "demo" else f"sweep_{kind}") and st.button("Start sweep"):
    job = start_job(f"{kind} sweep ({size})", ["sweep", "--kind", kind] + (["--demo"] if size == "demo" else []))
    st.toast(f"Started: {job['command']}")

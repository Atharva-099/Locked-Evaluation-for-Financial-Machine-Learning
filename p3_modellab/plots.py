"""Charts for the report and the dashboard. One consistent look: colour-blind-safe model
colours, verdict colours (green better, red worse, grey unclear), clear hover labels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MODEL_COLORS = {"persistence": "#9E9E9E", "har": "#0072B2", "ridge": "#009E73", "lightgbm": "#E69F00",
                "xgboost": "#D55E00", "catboost": "#56B4E9", "extratrees": "#8C6D31", "mlp": "#7B61FF"}
VERDICT_COLORS = {"better": "#2E7D32", "worse": "#C62828", "no detectable difference": "#9E9E9E", "too little data": "#D0D0D0"}
NICE = {"persistence": "Persistence", "har": "HAR", "ridge": "Ridge", "lightgbm": "LightGBM",
        "xgboost": "XGBoost", "catboost": "CatBoost", "extratrees": "ExtraTrees", "mlp": "Neural net"}
ACTUAL_COLOR = "#CC79A7"  # visible on light and dark backgrounds
ERR = dict(color="rgba(150,150,150,0.9)")


def style(fig: go.Figure, title: str = "", height: int = 420, legend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16)),
        height=height, margin=dict(l=10, r=10, t=60 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1) if legend else None,
        showlegend=legend, hoverlabel=dict(font_size=13), font=dict(size=13),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def _verdict(lo: float, hi: float) -> str:
    return "better" if hi < 0 else "worse" if lo > 0 else "no detectable difference"


def load(report_dir: Path) -> dict:
    d = Path(report_dir)
    return {
        "comparisons": pd.read_json(d / "comparisons.json"),
        "summary": pd.read_json(d / "model_summary.json"),
        "by_year": pd.read_parquet(d / "by_year.parquet"),
        "slices": pd.read_parquet(d / "slices.parquet"),
        "worst": pd.read_parquet(d / "worst_cases.parquet"),
        "calibration": pd.read_parquet(d / "calibration.parquet"),
        "monthly": pd.read_parquet(d / "monthly_losses.parquet"),
        "skill": pd.read_parquet(d / "rank_skill.parquet"),
        "importance": pd.read_parquet(d / "importance.parquet"),
    }


def comparisons(comp: pd.DataFrame, view: str = "mean") -> go.Figure:
    c = comp.iloc[::-1]
    est, lo, hi = c[f"{view}_estimate"], c[f"{view}_ci_lo"], c[f"{view}_ci_hi"]
    verdicts = [_verdict(a, b) for a, b in zip(lo, hi)]
    labels = [f"{NICE[m]} vs {NICE[b]}" for m, b in zip(c["model"], c["baseline"])]
    fig = go.Figure(go.Scatter(
        x=est, y=labels, mode="markers+text", text=[f"{v:+.3f}" for v in est], textposition="top center",
        marker=dict(size=13, color=[VERDICT_COLORS[v] for v in verdicts], line=dict(width=1, color="white")),
        error_x=dict(type="data", symmetric=False, array=hi - est, arrayminus=est - lo, thickness=2.5, width=6, **ERR),
        customdata=np.stack([lo, hi, verdicts], axis=1),
        hovertemplate="<b>%{y}</b><br>difference %{x:+.4f}<br>95% interval [%{customdata[0]:+.4f}, %{customdata[1]:+.4f}]<br>%{customdata[2]}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_xaxes(title="QLIKE difference  (left of 0 = first model better)")
    return style(fig, "Head to head", height=380, legend=False)


def over_time(by_year: pd.DataFrame, baseline: str = "har") -> go.Figure:
    fig = go.Figure()
    for m, g in by_year[by_year["baseline"] == baseline].groupby("model"):
        fig.add_trace(go.Bar(x=g["year"], y=g["estimate"], name=NICE[m], marker_color=MODEL_COLORS[m],
                             error_y=dict(type="data", symmetric=False, array=g["ci_hi"] - g["estimate"], arrayminus=g["estimate"] - g["ci_lo"], thickness=1.2, **ERR),
                             hovertemplate=f"<b>{NICE[m]}</b> %{{x}}<br>difference %{{y:+.4f}}<extra></extra>"))
    fig.add_hline(y=0, line_color="grey")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title=f"vs {NICE[baseline]} (below 0 = better)")
    return style(fig, f"Year by year against {NICE[baseline]}", height=400)


def year_race(by_year: pd.DataFrame, baseline: str = "har") -> go.Figure:
    """Animated: each model's gap to the baseline, one frame per year (press play)."""
    d = by_year[by_year["baseline"] == baseline]
    models = sorted(d["model"].unique(), key=list(NICE).index)
    years = sorted(d["year"].unique())
    lo, hi = min(d["estimate"].min(), 0), max(d["estimate"].max(), 0)
    pad = 0.35 * (hi - lo)

    def bars(y: int) -> go.Bar:
        g = d[d["year"] == y].set_index("model").reindex(models)
        return go.Bar(x=[NICE[m] for m in models], y=g["estimate"], marker_color=[MODEL_COLORS[m] for m in models],
                      error_y=dict(type="data", symmetric=False, array=g["ci_hi"] - g["estimate"], arrayminus=g["estimate"] - g["ci_lo"], **ERR),
                      text=[f"{v:+.3f}" for v in g["estimate"]], textposition="outside", hovertemplate="%{x}: %{y:+.4f}<extra></extra>")

    fig = go.Figure(data=[bars(years[0])], frames=[go.Frame(data=[bars(y)], name=str(y)) for y in years])
    fig.update_layout(
        updatemenus=[dict(type="buttons", x=0, y=-0.12, xanchor="left", direction="left", bgcolor="#E8E8E8", font=dict(color="#111111"), buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=700, redraw=True), fromcurrent=True, transition=dict(duration=300))]),
            dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.2, len=0.8, y=-0.05, currentvalue=dict(prefix="Year: ", xanchor="right", font=dict(size=15)),
                      steps=[dict(label=str(y), method="animate", args=[[str(y)], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for y in years])])
    fig.add_hline(y=0, line_color="grey")
    fig.update_yaxes(range=[lo - pad, hi + pad], title=f"vs {NICE[baseline]} (below 0 = better)")
    return style(fig, f"Press play: each year against {NICE[baseline]}", height=460, legend=False)


def slices(sl: pd.DataFrame, model: str, baseline: str) -> go.Figure:
    s = sl[(sl["model"] == model) & (sl["baseline"] == baseline)].copy()
    s["label"] = s["slice"] + ": " + s["value"]
    s = s.iloc[::-1]
    fig = go.Figure()
    for v, g in s.groupby("verdict"):
        fig.add_trace(go.Scatter(x=g["diff"], y=g["label"], mode="markers", name=v, marker=dict(color=VERDICT_COLORS[v], size=9),
                                 error_x=dict(type="data", symmetric=False, array=g["ci_hi"] - g["diff"], arrayminus=g["diff"] - g["ci_lo"], **ERR),
                                 customdata=np.stack([g["n_obs"], g["n_firms"], g["n_months"]], axis=1),
                                 hovertemplate="<b>%{y}</b><br>difference %{x:+.4f}<br>%{customdata[0]:,} examples, %{customdata[1]:,} firms, %{customdata[2]} months<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_yaxes(categoryorder="array", categoryarray=list(s["label"]), dtick=1)
    ok = s[s["verdict"] != "too little data"]
    if len(ok):
        lo, hi = min(ok["ci_lo"].min(), 0), max(ok["ci_hi"].max(), 0)
        fig.update_xaxes(range=[lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)])
    fig.update_xaxes(title=f"left of 0 = {NICE[model]} better (bars for groups with too little data may run off the edge)")
    return style(fig, f"Where {NICE[model]} beats or loses to {NICE[baseline]}", height=max(500, 26 * len(s)))


def calibration(cal: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for m, g in cal.groupby("model"):
        fig.add_trace(go.Scatter(x=g["mean_forecast"], y=g["mean_answer"], mode="lines+markers", name=NICE[m], line=dict(color=MODEL_COLORS[m], width=2.5),
                                 hovertemplate=f"<b>{NICE[m]}</b><br>average forecast %{{x:.4f}}<br>average actual %{{y:.4f}}<extra></extra>"))
    lo, hi = cal[["mean_forecast", "mean_answer"]].min().min(), cal[["mean_forecast", "mean_answer"]].max().max()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect", line=dict(color="grey", dash="dash")))
    fig.update_xaxes(type="log", title="forecast (tenths of forecasts)")
    fig.update_yaxes(type="log", title="actual")
    return style(fig, "Are forecasts the right size?", height=420)


def importance(imp: pd.DataFrame, kind: str = "group") -> go.Figure:
    s = imp[imp["kind"] == kind]
    order = s.groupby("input")["mean"].max().sort_values().index.tolist()
    fig = go.Figure()
    for m, g in s.groupby("model"):
        g = g.set_index("input").reindex(order).reset_index()
        fig.add_trace(go.Bar(x=g["mean"], y=g["input"], orientation="h", name=NICE[m], marker_color=MODEL_COLORS[m],
                             error_x=dict(type="data", array=g["std"], thickness=1, **ERR),
                             hovertemplate=f"<b>{NICE[m]}</b> %{{y}}<br>damage when scrambled %{{x:+.4f}}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="how much worse when scrambled")
    return style(fig, "What each model relies on" + ("" if kind == "group" else " (single inputs)"), height=260 + 26 * s["input"].nunique())


def rank_skill(skill: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for m, g in skill.groupby("model"):
        g = g.sort_values("yyyymm")
        x = pd.to_datetime(g["yyyymm"].astype(str), format="%Y%m")
        fig.add_trace(go.Scatter(x=x, y=g["spearman"].rolling(12, min_periods=6).mean(), name=NICE[m], line=dict(color=MODEL_COLORS[m], width=2.5),
                                 hovertemplate="%{y:.3f}"))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="rank correlation (12-month average)")
    return style(fig, "Ranking skill over time", height=400)


def stock_chart(hist: pd.DataFrame, models: list[str], title: str) -> go.Figure:
    """Actual yearly volatility vs each model's forecast, month by month; biggest misses marked."""
    x = pd.to_datetime(hist["yyyymm"].astype(str), format="%Y%m")
    vol = lambda v: np.sqrt(12 * v)
    fig = go.Figure(go.Scatter(x=x, y=vol(hist["target_rv"]), name="Actual", mode="lines+markers",
                               line=dict(color=ACTUAL_COLOR, width=3), marker=dict(size=5), hovertemplate="%{y:.0%}"))
    for m in models:
        fig.add_trace(go.Scatter(x=x, y=vol(hist[f"f_{m}"]), name=NICE[m], mode="lines", line=dict(color=MODEL_COLORS[m], width=2),
                                 hovertemplate="%{y:.0%}"))
    if models and len(hist):
        m = models[-1]
        worst = hist.nlargest(min(5, len(hist)), f"q_{m}")
        fig.add_trace(go.Scatter(x=pd.to_datetime(worst["yyyymm"].astype(str), format="%Y%m"), y=vol(worst["target_rv"]), mode="markers",
                                 name=f"{NICE[m]}'s 5 worst misses", marker=dict(symbol="x", size=13, color="#C62828", line=dict(width=2)),
                                 hovertemplate="big miss<extra></extra>"))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="yearly volatility (forecast month)", tickformat=".0%")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    return style(fig, title, height=480)


def custom_monthly(monthly: pd.Series, a: str, b: str) -> go.Figure:
    x = pd.to_datetime(monthly.index.astype(str), format="%Y%m")
    colors = np.where(monthly.to_numpy() < 0, VERDICT_COLORS["better"], VERDICT_COLORS["worse"])
    fig = go.Figure(go.Bar(x=x, y=monthly.to_numpy(), marker_color=colors, name="each month", hovertemplate="%{x|%b %Y}: %{y:+.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=monthly.cumsum().to_numpy(), name="running total", yaxis="y2", line=dict(color=ACTUAL_COLOR, width=2),
                             hovertemplate="running total %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=0, line_color="grey")
    style(fig, "Month by month (green = " + NICE[a] + " better)", height=420)
    fig.update_layout(yaxis=dict(title=f"{NICE[a]} minus {NICE[b]}"), yaxis2=dict(overlaying="y", side="right", showgrid=False, title="running total"))
    return fig


def month_scatter(snap: pd.DataFrame, models: list[str], label: str) -> go.Figure:
    fig = go.Figure()
    vol = lambda v: np.sqrt(12 * np.maximum(v, 1e-8))
    for m in models:
        fig.add_trace(go.Scatter(x=vol(snap[f"f_{m}"]), y=vol(snap["target_rv"]), mode="markers", name=NICE[m], text=snap["ticker"],
                                 marker=dict(color=MODEL_COLORS[m], size=6, opacity=0.55),
                                 hovertemplate="<b>%{text}</b><br>forecast %{x:.0%}<br>actual %{y:.0%}<extra>" + NICE[m] + "</extra>"))
    if len(snap):
        lo, hi = 0.03, float(vol(snap["target_rv"]).max())
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect", line=dict(color="grey", dash="dash")))
    fig.update_xaxes(type="log", title="forecast yearly volatility", tickformat=".0%")
    fig.update_yaxes(type="log", title="actual yearly volatility", tickformat=".0%")
    return style(fig, f"{label}: forecast vs actual (sample of stocks)", height=480)


def loss_timeline(ml: pd.DataFrame, selected: int | None = None, title: str | None = None) -> go.Figure:
    x = pd.to_datetime(ml.index.astype(str), format="%Y%m")
    fig = go.Figure()
    for m in ml.columns:
        fig.add_trace(go.Scatter(x=x, y=ml[m], name=NICE.get(m, m), line=dict(color=MODEL_COLORS.get(m), width=2), hovertemplate="%{y:.3f}",
                                 visible="legendonly" if m == "persistence" else True))
    shown = ml.drop(columns=["persistence"], errors="ignore")
    shown = shown if shown.shape[1] else ml
    lo, hi = np.nanquantile(shown.to_numpy(), [0.01, 0.97])
    if selected is not None:
        fig.add_vline(x=pd.to_datetime(str(selected), format="%Y%m"), line_color="#C62828", line_width=2)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="average error that month (lower better)", range=[lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)])
    default = "Every month's error (red line = selected month; click Persistence to show it)" if selected is not None else "Error month by month"
    return style(fig, title or default, height=380)


def _hover_extra(series: pd.DataFrame, scope: str) -> list[str]:
    if scope == "stock":
        if "share_price" not in series.columns:
            return [""] * len(series)
        return [f"share price ${p:,.2f}" if pd.notna(p) else "" for p in series["share_price"]]
    if {"market_vol", "mood", "stocks"}.issubset(series.columns):
        return [f"market swing {v:.0%} ({str(m).split(' (')[0]})<br>{n:,} stocks" for v, m, n in zip(series["market_vol"], series["mood"], series["stocks"])]
    if "stocks" in series.columns:
        return [f"{int(n):,} stocks" for n in series["stocks"]]
    return [""] * len(series)


def main_chart(series: pd.DataFrame, models: list[str], view: str = "2D", scope: str = "all", title: str = "") -> go.Figure:
    """Actual vs forecast swing over time. The same data as a 2D line chart or 3D ribbons (one ribbon per series)."""
    lanes = [("actual", "Actual", ACTUAL_COLOR)] + [(m, NICE[m], MODEL_COLORS[m]) for m in models]
    extra = _hover_extra(series, scope)
    dates = pd.to_datetime(series.index.astype(str), format="%Y%m")
    if view == "2D":
        fig = go.Figure()
        for key, name, color in lanes:
            is_actual = key == "actual"
            fig.add_trace(go.Scatter(
                x=dates, y=series[key], name=name, mode="lines", line=dict(color=color, width=3.2 if is_actual else 2),
                customdata=extra, hovertemplate=(f"<b>{name}</b> %{{y:.0%}}<br>%{{customdata}}<extra></extra>" if is_actual
                                                 else f"<b>{name}</b> %{{y:.0%}}<extra></extra>")))
        fig.update_layout(hovermode="x unified")
        fig.update_yaxes(title="yearly swing (volatility)", tickformat=".0%")
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
        return style(fig, title, height=500)

    xf = (series.index // 100 + (series.index % 100 - 0.5) / 12).to_numpy(dtype=float)
    n = len(xf)
    fig = go.Figure()
    text = [f"{d:%b %Y}" for d in dates]
    top = float(np.nanmax(series[[k for k, _, _ in lanes]].to_numpy()))
    for i, (key, name, color) in enumerate(lanes):
        z = series[key].to_numpy(dtype=float)
        # a curtain filled down to zero: the top edge is the line, so ups and downs read like the 2D chart
        fig.add_trace(go.Surface(x=np.vstack([xf, xf]), y=np.full((2, n), float(i)), z=np.vstack([np.zeros(n), z]),
                                 surfacecolor=np.zeros((2, n)), cmin=0, cmax=1, colorscale=[[0, color], [1, color]], showscale=False,
                                 opacity=0.55, hoverinfo="skip", showlegend=False, lighting=dict(ambient=0.9, diffuse=0.3)))
        fig.add_trace(go.Scatter3d(x=xf, y=np.full(n, float(i)), z=z, mode="lines", name=name, line=dict(color=color, width=7),
                                   customdata=np.stack([text, extra], axis=1),
                                   hovertemplate=f"<b>{name}</b> %{{z:.0%}}<br>%{{customdata[0]}}<br>%{{customdata[1]}}<extra></extra>"))
    fig.update_layout(scene=dict(
        xaxis=dict(title="year"), yaxis=dict(title="", tickvals=list(range(len(lanes))), ticktext=[l[1] for l in lanes],
                                               range=[-0.6, len(lanes) - 0.4]),
        zaxis=dict(title="yearly swing", tickformat=".0%", range=[0, top * 1.05]), aspectmode="manual",
        aspectratio=dict(x=2.0, y=0.35 + 0.25 * len(lanes), z=0.8),
        camera=dict(eye=dict(x=-1.5, y=-2.35, z=0.95), center=dict(x=0.05, y=0, z=-0.12))))
    style(fig, title, height=620)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig


def all_figures(report_dir: Path) -> dict:
    r = load(report_dir)
    figs = {
        "comparisons_mean": comparisons(r["comparisons"], "mean"),
        "comparisons_median": comparisons(r["comparisons"], "median"),
        "comparisons_trimmed": comparisons(r["comparisons"], "trimmed"),
        "over_time_vs_har": over_time(r["by_year"], "har"),
        "over_time_vs_persistence": over_time(r["by_year"], "persistence"),
        "year_race_vs_har": year_race(r["by_year"], "har"),
        "calibration": calibration(r["calibration"]),
        "importance_groups": importance(r["importance"], "group"),
        "importance_inputs": importance(r["importance"], "feature"),
        "rank_skill": rank_skill(r["skill"]),
    }
    for m, b in r["slices"][["model", "baseline"]].drop_duplicates().itertuples(index=False):
        figs[f"slices_{m}_vs_{b}"] = slices(r["slices"], m, b)
    return figs


def write_all(report_dir: Path) -> list[Path]:
    figs = all_figures(report_dir)
    figdir = Path(report_dir) / "figures"
    figdir.mkdir(exist_ok=True)
    paths = []
    for name, fig in figs.items():
        p = figdir / f"{name}.html"
        fig.write_html(p, include_plotlyjs="cdn")
        paths.append(p)
    return paths

"""Charts for sweeps: 3D surfaces (always with a matching heatmap), lines with seed bands,
box plots over seeds, parallel coordinates, and the greedy input path."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .plots import MODEL_COLORS, NICE

NICE_SWEEP = {**NICE, "constant": "No inputs"}


def load(sweep_dir: Path) -> dict:
    d = Path(sweep_dir)
    out = {"manifest": json.loads((d / "manifest.json").read_text()), "trials": pd.read_parquet(d / "trials.parquet"),
           "monthly": pd.read_parquet(d / "monthly.parquet")}
    if (d / "candidates.parquet").exists():
        out["candidates"] = pd.read_parquet(d / "candidates.parquet")
    return out


def _grid(t: pd.DataFrame, x: str, y: str, z: str) -> pd.DataFrame:
    return t.groupby([y, x])[z].mean().unstack(x).sort_index().sort_index(axis=1)


def _best_by_validation(t: pd.DataFrame, x: str, y: str) -> tuple:
    g = t.groupby([x, y])["val_qlike"].mean()
    return g.idxmin()


def surface(t: pd.DataFrame, x: str, y: str, z: str = "eval_qlike", title: str = "", log_x: bool = False, log_y: bool = False,
            highlight: tuple | None = None) -> go.Figure:
    grid = _grid(t, x, y, z)
    xs = np.log2(grid.columns.to_numpy(dtype=float)) if log_x else grid.columns.to_numpy(dtype=float)
    ys = np.log10(grid.index.to_numpy(dtype=float)) if log_y else grid.index.to_numpy(dtype=float)
    fig = go.Figure(go.Surface(x=xs, y=ys, z=grid.to_numpy(), colorscale="Viridis_r", opacity=0.92,
                               colorbar=dict(title="score", len=0.6, thickness=14),
                               hovertemplate=f"{x} %{{x}}<br>{y} %{{y}}<br>score %{{z:.4f}}<extra></extra>"))
    if highlight is not None:
        hx, hy = highlight
        hz = t[(t[x] == hx) & (t[y] == hy)][z].mean()
        fig.add_trace(go.Scatter3d(x=[np.log2(hx) if log_x else hx], y=[np.log10(hy) if log_y else hy], z=[hz], mode="markers",
                                   marker=dict(size=11, color="#FFD400", symbol="circle", line=dict(width=2, color="black")), name="your pick"))
    pts = t.assign(_x=np.log2(t[x]) if log_x else t[x], _y=np.log10(t[y]) if log_y else t[y])
    fig.add_trace(go.Scatter3d(x=pts["_x"], y=pts["_y"], z=pts[z], mode="markers", marker=dict(size=3, color="black"), name="each seed"))
    bx, by = _best_by_validation(t, x, y)
    bz = t[(t[x] == bx) & (t[y] == by)][z].mean()
    fig.add_trace(go.Scatter3d(x=[np.log2(bx) if log_x else bx], y=[np.log10(by) if log_y else by], z=[bz], mode="markers",
                               marker=dict(size=8, color="red", symbol="diamond"), name="chosen on validation"))
    fig.update_layout(title=title, height=640, margin=dict(l=0, r=0, t=50, b=0),
                      legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="center", x=0.5),
                      scene=dict(xaxis=dict(title=f"log2({x})" if log_x else x), yaxis=dict(title=f"log10({y})" if log_y else y),
                                 zaxis=dict(title=f"{z} (lower better)"), aspectmode="cube",
                                 camera=dict(eye=dict(x=1.6, y=1.6, z=1.0))))
    return fig


def heatmap(t: pd.DataFrame, x: str, y: str, z: str = "eval_qlike", title: str = "") -> go.Figure:
    grid = _grid(t, x, y, z)
    fig = go.Figure(go.Heatmap(z=grid.to_numpy(), x=[str(v) for v in grid.columns], y=[str(v) for v in grid.index], colorscale="Viridis_r",
                               text=np.round(grid.to_numpy(), 4), texttemplate="%{text}", colorbar=dict(title=z)))
    bx, by = _best_by_validation(t, x, y)
    fig.add_trace(go.Scatter(x=[str(bx)], y=[str(by)], mode="markers", marker=dict(symbol="square-open", size=40, color="red", line=dict(width=3)),
                             name="chosen on validation"))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=520,
                      legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5))
    return fig


def lines(t: pd.DataFrame, x: str, y: str = "eval_qlike", group: str = "model", title: str = "", log_x: bool = False,
          references: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    for g, s in t.groupby(group):
        a = s.groupby(x)[y].agg(["mean", "min", "max"]).reset_index().sort_values(x)
        color = MODEL_COLORS.get(g) if group == "model" else None
        fig.add_trace(go.Scatter(x=list(a[x]) + list(a[x][::-1]), y=list(a["max"]) + list(a["min"][::-1]), fill="toself",
                                 fillcolor=color or "grey", opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=a[x], y=a["mean"], mode="lines+markers", name=NICE_SWEEP.get(g, str(g)), line=dict(color=color)))
    if references is not None:
        for _, r in references.iterrows():
            fig.add_hline(y=r["eval_qlike"], line_dash="dot", line_color=MODEL_COLORS.get(r["model"], "black"),
                          annotation_text=f"{NICE_SWEEP[r['model']]} (clean)", annotation_position="top left")
    fig.update_layout(title=title + " (line = average over seeds, band = lowest to highest seed)", xaxis_title=x,
                      yaxis_title=f"{y} (lower better)", height=500, legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def box(t: pd.DataFrame, x: str, y: str = "eval_qlike", group: str = "model", title: str = "") -> go.Figure:
    fig = go.Figure()
    for g, s in t.groupby(group):
        fig.add_trace(go.Box(x=s[x].astype(str), y=s[y], name=NICE_SWEEP.get(g, str(g)), marker_color=MODEL_COLORS.get(g), boxpoints="all"))
    fig.update_layout(title=title + " (spread across seeds)", xaxis_title=x, yaxis_title=f"{y} (lower better)", boxmode="group", height=460)
    return fig


def parallel(t: pd.DataFrame, dims: list[str], color: str = "eval_qlike", title: str = "") -> go.Figure:
    fig = go.Figure(go.Parcoords(line=dict(color=t[color], colorscale="Viridis_r", showscale=True, colorbar=dict(title=color)),
                                 dimensions=[dict(label=d, values=t[d]) for d in dims + [color]]))
    fig.update_layout(title=title, height=460)
    return fig


def greedy_path(t: pd.DataFrame, title: str = "Adding inputs one at a time (greedy)") -> go.Figure:
    fig = go.Figure()
    for model, s in t.groupby("model"):
        a = s.groupby("step").agg(mean=("eval_qlike", "mean"), lo=("eval_qlike", "min"), hi=("eval_qlike", "max"), val=("val_qlike", "mean")).reset_index()
        first = s[s["seed"] == s["seed"].min()].set_index("step")["added"]
        color = MODEL_COLORS.get(model)
        fig.add_trace(go.Scatter(x=list(a["step"]) + list(a["step"][::-1]), y=list(a["hi"]) + list(a["lo"][::-1]), fill="toself",
                                 fillcolor=color, opacity=0.15, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=a["step"], y=a["mean"], mode="lines+markers+text", name=f"{NICE_SWEEP[model]} (evaluation)",
                                 text=[first.get(k, "") for k in a["step"]], textposition="top center", line=dict(color=color)))
        fig.add_trace(go.Scatter(x=a["step"], y=a["val"], mode="lines", name=f"{NICE_SWEEP[model]} (validation, used to choose)",
                                 line=dict(color=color, dash="dot")))
    fig.update_layout(title=title, xaxis_title="Number of inputs (label = input added at that step, first seed)",
                      yaxis_title="QLIKE (lower better)", height=560, legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5))
    return fig


def greedy_candidates(c: pd.DataFrame, model: str, seed: int) -> go.Figure:
    s = c[(c["model"] == model) & (c["seed"] == seed)]
    grid = s.pivot(index="candidate", columns="step", values="val_qlike")
    fig = go.Figure(go.Heatmap(z=grid.to_numpy(), x=[str(v) for v in grid.columns], y=list(grid.index), colorscale="Viridis_r",
                               colorbar=dict(title="validation QLIKE")))
    fig.update_layout(title=f"{NICE_SWEEP[model]}: validation score of every candidate input at every step (seed {seed}); blank = already chosen",
                      xaxis_title="Step", yaxis_title="Candidate input", height=520)
    return fig


def figures(sweep_dir: Path) -> dict:
    s = load(sweep_dir)
    kind, t = s["manifest"]["kind"], s["trials"]
    refs = t[t["role"] == "reference"]
    body = t[t["role"].isna()]
    figs = {}
    if kind == "settings":
        lg = body[body["model"] == "lightgbm"]
        figs["surface_lightgbm"] = surface(lg, "num_leaves", "learning_rate", title="LightGBM: tree size x learning speed (evaluation score)", log_x=True, log_y=True)
        figs["heatmap_lightgbm"] = heatmap(lg, "num_leaves", "learning_rate", title="LightGBM: same numbers as a heatmap (red box = chosen on validation)")
        figs["heatmap_lightgbm_validation"] = heatmap(lg, "num_leaves", "learning_rate", z="val_qlike", title="LightGBM: validation score (what the choice was based on)")
        figs["box_lightgbm_leaves"] = box(lg, "num_leaves", title="LightGBM by tree size")
        figs["parallel_lightgbm"] = parallel(lg, ["num_leaves", "learning_rate", "chosen_best_iteration"], title="LightGBM settings, one line per trial")
        figs["lines_ridge_alpha"] = lines(body[body["model"] == "ridge"], "alpha", title="Ridge: penalty strength", log_x=True, references=refs)
    elif kind == "greedy":
        figs["greedy_path"] = greedy_path(body)
        for model in body["model"].unique():
            seed = int(body[body["model"] == model]["seed"].min())
            figs[f"greedy_candidates_{model}"] = greedy_candidates(s["candidates"], model, seed)
    elif kind == "data":
        for model in body["model"].unique():
            figs[f"surface_{model}"] = surface(body[body["model"] == model], "train_years", "firm_percent",
                                               title=f"{NICE_SWEEP[model]}: years of training data x share of firms (evaluation score)")
            figs[f"heatmap_{model}"] = heatmap(body[body["model"] == model], "train_years", "firm_percent", title=f"{NICE_SWEEP[model]}: same numbers as a heatmap")
        full = body[body["firm_percent"] == body["firm_percent"].max()]
        figs["lines_train_years"] = lines(full, "train_years", title="Learning curve: years of training data (all firms)", references=refs)
        last = body[body["train_years"] == body["train_years"].max()]
        figs["lines_firm_percent"] = lines(last, "firm_percent", title="Learning curve: share of firms (longest window)", references=refs)
    elif kind == "noise":
        for kind_c, label in (("noise", "Noise added to inputs (in units of each input's spread)"), ("blank", "Share of input values blanked out")):
            figs[f"lines_{kind_c}"] = lines(body[body["corruption"] == kind_c], "level", title=label, references=refs)
            figs[f"box_{kind_c}"] = box(body[body["corruption"] == kind_c], "level", title=label)
    return figs


def write_figures(sweep_dir: Path) -> list[Path]:
    figs = figures(sweep_dir)
    figdir = Path(sweep_dir) / "figures"
    figdir.mkdir(exist_ok=True)
    paths = []
    for name, fig in figs.items():
        p = figdir / f"{name}.html"
        fig.write_html(p, include_plotlyjs="cdn")
        paths.append(p)
    return paths

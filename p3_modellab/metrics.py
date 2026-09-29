"""Scores for saved forecasts, and paired comparisons between models.

Every comparison uses exactly the same examples for both models. Per-example
loss differences are averaged within each month, and every month counts equally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .losses import qlike, squared_error
from .models import FLOOR
from .uncertainty import block_bootstrap, holm, monthly_means

PRIMARY_BLOCK = 12
SENSITIVITY_BLOCKS = (3, 6, 12, 24)
TRIM_QUANTILE = 0.999
BASELINES = ("persistence", "har")


def default_pairs(models) -> list[tuple[str, str]]:
    """HAR vs persistence; every learned model vs persistence, vs HAR, and (other than Ridge) vs Ridge."""
    learned = [m for m in models if m not in BASELINES]
    pairs = [("har", "persistence")] if {"har", "persistence"} <= set(models) else []
    pairs += [(m, b) for m in learned for b in BASELINES if b in models]
    if "ridge" in models:
        pairs += [(m, "ridge") for m in learned if m != "ridge"]
    return pairs


def add_losses(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    y, h = out["target_rv"].to_numpy(), out["forecast"].to_numpy()
    out["qlike"] = qlike(y, h)
    out["se"] = squared_error(y, h)
    out["log_se"] = (np.log(np.maximum(y, FLOOR)) - np.log(h)) ** 2
    return out


def wide(pred: pd.DataFrame, value: str) -> pd.DataFrame:
    """One row per example, one column per model. Refuses if models were scored on different examples."""
    sets = pred.groupby("model")["row_id"].apply(lambda s: frozenset(s))
    if len(set(sets)) != 1:
        raise ValueError("models were not scored on identical examples")
    w = pred.pivot(index="row_id", columns="model", values=value)
    meta = pred.drop_duplicates("row_id").set_index("row_id")[["yyyymm", "target_rv"]]
    return w.join(meta)


def rank_skill(pred: pd.DataFrame) -> pd.DataFrame:
    """Per model and month: Spearman correlation between forecast and answer (ability to rank stocks)."""
    rows = []
    for (model, ym), g in pred.groupby(["model", "yyyymm"]):
        rows.append((model, ym, spearmanr(g["forecast"], g["target_rv"]).statistic if len(g) > 2 else np.nan))
    return pd.DataFrame(rows, columns=["model", "yyyymm", "spearman"])


def model_summary(pred: pd.DataFrame, skill: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in pred.groupby("model"):
        s = skill[skill["model"] == m]["spearman"]
        rows.append({
            "model": m,
            "qlike": monthly_means(g["qlike"].to_numpy(), g["yyyymm"].to_numpy()).mean(),
            "mse": monthly_means(g["se"].to_numpy(), g["yyyymm"].to_numpy()).mean(),
            "log_mse": monthly_means(g["log_se"].to_numpy(), g["yyyymm"].to_numpy()).mean(),
            "rank_corr": s.mean(),
            "rank_corr_ci": block_bootstrap(s.to_numpy(), block=PRIMARY_BLOCK).as_dict(),
            "n_examples": int(len(g)),
            "n_months": int(g["yyyymm"].nunique()),
        })
    return pd.DataFrame(rows)


def compare(pred: pd.DataFrame, pairs: list[tuple[str, str]] | None = None, seed: int = 0) -> pd.DataFrame:
    """Paired differences (model minus baseline; negative means the model is better) in several views."""
    pairs = pairs if pairs is not None else default_pairs(pred["model"].unique())
    q, se, lse = wide(pred, "qlike"), wide(pred, "se"), wide(pred, "log_se")
    months = q["yyyymm"].to_numpy()
    keep = q["target_rv"] <= q["target_rv"].quantile(TRIM_QUANTILE)
    rows = []
    for model, base in pairs:
        if model not in q or base not in q:
            continue
        d = (q[model] - q[base]).to_numpy()
        dm = monthly_means(d, months)
        row = {"model": model, "baseline": base, "metric": "qlike"}
        row.update({f"mean_{k}": v for k, v in block_bootstrap(dm.to_numpy(), block=PRIMARY_BLOCK, seed=seed).as_dict().items()})
        for b in SENSITIVITY_BLOCKS:
            iv = block_bootstrap(dm.to_numpy(), block=b, seed=seed)
            row[f"ci_block{b}"] = (iv.lo, iv.hi)
        row.update({f"median_{k}": v for k, v in block_bootstrap(dm.to_numpy(), stat=np.median, block=PRIMARY_BLOCK, seed=seed).as_dict().items()})
        dt = monthly_means(d[keep.to_numpy()], months[keep.to_numpy()])
        row.update({f"trimmed_{k}": v for k, v in block_bootstrap(dt.to_numpy(), block=PRIMARY_BLOCK, seed=seed).as_dict().items()})
        row["months_model_better"] = float((dm < 0).mean())
        row["mse_diff"] = block_bootstrap(monthly_means((se[model] - se[base]).to_numpy(), months).to_numpy(), block=PRIMARY_BLOCK, seed=seed).as_dict()
        row["log_mse_diff"] = block_bootstrap(monthly_means((lse[model] - lse[base]).to_numpy(), months).to_numpy(), block=PRIMARY_BLOCK, seed=seed).as_dict()
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["mean_p_holm"] = holm(out["mean_p_value"].tolist())
    return out


def by_year(pred: pd.DataFrame, pairs: list[tuple[str, str]] | None = None, seed: int = 0) -> pd.DataFrame:
    pairs = pairs if pairs is not None else default_pairs(pred["model"].unique())
    q = wide(pred, "qlike")
    rows = []
    for model, base in pairs:
        if model not in q or base not in q:
            continue
        d = monthly_means((q[model] - q[base]).to_numpy(), q["yyyymm"].to_numpy())
        for year, s in d.groupby(d.index // 100):
            iv = block_bootstrap(s.to_numpy(), block=3, seed=seed)
            rows.append({"model": model, "baseline": base, "year": int(year), **iv.as_dict()})
    return pd.DataFrame(rows)


def monthly_losses(pred: pd.DataFrame) -> pd.DataFrame:
    """Average loss per model per month, for charts."""
    return pred.groupby(["model", "yyyymm"])[["qlike", "se", "log_se"]].mean().reset_index()

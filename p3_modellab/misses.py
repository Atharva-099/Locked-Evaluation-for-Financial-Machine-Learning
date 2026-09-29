"""Where does a model improve on its baseline, where does it fail, and how sure are we?

For each group of examples ("slice": a year, an industry, a size bucket...) we
compare model and baseline on the same examples, with a block-bootstrap interval
on the gap, and give a verdict:
- better: whole interval below zero (model's loss lower)
- worse: whole interval above zero
- no detectable difference: interval spans zero
- too little data: fewer than the minimum examples, firms or months
Many slices are checked at once, so a few "better"/"worse" verdicts are expected
by chance alone; these are exploratory findings to re-check on later data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.industry import ff12
from .uncertainty import block_bootstrap, monthly_means

MIN_OBS, MIN_FIRMS, MIN_MONTHS = 2000, 50, 24
MIN_MONTHS_YEAR_SLICE = 10
BASELINES = ("persistence", "har")


def slice_pairs(models) -> list[tuple[str, str]]:
    """Every learned model vs HAR, and HAR vs persistence."""
    pairs = [(m, "har") for m in models if m not in BASELINES and "har" in models]
    return pairs + ([("har", "persistence")] if {"har", "persistence"} <= set(models) else [])

SLICES = {
    "year": "Test year",
    "industry": "Fama-French 12 industry (from SIC code)",
    "size": "Market value quintile within the month (Q1 smallest)",
    "price": "Share price at the forecast",
    "exchange": "Listing exchange",
    "age": "Years since first listed",
    "missing_days": "Share of the last 22 trading days with a return",
    "market_regime": "Market-wide volatility at the forecast (annualised)",
    "extreme_answer": "RETROSPECTIVE: answer in the top 1 percent of its month (uses the outcome; diagnostic only)",
}
RETROSPECTIVE = {"extreme_answer"}


def add_slices(df: pd.DataFrame, first_listed: pd.Series) -> pd.DataFrame:
    """Add one column per slice. Inputs: panel columns joined to predictions, and each permno's first listing date."""
    out = df.copy()
    out["year"] = (out["yyyymm"] // 100).astype(int).astype(str)
    out["industry"] = ff12(out["siccd"])
    q = out.groupby("yyyymm")["log_mcap"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) if s.notna().sum() >= 5 else np.nan)
    out["size"] = ("Q" + (q + 1).astype("Int64").astype(str)).where(q.notna(), "unknown")
    price = np.round(np.exp(out["log_price"]), 4)  # undo log-scale rounding so $5.00 is not read as 4.9999999
    out["price"] = np.select([price < 5, price < 20, price >= 20], ["under $5", "$5 to $20", "$20 and up"], default="unknown")
    out["exchange"] = out["exchcd"].map({1: "NYSE", 2: "AMEX", 3: "Nasdaq"}).fillna("other")
    age = (out["prediction_time"] - out["permno"].map(first_listed)).dt.days / 365.25
    out["age"] = pd.cut(age, [-np.inf, 2, 5, 10, 20, np.inf], labels=["under 2y", "2 to 5y", "5 to 10y", "10 to 20y", "20y and up"]).astype(str).replace("nan", "unknown")
    out["missing_days"] = np.select([out["frac_valid_m"] >= 1, out["frac_valid_m"] >= 0.9], ["none missing", "1 or 2 missing"], default="3 or more missing")
    mvol = np.sqrt(np.exp(out["mkt_log_rv_m"]) * 12)
    out["market_regime"] = np.select([mvol < 0.15, mvol < 0.25], ["calm (under 15%)", "normal (15 to 25%)"], default="stressed (25% and up)")
    top = out.groupby("yyyymm")["target_rv"].transform(lambda s: s >= s.quantile(0.99))
    out["extreme_answer"] = np.where(top, "top 1% answer", "other")
    return out


def verdict(n_obs: int, n_firms: int, n_months: int, lo: float, hi: float, min_months: int) -> str:
    if n_obs < MIN_OBS or n_firms < MIN_FIRMS or n_months < min_months or np.isnan(lo):
        return "too little data"
    if hi < 0:
        return "better"
    if lo > 0:
        return "worse"
    return "no detectable difference"


def slice_table(df: pd.DataFrame, loss: str = "qlike", pairs: list[tuple[str, str]] | None = None, seed: int = 0) -> pd.DataFrame:
    """df: one row per (example, model) with slice columns and a loss column."""
    pairs = pairs if pairs is not None else slice_pairs(df["model"].unique())
    w = df.pivot(index="row_id", columns="model", values=loss)
    meta = df.drop_duplicates("row_id").set_index("row_id")[["yyyymm", "permno", *SLICES]]
    w = w.join(meta)
    rows = []
    for model, base in pairs:
        if model not in w or base not in w:
            continue
        d_all = w[model] - w[base]
        for name in SLICES:
            block = 3 if name == "year" else 12
            min_months = MIN_MONTHS_YEAR_SLICE if name == "year" else MIN_MONTHS
            for value, idx in w.groupby(name).groups.items():
                g = w.loc[idx]
                dm = monthly_means(d_all.loc[idx].to_numpy(), g["yyyymm"].to_numpy())
                iv = block_bootstrap(dm.to_numpy(), block=block, seed=seed)
                n_obs, n_firms, n_months = len(g), g["permno"].nunique(), g["yyyymm"].nunique()
                rows.append({
                    "model": model, "baseline": base, "slice": name, "value": str(value),
                    "retrospective": name in RETROSPECTIVE,
                    "n_obs": n_obs, "n_firms": n_firms, "n_months": n_months,
                    "model_loss": monthly_means(g[model].to_numpy(), g["yyyymm"].to_numpy()).mean(),
                    "baseline_loss": monthly_means(g[base].to_numpy(), g["yyyymm"].to_numpy()).mean(),
                    "diff": iv.estimate, "ci_lo": iv.lo, "ci_hi": iv.hi,
                    "verdict": verdict(n_obs, n_firms, n_months, iv.lo, iv.hi, min_months),
                })
    return pd.DataFrame(rows)


def worst_cases(df: pd.DataFrame, pairs: list[tuple[str, str]] | None = None, n: int = 25) -> pd.DataFrame:
    """Examples where each model did much worse, and much better, than its baseline (QLIKE gap)."""
    pairs = pairs if pairs is not None else [(m, "har") for m in df["model"].unique() if m not in BASELINES]
    w = df.pivot(index="row_id", columns="model", values="qlike")
    f = df.pivot(index="row_id", columns="model", values="forecast").add_prefix("forecast_")
    meta = df.drop_duplicates("row_id").set_index("row_id")[["permno", "ticker", "yyyymm", "target_rv", "industry"]]
    base = w.join(f).join(meta)
    out = []
    for model, b in pairs:
        if model not in w or b not in w:
            continue
        gap = (w[model] - w[b]).rename("qlike_gap")
        t = base.join(gap)
        for kind, sub in (("model much worse", t.nlargest(n, "qlike_gap")), ("model much better", t.nsmallest(n, "qlike_gap"))):
            out.append(sub.assign(model=model, baseline=b, kind=kind).reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def calibration(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Per model: group forecasts into deciles and compare the average forecast with the average and median answer."""
    rows = []
    for m, g in df.groupby("model"):
        b = pd.qcut(g["forecast"].rank(method="first"), bins, labels=False)
        for k, s in g.groupby(b):
            rows.append({"model": m, "bin": int(k) + 1, "n": len(s), "mean_forecast": s["forecast"].mean(),
                         "mean_answer": s["target_rv"].mean(), "median_answer": s["target_rv"].median()})
    return pd.DataFrame(rows)

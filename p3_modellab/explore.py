"""Data behind the interactive dashboard pages. Everything is computed from a saved run
(forecasts) plus the saved table; nothing is retrained, so answers come back in seconds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import misses
from .losses import qlike
from .uncertainty import block_bootstrap, monthly_means

FILTERS = {"size": "Size (within month)", "price": "Share price", "industry": "Industry", "exchange": "Exchange"}
PANEL_COLS = ["row_id", "ticker", "exchcd", "siccd", "log_mcap", "log_price", "frac_valid_m", "mkt_log_rv_m", "prediction_time"]


def explore_table(run_dir: Path, panel: pd.DataFrame, first_listed: pd.Series) -> pd.DataFrame:
    """One row per scored example: answer, each model's forecast (f_<model>) and QLIKE loss (q_<model>), plus groups."""
    pred = pd.read_parquet(Path(run_dir) / "predictions.parquet", columns=["row_id", "permno", "yyyymm", "model", "forecast", "target_rv"])
    pred["q"] = qlike(pred["target_rv"].to_numpy(), pred["forecast"].to_numpy())
    f = pred.pivot(index="row_id", columns="model", values="forecast").add_prefix("f_")
    q = pred.pivot(index="row_id", columns="model", values="q").add_prefix("q_")
    base = pred.drop_duplicates("row_id").set_index("row_id")[["permno", "yyyymm", "target_rv"]]
    t = base.join(f).join(q).reset_index()
    t = t.merge(panel[PANEL_COLS], on="row_id", how="left", validate="one_to_one")
    t = misses.add_slices(t, first_listed)
    t["share_price"] = np.round(np.exp(t["log_price"]), 2)
    return t.drop(columns=["siccd", "log_mcap", "log_price", "frac_valid_m", "extreme_answer"])


def models_in(t: pd.DataFrame) -> list[str]:
    return [c[2:] for c in t.columns if c.startswith("q_")]


def custom_compare(t: pd.DataFrame, a: str, b: str, years: tuple[int, int], filters: dict[str, list[str]] | None = None,
                   block: int = 12, seed: int = 0) -> dict:
    """Model a minus model b on the chosen years and groups. Negative = a better."""
    mask = t["yyyymm"].between(years[0] * 100 + 1, years[1] * 100 + 12)
    for col, keep in (filters or {}).items():
        if keep:
            mask &= t[col].isin(keep)
    s = t[mask]
    if s.empty:
        return {"n_obs": 0}
    dm = monthly_means((s[f"q_{a}"] - s[f"q_{b}"]).to_numpy(), s["yyyymm"].to_numpy())
    iv = block_bootstrap(dm.to_numpy(), block=min(block, max(1, len(dm) // 4)), seed=seed)
    return {"n_obs": int(len(s)), "n_firms": int(s["permno"].nunique()), "n_months": int(len(dm)),
            "estimate": iv.estimate, "ci_lo": iv.lo, "ci_hi": iv.hi, "months_a_better": float((dm < 0).mean()),
            "monthly": dm, "block": iv.block}


def verdict(res: dict) -> str:
    if res.get("n_obs", 0) == 0:
        return "no examples"
    if res["n_months"] < 12 or res["n_obs"] < 500:
        return "too little data"
    if res["ci_hi"] < 0:
        return "better"
    if res["ci_lo"] > 0:
        return "worse"
    return "no detectable difference"


def names(msenames: pd.DataFrame) -> pd.DataFrame:
    """Each permno's most recent ticker and company name, for searching."""
    last = msenames.sort_values("nameendt").groupby("permno").tail(1)
    return last[["permno", "ticker", "comnam"]].reset_index(drop=True)


def search(nm: pd.DataFrame, text: str, have: set[int], limit: int = 30) -> pd.DataFrame:
    text = text.strip().upper()
    if not text:
        return nm.iloc[0:0]
    hit = nm[nm["permno"].isin(have) & (nm["ticker"].fillna("").str.upper().eq(text) | nm["comnam"].fillna("").str.contains(text, regex=False))]
    exact = hit["ticker"].fillna("").str.upper().eq(text)
    return pd.concat([hit[exact], hit[~exact]]).head(limit)


def stock_history(t: pd.DataFrame, permno: int) -> pd.DataFrame:
    return t[t["permno"] == permno].sort_values("yyyymm").reset_index(drop=True)


def to_vol(v) -> np.ndarray:
    """21-day variance to yearly volatility (for display): sqrt(12 x variance)."""
    return np.sqrt(12 * np.asarray(v, dtype="float64"))


def monthly_losses(t: pd.DataFrame) -> pd.DataFrame:
    """Average QLIKE per model per month (every stock in a month counts equally)."""
    cols = [f"q_{m}" for m in models_in(t)]
    return t.groupby("yyyymm")[cols].mean().rename(columns=lambda c: c[2:])


def filtered(t: pd.DataFrame, years: tuple[int, int], permno: int | None = None) -> pd.DataFrame:
    s = t[t["yyyymm"].between(years[0] * 100 + 1, years[1] * 100 + 12)]
    return s[s["permno"] == permno] if permno is not None else s


WITHIN = 1.25  # "within 25 percent": forecast between actual / 1.25 and actual x 1.25


def accuracy(s: pd.DataFrame, model: str) -> dict:
    """Plain-language accuracy of one model on the rows given, in yearly-volatility terms."""
    if s.empty:
        return {"n": 0}
    ratio = np.sqrt(np.maximum(s[f"f_{model}"].to_numpy(), 1e-8) / np.maximum(s["target_rv"].to_numpy(), 1e-8))
    off = np.abs(np.log(ratio))
    return {
        "n": int(len(s)),
        "typical_off": float(np.exp(np.median(off)) - 1),  # e.g. 0.25 = typically 25 percent too high or too low
        "within": float(np.mean(off <= np.log(WITHIN))),
        "too_low": float(np.mean(ratio < 1)),
        "qlike": float(monthly_means(s[f"q_{model}"].to_numpy(), s["yyyymm"].to_numpy()).mean()),
    }


def rank_skill(s: pd.DataFrame, model: str) -> float:
    """Average over months of the rank correlation between forecast and actual across stocks."""
    g = s[["yyyymm", f"f_{model}", "target_rv"]].copy()
    g = g[g.groupby("yyyymm")["target_rv"].transform("size") > 2]
    if g.empty:
        return float("nan")
    r = g.groupby("yyyymm")[[f"f_{model}", "target_rv"]].rank()
    r["yyyymm"] = g["yyyymm"]
    return float(r.groupby("yyyymm").apply(lambda x: x.iloc[:, 0].corr(x.iloc[:, 1]), include_groups=False).mean())


def market_series(s: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """All stocks: the typical (median) actual and forecast yearly volatility each month, plus market factors for hovering."""
    g = s.groupby("yyyymm")
    out = pd.DataFrame(index=g.size().index)  # keep the month labels (to_vol would drop them)
    out["actual"] = np.sqrt(12 * g["target_rv"].median())
    for m in models:
        out[m] = np.sqrt(12 * g[f"f_{m}"].median())
    out["stocks"] = g.size()
    out["market_vol"] = np.sqrt(12 * np.exp(g["mkt_log_rv_m"].first()))
    out["mood"] = g["market_regime"].first()
    return out


def stock_series(s: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """One stock: actual and forecast yearly volatility each month, plus the share price at the forecast."""
    s = s.sort_values("yyyymm").set_index("yyyymm")
    out = pd.DataFrame({"actual": to_vol(s["target_rv"])}, index=s.index)
    for m in models:
        out[m] = to_vol(s[f"f_{m}"])
    out["share_price"] = s["share_price"]
    return out


def scorecard(report: dict) -> pd.DataFrame:
    """Rank models on four things (1 = best) and pick the best overall by average rank:
    average error, error in a typical month, ranking skill, consistency (years clearly beating persistence)."""
    summ = report["summary"].set_index("model")
    typical = report["monthly"].groupby("model")["qlike"].median()
    yrs = report["by_year"]
    beat = yrs[(yrs["baseline"] == "persistence") & (yrs["ci_hi"] < 0)].groupby("model").size()
    t = pd.DataFrame({"average error": summ["qlike"], "typical month error": typical, "ranking skill": summ["rank_corr"]})
    t["years beating persistence"] = beat.reindex(t.index).fillna(0).astype(int)
    ranks = pd.DataFrame({
        "average error": t["average error"].rank(method="min"),
        "typical month error": t["typical month error"].rank(method="min"),
        "ranking skill": t["ranking skill"].rank(method="min", ascending=False),
        "years beating persistence": t["years beating persistence"].rank(method="min", ascending=False),
    })
    t["average rank"] = ranks.mean(axis=1)
    return t.sort_values(["average rank", "average error"])


def month_snapshot(t: pd.DataFrame, yyyymm: int, n: int = 1500, seed: int = 0) -> pd.DataFrame:
    s = t[t["yyyymm"] == yyyymm]
    return s.sample(n=min(n, len(s)), random_state=seed) if len(s) else s

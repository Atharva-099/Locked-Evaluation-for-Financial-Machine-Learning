"""Volatility task: at each month end, predict each stock's realised variance next month.

Units: every variance here is "21-trading-day variance": the average squared daily
return over the days that have a return, times 21. No demeaning, no annualising.

Row eligibility (all must hold):
- the stock is in the monthly universe at month t (common shares, NYSE/AMEX/Nasdaq);
- at least `min_valid_window` of the last 22 trading days up to the forecast have
  a daily return (so the persistence baseline exists);
- at least `min_valid_label` trading days in month t+1 have a daily return.
Rows failing a rule are kept in a separate exclusions table with the reason.

Timing: prediction_time = feature_available_time = last trading day of month t;
label_end_time = label_available_time = last trading day of month t+1 (the
answer is treated as known at that day's close).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from ..contract import validate_panel
from ..data import raw
from ..data.audit import load_audit
from ..data.clean import FAKE_JUMP_THRESHOLD, clean_daily, quote_reversal_flags
from ..manifest import code_hash, data_quick_fingerprint

TASK = "volatility"
PANEL_MODULES = ["config.py", "contract.py", "data/raw.py", "data/clean.py", "tasks/volatility.py"]

# name, window length in trading days, minimum days with a return
WINDOWS = (("d", 1, 1), ("w", 5, 3), ("m", 22, 15), ("q", 63, 40), ("y", 252, 150))

FEATURE_GROUPS = {
    "volatility_history": ["log_rv_d", "log_rv_w", "log_rv_m", "log_rv_q", "log_rv_y"],
    "return_history": ["ret_m", "ret_q"],
    "size_liquidity": ["log_mcap", "log_price", "log_turnover_m", "log_amihud_m", "frac_valid_m"],
    "market": ["mkt_log_rv_m"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
FEATURE_DESCRIPTIONS = {
    "log_rv_d": "log of 21 x (last day's return squared)",
    "log_rv_w": "log of 21-day variance from the last 5 trading days (needs 3 with a return)",
    "log_rv_m": "log of 21-day variance from the last 22 trading days (needs 15)",
    "log_rv_q": "log of 21-day variance from the last 63 trading days (needs 40)",
    "log_rv_y": "log of 21-day variance from the last 252 trading days (needs 150)",
    "ret_m": "compounded return over the last 22 trading days",
    "ret_q": "compounded return over the last 63 trading days (needs 40)",
    "log_mcap": "log of month-end market value (abs price x shares outstanding, thousands of dollars)",
    "log_price": "log of month-end absolute price",
    "log_turnover_m": "log of (mean daily volume / shares outstanding + 1e-6) over the last 22 trading days (needs 15)",
    "log_amihud_m": "log of mean daily |return| per million dollars traded over the last 22 trading days (needs 10)",
    "frac_valid_m": "share of the last 22 trading days with a return",
    "mkt_log_rv_m": "log of 21-day variance of the equal-weighted universe daily return over the last 22 trading days",
}


@dataclass(frozen=True)
class VolConfig:
    days_basis: int = 21
    min_valid_window: int = 15
    min_valid_label: int = 15
    min_turnover_days: int = 15
    min_amihud_days: int = 10
    variance_floor: float = 1e-6
    history_days: int = 252
    first_prediction_month: str = "1991-01"


def _cum(x: np.ndarray) -> np.ndarray:
    return np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])


def _window(c: np.ndarray, end: np.ndarray, w: int) -> np.ndarray:
    """Sum over the w rows ending at each index in `end` (inclusive), from a cumulative array. Shape (len(end), N)."""
    lo = np.maximum(end + 1 - w, 0)
    return c[end + 1] - c[lo]


def _pit_window(pair: tuple[np.ndarray, np.ndarray], end: np.ndarray, w: int) -> np.ndarray:
    """Window of w days ending at `end`: days before `end` from the cumulative "full" array, day `end` from "now"."""
    c_full, x_now = pair
    lo = np.maximum(end + 1 - w, 0)
    return c_full[end] - c_full[lo] + x_now[end]


def _pit_range(pair: tuple[np.ndarray, np.ndarray], start: np.ndarray, end: np.ndarray) -> np.ndarray:
    c_full, x_now = pair
    return c_full[end] - c_full[start] + x_now[end]


def _ret_terms(R: np.ndarray, P: np.ndarray, V: np.ndarray) -> dict[str, np.ndarray]:
    """Per-day pieces that are summed over windows: valid-day count, squared return, log return, Amihud."""
    valid = ~np.isnan(R)
    r = np.nan_to_num(R)
    am_ok = valid & ~np.isnan(V) & (V > 0) & ~np.isnan(P) & (P > 0)
    return {
        "n": valid.astype(float),
        "r2": np.where(valid, r * r, 0.0),
        "lr": np.where(valid, np.log1p(np.clip(r, -0.999999, None)), 0.0),
        "an": am_ok.astype(float),
        "am": np.where(am_ok, np.abs(r) / np.where(am_ok, P * V, 1.0) * 1e6, 0.0),
    }


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def build_chunk(
    daily: pd.DataFrame,
    universe: pd.DataFrame,
    mkt_ret: pd.Series,
    calendar: pd.DatetimeIndex,
    pred_months: list[pd.Period],
    cfg: VolConfig = VolConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rows for `pred_months`, using only the trading days in `calendar`. Returns (panel, exclusions, cleaning info).

    daily: permno, date, ret, prc, vol, shrout.
    universe: permno, month (Period), prc, shrout, exchcd, siccd, ticker for the prediction months.
    mkt_ret: equal-weighted universe daily return indexed by date.
    calendar must run from enough history before the first prediction month to the
    last trading day of the month after the last prediction month.
    """
    cal = pd.DatetimeIndex(calendar)
    pos = pd.Series(np.arange(len(cal)), index=cal)
    by_month = pos.groupby(cal.to_period("M"))
    last_idx, first_idx = by_month.max(), by_month.min()
    months = pd.PeriodIndex(pred_months, freq="M")
    missing = [str(m + 1) for m in months if (m + 1) not in last_idx.index] + [str(m) for m in months if m not in last_idx.index]
    if missing:
        raise ValueError(f"calendar does not cover months {missing}")

    univ = universe[universe["month"].isin(months)]
    permnos = np.sort(univ["permno"].unique())
    d = daily[daily["permno"].isin(permnos) & daily["date"].isin(cal)]
    wide = {c: d.pivot(index="date", columns="permno", values=c).reindex(index=cal, columns=permnos).to_numpy(dtype="float64")
            for c in ("ret", "prc", "vol", "shrout")}
    P, V, S = np.abs(wide["prc"]), wide["vol"], wide["shrout"]
    # Two versions of each return: "now" (same-day cleaning only, known on the day) and
    # "full" (also the reversal rule, known one trading day later). A window ending on day e
    # uses "full" for days before e and "now" for day e itself, so nothing uses next-day data.
    r_now = wide["ret"]
    flags = quote_reversal_flags(r_now, wide["prc"], V)
    r_full = np.where(flags, np.nan, r_now)
    own = slice(int(first_idx.loc[months[0]]), int(last_idx.loc[months[-1]]) + 1)  # days belonging to this chunk's months
    after_flag = np.vstack([np.zeros((1, flags.shape[1]), bool), flags[:-1]])
    with np.errstate(invalid="ignore"):
        leftover = after_flag & (np.abs(r_full) > FAKE_JUMP_THRESHOLD)
    info = {"reversal_jumps_removed": int(flags[own].sum()), "leftovers_after_reversal": int(leftover[own].sum())}
    pit = {k: (_cum(f), n) for (k, f), n in zip(_ret_terms(r_full, P, V).items(), _ret_terms(r_now, P, V).values())}

    tv_ok = ~np.isnan(V) & ~np.isnan(S) & (S > 0)
    c_tn = _cum(tv_ok.astype(float))
    c_tv = _cum(np.where(tv_ok, V / np.where(tv_ok, S * 1000.0, 1.0), 0.0))
    m = mkt_ret.reindex(cal).to_numpy(dtype="float64")[:, None]
    m_ok = ~np.isnan(m)
    c_mn, c_m2 = _cum(m_ok.astype(float)), _cum(np.where(m_ok, m * m, 0.0))

    end = last_idx.loc[months].to_numpy()
    feats: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for name, w, minv in WINDOWS:
        n = _pit_window(pit["n"], end, w)
        rv = cfg.days_basis * _safe_div(_pit_window(pit["r2"], end, w), n)
        rv[n < minv] = np.nan
        feats[f"rv_{name}"] = rv
        counts[name] = n
    n_m, n_q = counts["m"], counts["q"]
    ret_m = np.expm1(_pit_window(pit["lr"], end, 22))  # rows with too few window days are excluded below
    ret_q = np.expm1(_pit_window(pit["lr"], end, 63))
    ret_q[n_q < 40] = np.nan
    tn = _window(c_tn, end, 22)
    turnover = _safe_div(_window(c_tv, end, 22), tn)
    turnover[tn < cfg.min_turnover_days] = np.nan
    an = _pit_window(pit["an"], end, 22)
    amihud = _safe_div(_pit_window(pit["am"], end, 22), an)
    amihud[an < cfg.min_amihud_days] = np.nan
    mn = _window(c_mn, end, 22)
    mkt_rv = cfg.days_basis * _safe_div(_window(c_m2, end, 22), mn)
    mkt_rv[mn < cfg.min_valid_window] = np.nan

    lab_s = first_idx.loc[months + 1].to_numpy()
    lab_e = last_idx.loc[months + 1].to_numpy()
    n_lab = _pit_range(pit["n"], lab_s, lab_e)
    target_rv = cfg.days_basis * _safe_div(_pit_range(pit["r2"], lab_s, lab_e), n_lab)

    # Long format: one entry per (month, permno) where the permno is in the universe that month.
    mi = pd.Index(months).get_indexer(univ["month"])
    pj = np.searchsorted(permnos, univ["permno"].to_numpy())
    out = pd.DataFrame({
        "permno": univ["permno"].to_numpy(),
        "yyyymm": (univ["month"].dt.year * 100 + univ["month"].dt.month).to_numpy(),
        "ticker": univ["ticker"].to_numpy(),
        "exchcd": univ["exchcd"].to_numpy(),
        "siccd": univ["siccd"].to_numpy(),
        "prediction_time": cal[end[mi]],
        "label_end_time": cal[lab_e[mi]],
    })
    out["feature_available_time"] = out["prediction_time"]
    out["label_available_time"] = out["label_end_time"]
    out["row_id"] = out["permno"].astype("int64") * 1_000_000 + out["yyyymm"].astype("int64")
    out["n_window_days"] = n_m[mi, pj]
    out["n_label_days"] = n_lab[mi, pj]
    out["target_rv"] = target_rv[mi, pj]
    out["target_log_rv"] = np.log(np.maximum(out["target_rv"], cfg.variance_floor))
    for name, _, _ in WINDOWS:
        out[f"rv_{name}"] = feats[f"rv_{name}"][mi, pj]
        out[f"log_rv_{name}"] = np.log(np.maximum(out[f"rv_{name}"], cfg.variance_floor))
    out["ret_m"] = ret_m[mi, pj]
    out["ret_q"] = ret_q[mi, pj]
    prc = np.abs(univ["prc"].to_numpy(dtype="float64"))
    shr = univ["shrout"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_price"] = np.where(prc > 0, np.log(prc), np.nan)
        out["log_mcap"] = np.where((prc > 0) & (shr > 0), np.log(prc * shr), np.nan)
    out["log_turnover_m"] = np.log(turnover[mi, pj] + 1e-6)
    out["log_amihud_m"] = np.log(np.maximum(amihud[mi, pj], 1e-9))
    out["frac_valid_m"] = n_m[mi, pj] / 22.0
    out["mkt_log_rv_m"] = np.log(np.maximum(mkt_rv[mi, 0], cfg.variance_floor))

    window_short = out["n_window_days"] < cfg.min_valid_window
    label_short = ~window_short & (out["n_label_days"] < cfg.min_valid_label)
    reason = np.select([window_short, label_short], ["window_lt_min_days", "label_month_lt_min_days"], default="")
    excluded = out.loc[reason != "", ["row_id", "permno", "yyyymm", "n_window_days", "n_label_days"]].assign(reason=reason[reason != ""])
    panel = out.loc[reason == ""].reset_index(drop=True)
    return panel, excluded.reset_index(drop=True), info


def chunk_calendar(cal: pd.DatetimeIndex, months: list[pd.Period], history_days: int) -> pd.DatetimeIndex:
    """Trading days a chunk needs: `history_days` days up to the first month's end, through the last label month."""
    cal_months = cal.to_period("M")
    e0 = int(np.flatnonzero(cal_months == months[0])[-1])
    e1 = int(np.flatnonzero(cal_months == months[-1] + 1)[-1])
    return cal[max(0, e0 - (history_days - 1)): e1 + 1]


def market_daily_returns(daily: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    """Equal-weighted mean daily return over stocks in the monthly universe that month."""
    d = daily[["permno", "date", "ret"]].copy()
    d["month"] = d["date"].dt.to_period("M")
    d = d.merge(membership, on=["permno", "month"], how="inner")
    return d.groupby("date")["ret"].mean()


def load_clean_year(year: int, membership: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """One year of daily data with fake jumps removed, its market return, and cleaning counts."""
    d, counts = clean_daily(raw.load_daily_year(year))
    return d, market_daily_returns(d, membership), counts


def build_panel(cfg: VolConfig = VolConfig(), audit: dict | None = None, progress=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    audit = audit or load_audit()
    last_pred = pd.Period(audit["cutoffs"]["one_month_ahead_last_prediction_month"], freq="M")
    first_pred = pd.Period(cfg.first_prediction_month, freq="M")
    msf = raw.load_msf()
    msf["month"] = msf["date"].dt.to_period("M")
    membership = msf[["permno", "month"]]
    cal = raw.trading_calendar()
    usable, _ = raw.usable_daily_years()

    cache: dict[int, tuple[pd.DataFrame, pd.Series]] = {}
    cleaning: dict[str, dict] = {}

    def year_data(y: int) -> tuple[pd.DataFrame, pd.Series]:
        if y not in cache:
            d, mkt, counts = load_clean_year(y, membership)
            cache[y] = (d, mkt)
            cleaning[str(y)] = counts
        return cache[y]

    panels, exclusions = [], []
    reversal: dict[str, dict] = {}
    for year in range(first_pred.year, last_pred.year + 1):
        months = list(pd.period_range(max(first_pred, pd.Period(f"{year}-01", "M")), min(last_pred, pd.Period(f"{year}-12", "M")), freq="M"))
        if not months:
            continue
        cal_slice = chunk_calendar(cal, months, cfg.history_days)
        years = [y for y in range(cal_slice[0].year, cal_slice[-1].year + 1) if y in usable]
        for old in [y for y in cache if y < years[0]]:
            del cache[old]
        daily = pd.concat([year_data(y)[0] for y in years], ignore_index=True)
        mkt = pd.concat([year_data(y)[1] for y in years])
        p, x, info = build_chunk(daily, msf[msf["month"].isin(months)], mkt, cal_slice, months, cfg)
        panels.append(p)
        exclusions.append(x)
        reversal[str(year)] = info
        progress(f"{year}: {len(p):,} rows, {len(x):,} excluded")

    panel = pd.concat(panels, ignore_index=True)
    excl = pd.concat(exclusions, ignore_index=True)
    validate_panel(panel)

    # Did excluded stocks delist in month t or t+1? (code 100 = still active, not a delisting)
    dl = raw.load_msedelist()
    dl = dl[dl["dlstcd"] != 100]
    dl_keys = set(zip(dl["permno"].astype("int64"), (dl["dlstdt"].dt.year * 100 + dl["dlstdt"].dt.month).astype("int64")))
    t = pd.PeriodIndex([pd.Period(year=v // 100, month=v % 100, freq="M") for v in excl["yyyymm"]], freq="M")
    t1 = t + 1
    excl["delisted_in_t_or_t1"] = [
        (p, a.year * 100 + a.month) in dl_keys or (p, b.year * 100 + b.month) in dl_keys
        for p, a, b in zip(excl["permno"].astype("int64"), t, t1)
    ]

    summary = {
        "rows": int(len(panel)),
        "excluded": int(len(excl)),
        "excluded_by_reason": {str(k): int(v) for k, v in excl["reason"].value_counts().items()},
        "label_exclusions_that_delisted": int(excl.loc[excl["reason"] == "label_month_lt_min_days", "delisted_in_t_or_t1"].sum()),
        "first_prediction_month": str(first_pred),
        "last_prediction_month": str(last_pred),
        "rows_per_year": {str(k): int(v) for k, v in (panel["yyyymm"] // 100).value_counts().sort_index().items()},
        "cleaning": {
            "rules": [f"same day: |ret| > {FAKE_JUMP_THRESHOLD} on a quote-price day with zero or missing volume",
                      f"reversal: |ret| > {FAKE_JUMP_THRESHOLD} on a quote-price day with volume, undone the next trading day; "
                      "applied only to days before the day the information is used"],
            "zero_trade_jumps_removed_all_stocks": sum(c["zero_trade_jumps_removed"] for c in cleaning.values()),
            "leftovers_after_zero_trade_all_stocks": sum(c["leftovers_after_zero_trade"] for c in cleaning.values()),
            "reversal_jumps_removed_universe_stocks": sum(c["reversal_jumps_removed"] for c in reversal.values()),
            "leftovers_after_reversal_universe_stocks": sum(c["leftovers_after_reversal"] for c in reversal.values()),
            "same_day_by_year": dict(sorted(cleaning.items())),
            "reversal_by_prediction_year": reversal,
        },
    }
    meta = {
        "task": TASK,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "code_hash": code_hash(PANEL_MODULES),
        "data": data_quick_fingerprint(["msf", "msedelist"], daily=True),
        "audit_created_utc": audit["created_utc"],
        "features": FEATURES,
        "feature_groups": FEATURE_GROUPS,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "summary": summary,
    }
    return panel, excl, meta


def panel_paths():
    d = config.PANELS_DIR
    return d / f"{TASK}.parquet", d / f"{TASK}_exclusions.parquet", d / f"{TASK}_meta.json"


def save_panel(panel: pd.DataFrame, excl: pd.DataFrame, meta: dict) -> None:
    p, x, m = panel_paths()
    p.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(p, index=False)
    excl.to_parquet(x, index=False)
    m.write_text(json.dumps(meta, indent=2))


def is_stale(meta: dict, cfg: VolConfig = VolConfig()) -> list[str]:
    reasons = []
    if meta.get("code_hash") != code_hash(PANEL_MODULES):
        reasons.append("panel code changed")
    if meta.get("config") != asdict(cfg):
        reasons.append("config changed")
    if meta.get("data") != data_quick_fingerprint(["msf", "msedelist"], daily=True):
        reasons.append("input data files changed")
    return reasons


def load_or_build(cfg: VolConfig = VolConfig(), rebuild: bool = False, progress=print) -> tuple[pd.DataFrame, dict]:
    p, _, m = panel_paths()
    if not rebuild and p.exists() and m.exists():
        meta = json.loads(m.read_text())
        stale = is_stale(meta, cfg)
        if not stale:
            return pd.read_parquet(p), meta
        progress(f"rebuilding panel: {', '.join(stale)}")
    panel, excl, meta = build_panel(cfg, progress=progress)
    save_panel(panel, excl, meta)
    return panel, meta

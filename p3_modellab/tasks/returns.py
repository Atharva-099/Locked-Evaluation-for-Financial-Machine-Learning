"""Stage 8: predict each common stock's next-month total return."""
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
from ..manifest import code_hash, data_quick_fingerprint
from .crsp_monthly import FEATURE_DESCRIPTIONS, FEATURE_GROUPS, FEATURES, build_features, month_end_map

TASK = "returns"
PANEL_MODULES = ["config.py", "contract.py", "data/raw.py", "tasks/crsp_monthly.py", "tasks/returns.py"]


@dataclass(frozen=True)
class ReturnConfig:
    first_prediction_month: str = "1991-01"


def _delisting_returns(delist: pd.DataFrame) -> pd.DataFrame:
    d = delist[["permno", "dlstdt", "dlret", "dlstcd"]].copy()
    d = d[d["dlstcd"] != 100]
    d["month"] = pd.to_datetime(d["dlstdt"]).dt.to_period("M")
    if d.duplicated(["permno", "month"]).any():
        # A security can have administrative duplicates. Prefer the last dated
        # record and refuse conflicting observed returns within a month.
        conflicts = d.groupby(["permno", "month"])["dlret"].nunique(dropna=True)
        if (conflicts > 1).any():
            raise ValueError("conflicting delisting returns for the same permno-month")
        d = d.sort_values("dlstdt").drop_duplicates(["permno", "month"], keep="last")
    return d[["permno", "month", "dlret", "dlstcd"]]


def build_panel(
    cfg: ReturnConfig = ReturnConfig(),
    audit: dict | None = None,
    msf: pd.DataFrame | None = None,
    delist: pd.DataFrame | None = None,
    calendar: pd.DatetimeIndex | None = None,
    progress=print,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build one-month-ahead return labels without conditioning on survival.

    CRSP monthly RET is combined with DLRET as (1 + RET)(1 + DLRET) - 1.
    A row with no next-month RET is retained when an observed DLRET exists.
    """
    supplied = any(value is not None for value in (msf, delist, calendar))
    audit = audit or load_audit()
    last_pred = pd.Period(audit["cutoffs"]["one_month_ahead_last_prediction_month"], freq="M")
    msf = raw.load_msf() if msf is None else msf.copy()
    delist = raw.load_msedelist() if delist is None else delist.copy()
    calendar = raw.trading_calendar() if calendar is None else pd.DatetimeIndex(calendar)
    features, base_excluded = build_features(msf, cfg.first_prediction_month)
    progress(f"monthly features: {len(features):,} eligible rows")

    monthly = msf[["permno", "date", "ret"]].copy()
    monthly["month"] = pd.to_datetime(monthly["date"]).dt.to_period("M")
    monthly = monthly.drop_duplicates(["permno", "month"])
    monthly["prediction_month"] = monthly["month"] - 1
    monthly = monthly.rename(columns={"ret": "next_ret"})[["permno", "prediction_month", "next_ret"]]

    dl = _delisting_returns(delist)
    dl["prediction_month"] = dl["month"] - 1
    dl = dl.rename(columns={"dlret": "next_dlret", "dlstcd": "next_dlstcd"})[
        ["permno", "prediction_month", "next_dlret", "next_dlstcd"]
    ]
    out = features.merge(monthly, left_on=["permno", "month"], right_on=["permno", "prediction_month"], how="left")
    out = out.merge(dl, left_on=["permno", "month"], right_on=["permno", "prediction_month"], how="left", suffixes=("", "_dl"))
    out = out.drop(columns=[c for c in ("prediction_month", "prediction_month_dl") if c in out])

    r = pd.to_numeric(out["next_ret"], errors="coerce").to_numpy(dtype="float64")
    dr = pd.to_numeric(out["next_dlret"], errors="coerce").to_numpy(dtype="float64")
    has_r, has_dr = np.isfinite(r), np.isfinite(dr)
    target = np.full(len(out), np.nan)
    target[has_r & ~has_dr] = r[has_r & ~has_dr]
    target[~has_r & has_dr] = dr[~has_r & has_dr]
    target[has_r & has_dr] = (1.0 + r[has_r & has_dr]) * (1.0 + dr[has_r & has_dr]) - 1.0
    out["target_ret"] = target
    out["target_includes_dlret"] = has_dr

    ends = month_end_map(calendar)
    label_month = out["month"] + 1
    out["label_end_time"] = label_month.map(ends)
    out["label_available_time"] = out["label_end_time"]
    bad = (out["month"] > last_pred) | ~np.isfinite(out["target_ret"]) | out["label_end_time"].isna()
    reason = np.select(
        [out["month"] > last_pred, ~np.isfinite(out["target_ret"]), out["label_end_time"].isna()],
        ["after_last_complete_prediction_month", "missing_next_month_total_return", "missing_label_month_calendar"],
        default="",
    )
    label_excluded = out.loc[bad, ["row_id", "permno", "yyyymm"]].copy()
    label_excluded["reason"] = reason[bad]
    panel = out.loc[~bad].drop(columns=["month", "next_ret", "next_dlret"]).reset_index(drop=True)
    validate_panel(panel)

    exclusions = pd.concat([
        base_excluded.assign(yyyymm=pd.NA, row_id=pd.NA)[["row_id", "permno", "yyyymm", "reason"]],
        label_excluded,
    ], ignore_index=True)
    summary = {
        "rows": int(len(panel)),
        "excluded": int(len(exclusions)),
        "excluded_by_reason": {str(k): int(v) for k, v in exclusions["reason"].value_counts().items()},
        "labels_with_delisting_return": int(panel["target_includes_dlret"].sum()),
        "first_prediction_month": str(panel["yyyymm"].min()),
        "last_prediction_month": str(panel["yyyymm"].max()),
        "target_mean": float(panel["target_ret"].mean()),
        "target_median": float(panel["target_ret"].median()),
    }
    meta = {
        "task": TASK,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "code_hash": code_hash(PANEL_MODULES),
        "data": {} if supplied else data_quick_fingerprint(["msf", "msedelist"], daily=False),
        "audit_created_utc": audit.get("created_utc"),
        "features": FEATURES,
        "feature_groups": FEATURE_GROUPS,
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "label": "next-month CRSP RET combined with observed DLRET",
        "summary": summary,
    }
    return panel, exclusions, meta


def panel_paths():
    d = config.PANELS_DIR
    return d / f"{TASK}.parquet", d / f"{TASK}_exclusions.parquet", d / f"{TASK}_meta.json"


def save_panel(panel: pd.DataFrame, exclusions: pd.DataFrame, meta: dict) -> None:
    p, x, m = panel_paths()
    p.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(p, index=False)
    exclusions.to_parquet(x, index=False)
    m.write_text(json.dumps(meta, indent=2))


def is_stale(meta: dict, cfg: ReturnConfig = ReturnConfig()) -> list[str]:
    reasons = []
    if meta.get("code_hash") != code_hash(PANEL_MODULES):
        reasons.append("panel code changed")
    if meta.get("config") != asdict(cfg):
        reasons.append("config changed")
    expected = data_quick_fingerprint(["msf", "msedelist"], daily=False)
    if meta.get("data") != expected:
        reasons.append("input data files changed")
    return reasons


def load_or_build(cfg: ReturnConfig = ReturnConfig(), rebuild: bool = False, progress=print) -> tuple[pd.DataFrame, dict]:
    p, _, m = panel_paths()
    if not rebuild and p.exists() and m.exists():
        meta = json.loads(m.read_text())
        stale = is_stale(meta, cfg)
        if not stale:
            return pd.read_parquet(p), meta
        progress(f"rebuilding returns panel: {', '.join(stale)}")
    panel, exclusions, meta = build_panel(cfg, progress=progress)
    # Real-data builds always record their source fingerprint.
    meta["data"] = data_quick_fingerprint(["msf", "msedelist"], daily=False)
    save_panel(panel, exclusions, meta)
    return panel, meta

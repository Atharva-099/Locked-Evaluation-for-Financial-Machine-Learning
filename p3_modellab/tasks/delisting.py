"""Stage 9 foundation: twelve-month adverse-delisting labels and features."""
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

TASK = "delisting"
HORIZON_MONTHS = 12
CODE_SOURCE = "https://www.crsp.org/wp-content/uploads/DelistCode.html"

# Frozen after checking the CRSP delisting-code table. Mergers, exchange moves,
# fund conversions, and still-active code 100 are deliberately excluded.
ADVERSE_CODE_REASONS = {
    400: "liquidation", 450: "liquidation with final payment", 460: "declared worthless",
    470: "liquidation with no final value yet", 480: "liquidation with no distribution yet",
    490: "liquidation with no distribution", 500: "dropped, reason unavailable", 520: "moved to OTC",
    550: "market makers", 551: "shareholders", 552: "low price", 560: "insufficient capital",
    561: "insufficient float", 570: "company request", 573: "deregistration", 574: "bankruptcy",
    575: "offer rescinded", 580: "delinquent", 581: "failure to register",
    582: "equity requirements", 583: "denied exception", 584: "financial guidelines",
    585: "public interest", 587: "violation", 589: "unlisted", 591: "SEC required",
}
ADVERSE_CODES = frozenset(ADVERSE_CODE_REASONS)
PANEL_MODULES = ["config.py", "contract.py", "data/raw.py", "tasks/crsp_monthly.py", "tasks/delisting.py"]


@dataclass(frozen=True)
class DelistingConfig:
    first_prediction_month: str = "1991-01"
    horizon_months: int = HORIZON_MONTHS


def _next_adverse_events(features: pd.DataFrame, delist: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    events = delist[delist["dlstcd"].isin(ADVERSE_CODES)][["permno", "dlstdt", "dlstcd"]].copy()
    events["dlstdt"] = pd.to_datetime(events["dlstdt"])
    events = events.sort_values(["permno", "dlstdt", "dlstcd"])
    dates = np.full(len(features), np.datetime64("NaT"), dtype="datetime64[ns]")
    codes = np.full(len(features), np.nan)
    by_permno = {int(p): g for p, g in events.groupby("permno", sort=False)}
    for permno, idx in features.groupby("permno", sort=False).groups.items():
        group = by_permno.get(int(permno))
        if group is None:
            continue
        positions = np.asarray(idx, dtype="int64")
        forecast_dates = features.loc[positions, "prediction_time"].to_numpy(dtype="datetime64[ns]")
        event_dates = group["dlstdt"].to_numpy(dtype="datetime64[ns]")
        event_codes = group["dlstcd"].to_numpy(dtype="float64")
        found = np.searchsorted(event_dates, forecast_dates, side="right")
        valid = found < len(event_dates)
        dates[positions[valid]] = event_dates[found[valid]]
        codes[positions[valid]] = event_codes[found[valid]]
    return dates, codes


def build_panel(
    cfg: DelistingConfig = DelistingConfig(),
    audit: dict | None = None,
    msf: pd.DataFrame | None = None,
    delist: pd.DataFrame | None = None,
    calendar: pd.DatetimeIndex | None = None,
    progress=print,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if cfg.horizon_months != HORIZON_MONTHS:
        raise ValueError(f"this task is defined at a fixed {HORIZON_MONTHS}-month horizon")
    supplied = any(value is not None for value in (msf, delist, calendar))
    audit = audit or load_audit()
    last_pred = pd.Period(audit["cutoffs"]["twelve_month_ahead_last_prediction_month"], freq="M")
    msf = raw.load_msf() if msf is None else msf.copy()
    delist = raw.load_msedelist() if delist is None else delist.copy()
    calendar = raw.trading_calendar() if calendar is None else pd.DatetimeIndex(calendar)
    features, base_excluded = build_features(msf, cfg.first_prediction_month)
    progress(f"delisting features: {len(features):,} eligible rows")

    ends = month_end_map(calendar)
    label_month = features["month"] + cfg.horizon_months
    features["label_end_time"] = label_month.map(ends)
    features["label_available_time"] = features["label_end_time"]
    next_date, next_code = _next_adverse_events(features, delist)
    features["next_adverse_date"] = pd.to_datetime(next_date)
    features["next_adverse_code"] = next_code
    positive = features["next_adverse_date"].notna() & (features["next_adverse_date"] <= features["label_end_time"])
    features["target_adverse_delisting"] = positive.astype("int8")
    features.loc[~positive, ["next_adverse_date", "next_adverse_code"]] = [pd.NaT, np.nan]

    incomplete = (features["month"] > last_pred) | features["label_end_time"].isna()
    reason = np.where(features["month"] > last_pred, "incomplete_twelve_month_followup", "missing_horizon_calendar")
    label_excluded = features.loc[incomplete, ["row_id", "permno", "yyyymm"]].copy()
    label_excluded["reason"] = reason[incomplete]
    panel = features.loc[~incomplete].drop(columns=["month"]).reset_index(drop=True)
    validate_panel(panel)
    exclusions = pd.concat([
        base_excluded.assign(yyyymm=pd.NA, row_id=pd.NA)[["row_id", "permno", "yyyymm", "reason"]],
        label_excluded,
    ], ignore_index=True)

    events_by_code = panel.loc[panel["target_adverse_delisting"] == 1, "next_adverse_code"].value_counts().sort_index()
    summary = {
        "rows": int(len(panel)), "positive_rows": int(panel["target_adverse_delisting"].sum()),
        "positive_rate": float(panel["target_adverse_delisting"].mean()),
        "unique_positive_events": int(panel.loc[panel["target_adverse_delisting"] == 1,
                                                  ["permno", "next_adverse_date"]].drop_duplicates().shape[0]),
        "positive_rows_by_code": {str(int(k)): int(v) for k, v in events_by_code.items()},
        "last_prediction_month": str(last_pred),
        "excluded_by_reason": {str(k): int(v) for k, v in exclusions["reason"].value_counts().items()},
    }
    meta = {
        "task": TASK, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg), "code_hash": code_hash(PANEL_MODULES),
        "data": {} if supplied else data_quick_fingerprint(["msf", "msedelist"], daily=False),
        "audit_created_utc": audit.get("created_utc"), "features": FEATURES,
        "feature_groups": FEATURE_GROUPS, "feature_descriptions": FEATURE_DESCRIPTIONS,
        "label": "any frozen adverse CRSP delisting code in months t+1 through t+12",
        "adverse_codes": {str(k): v for k, v in ADVERSE_CODE_REASONS.items()},
        "code_source": CODE_SOURCE, "summary": summary,
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


def load_or_build(cfg: DelistingConfig = DelistingConfig(), rebuild: bool = False, progress=print) -> tuple[pd.DataFrame, dict]:
    p, _, m = panel_paths()
    if not rebuild and p.exists() and m.exists():
        meta = json.loads(m.read_text())
        expected = data_quick_fingerprint(["msf", "msedelist"], daily=False)
        if meta.get("code_hash") == code_hash(PANEL_MODULES) and meta.get("config") == asdict(cfg) and meta.get("data") == expected:
            return pd.read_parquet(p), meta
        progress("rebuilding delisting panel: code, config, or source fingerprint changed")
    panel, exclusions, meta = build_panel(cfg, progress=progress)
    save_panel(panel, exclusions, meta)
    return panel, meta

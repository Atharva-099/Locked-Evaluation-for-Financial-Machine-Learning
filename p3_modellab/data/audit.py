"""Measure what the saved WRDS files actually contain, before any task is built.

Output: results/audit/audit.json (machine readable, tasks read cutoffs from it)
and results/audit/audit.md (readable summary).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import raw

DATE_COLUMNS = {
    "msf": ["date"],
    "msenames": ["namedt", "nameendt"],
    "msedelist": ["dlstdt"],
    "ccm": ["linkdt", "linkenddt"],
    "fundq": ["datadate", "rdq"],
}
KEYS = {
    "msf": ["permno", "date"],
    "msenames": ["permno", "namedt"],
    "msedelist": ["permno", "dlstdt"],
    "ccm": ["gvkey", "permno", "linkdt"],
    "fundq": ["gvkey", "datadate"],
}
# Column names that would indicate Compustat revision history (point-in-time vintages).
VINTAGE_HINTS = ("effdate", "thrudate", "effective", "vintage", "snapshot", "pitdate")

MIN_VALID_DAYS_LABEL = 15


def _parse_dates(s: pd.Series) -> tuple[pd.Series, int]:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s, 0
    parsed = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    bad = int((parsed.isna() & s.notna()).sum())
    return parsed, bad


def _source_report(name: str, full_hash: bool) -> dict:
    path = raw.find_source(name)
    df = pd.read_parquet(path)
    rep = {"fingerprint": raw.fingerprint(path, full_hash=full_hash), "rows": len(df), "empty": len(df) == 0}
    rep["null_share"] = {c: round(float(df[c].isna().mean()), 6) for c in df.columns}
    rep["duplicate_keys"] = int(df.duplicated(KEYS[name]).sum())
    rep["dates"] = {}
    for c in DATE_COLUMNS[name]:
        parsed, bad = _parse_dates(df[c])
        rep["dates"][c] = {
            "min": None if parsed.isna().all() else str(parsed.min().date()),
            "max": None if parsed.isna().all() else str(parsed.max().date()),
            "unparseable": bad,
        }
    if name == "fundq":
        rep["has_vintage_fields"] = any(h in c.lower() for c in df.columns for h in VINTAGE_HINTS)
    return rep


def _daily_scan(full_hash: bool, universe: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DatetimeIndex]:
    """Per-year report plus valid-return counts per (permno, month)."""
    files = raw.daily_files()
    usable, empty = raw.usable_daily_years()
    per_year, counts, dates = {}, [], []
    for year, path in files.items():
        rep = {"fingerprint": raw.fingerprint(path, full_hash=full_hash)}
        if year in empty:
            rep["empty"] = True
            per_year[str(year)] = rep
            continue
        d = raw.load_daily_year(year)
        rep.update(
            empty=False,
            rows=len(d),
            min=str(d["date"].min().date()),
            max=str(d["date"].max().date()),
            duplicate_keys=int(d.duplicated(["permno", "date"]).sum()),
            null_share={c: round(float(d[c].isna().mean()), 6) for c in ("ret", "prc", "vol", "shrout")},
            permnos=int(d["permno"].nunique()),
        )
        d["month"] = d["date"].dt.to_period("M")
        g = d.groupby(["permno", "month"]).agg(n_rows=("date", "size"), n_valid=("ret", "count")).reset_index()
        in_univ = g.merge(universe, on=["permno", "month"], how="left", indicator=True)
        rep["permno_months_outside_universe"] = int((in_univ["_merge"] == "left_only").sum())
        counts.append(g)
        dates.append(d["date"].unique())
        per_year[str(year)] = rep
    cal = pd.DatetimeIndex(np.unique(np.concatenate(dates)))
    return per_year, pd.concat(counts, ignore_index=True), cal


def _names_coverage(msf: pd.DataFrame, names: pd.DataFrame) -> float:
    """Share of monthly rows with a name record whose date range covers the row's date."""
    m = msf[["permno", "date"]].reset_index().merge(names[["permno", "namedt", "nameendt"]], on="permno", how="left")
    ok = (m["date"] >= m["namedt"]) & (m["date"] <= m["nameendt"])
    covered = ok.groupby(m["index"]).any()
    return float(covered.mean())


def run_audit(full_hash: bool = True) -> dict:
    report: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_dir": str(config.DATA_DIR),
        "sources": {name: _source_report(name, full_hash) for name in config.SOURCE_PREFIXES},
    }

    msf = raw.load_msf()
    msf["month"] = msf["date"].dt.to_period("M")
    universe = msf[["permno", "month"]].drop_duplicates()

    per_year, counts, cal = _daily_scan(full_hash, universe)
    report["daily"] = per_year

    cal_month_ends = pd.Series(cal).groupby(cal.to_period("M")).max()
    msf_month_dates = msf.groupby("month")["date"].agg(["min", "max"])
    same_months = cal_month_ends.index.equals(msf_month_dates.index)
    month_end_match = bool(
        same_months
        and (msf_month_dates["min"] == msf_month_dates["max"]).all()
        and (msf_month_dates["max"].values == cal_month_ends.values).all()
    )
    report["calendar"] = {
        "trading_days": len(cal),
        "first": str(cal[0].date()),
        "last": str(cal[-1].date()),
        "days_per_year": {str(k): int(v) for k, v in pd.Series(cal.year).value_counts().sort_index().items()},
        "msf_dates_equal_daily_month_ends": month_end_match,
    }

    # A month counts as complete when the monthly file has it and the daily data reaches that month's last trading day.
    last_msf_month = msf["month"].max()
    daily_last = cal[-1]
    msf_last_date = msf.loc[msf["month"] == last_msf_month, "date"].iloc[0]
    delist_max = pd.Timestamp(report["sources"]["msedelist"]["dates"]["dlstdt"]["max"])
    complete = daily_last >= msf_last_date and delist_max >= msf_last_date
    last_complete = last_msf_month if complete else last_msf_month - 1
    report["coverage"] = {
        "msf_last_month": str(last_msf_month),
        "daily_last_date": str(daily_last.date()),
        "delist_last_date": str(delist_max.date()),
        "last_complete_month": str(last_complete),
        "empty_daily_files": [int(y) for y, r in per_year.items() if r.get("empty")],
    }
    report["cutoffs"] = {
        "one_month_ahead_last_prediction_month": str(last_complete - 1),
        "twelve_month_ahead_last_prediction_month": str(last_complete - 12),
    }

    names = raw.load_msenames()
    delist = raw.load_msedelist()
    in_window = delist[delist["dlstdt"].between(msf["date"].min(), msf["date"].max())]
    groups = (in_window["dlstcd"] // 100 * 100).value_counts().sort_index()
    counts_next = counts.copy()
    counts_next["month"] = counts_next["month"] - 1  # label month t+1 attached to prediction month t
    lab = universe.merge(counts_next, on=["permno", "month"], how="left").fillna({"n_valid": 0})
    lab = lab[lab["month"] <= last_complete - 1]
    report["joins"] = {
        "msf_rows_with_covering_name_record": round(_names_coverage(msf, names), 6),
        "msf_permno_months_with_daily_rows": round(float(universe.merge(counts[["permno", "month"]], on=["permno", "month"], how="left", indicator=True)["_merge"].eq("both").mean()), 6),
    }
    # CRSP code 100 means "still active at the end of the file", not a delisting.
    report["delisting_records_in_sample_window"] = {
        "still_active_code_100": int(groups.get(100, 0)),
        "actual_delistings": int(groups.drop(100, errors="ignore").sum()),
        "by_code_group": {f"{int(k)}s": int(v) for k, v in groups.items()},
    }
    report["eligibility_preview"] = {
        "note": "Universe permno-months up to the one-month-ahead cutoff, split by whether the next month has enough valid daily returns for a volatility label. Exact per-task counts come from the panel builders.",
        "permno_months": int(len(lab)),
        f"next_month_valid_days_ge_{MIN_VALID_DAYS_LABEL}": int((lab["n_valid"] >= MIN_VALID_DAYS_LABEL).sum()),
        f"next_month_valid_days_lt_{MIN_VALID_DAYS_LABEL}": int((lab["n_valid"] < MIN_VALID_DAYS_LABEL).sum()),
    }
    return report


def write_audit(report: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = Path(out_dir or config.AUDIT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "audit.json"
    jpath.write_text(json.dumps(report, indent=2))
    mpath = out_dir / "audit.md"
    mpath.write_text(render_markdown(report))
    return jpath, mpath


def render_markdown(r: dict) -> str:
    lines = [f"# Data audit ({r['created_utc']})", "", f"Data folder: `{r['data_dir']}`", "", "## Sources", "",
             "| Source | Rows | Duplicate keys | Dates (min to max) | Unparseable dates |", "|---|---|---|---|---|"]
    for name, s in r["sources"].items():
        dr = "; ".join(f"{c}: {v['min']} to {v['max']}" for c, v in s["dates"].items())
        bad = sum(v["unparseable"] for v in s["dates"].values())
        lines.append(f"| {name} | {s['rows']:,} | {s['duplicate_keys']:,} | {dr} | {bad:,} |")
    lines += ["", "## Daily files", "", "| Year | Rows | Min | Max | Duplicate keys | ret null | Permno-months outside universe |", "|---|---|---|---|---|---|---|"]
    for y, d in r["daily"].items():
        if d.get("empty"):
            lines.append(f"| {y} | 0 (EMPTY) | | | | | |")
        else:
            lines.append(f"| {y} | {d['rows']:,} | {d['min']} | {d['max']} | {d['duplicate_keys']} | {d['null_share']['ret']:.4f} | {d['permno_months_outside_universe']:,} |")
    c, cal = r["coverage"], r["calendar"]
    lines += ["", "## Coverage and cutoffs", "",
              f"- Trading days: {cal['trading_days']:,} ({cal['first']} to {cal['last']})",
              f"- Monthly dates equal daily month ends: {cal['msf_dates_equal_daily_month_ends']}",
              f"- Last complete month: **{c['last_complete_month']}**",
              f"- Empty daily files: {c['empty_daily_files']}",
              f"- Last prediction month, one-month-ahead tasks: **{r['cutoffs']['one_month_ahead_last_prediction_month']}**",
              f"- Last prediction month, twelve-month-ahead tasks: **{r['cutoffs']['twelve_month_ahead_last_prediction_month']}**",
              f"- Compustat quarterly has revision-history (vintage) fields: {r['sources']['fundq'].get('has_vintage_fields')}",
              "", "## Joins", ""]
    lines += [f"- {k}: {v}" for k, v in r["joins"].items()]
    lines += ["", "## Delisting records in sample window", ""] + [f"- {k}: {v}" for k, v in r["delisting_records_in_sample_window"].items()]
    lines += [f"- Compustat quarterly duplicate (gvkey, datadate) rows: {r['sources']['fundq']['duplicate_keys']:,}"]
    lines += ["", "## Eligibility preview", ""] + [f"- {k}: {v}" for k, v in r["eligibility_preview"].items()]
    return "\n".join(lines) + "\n"


def load_audit(path: Path | None = None) -> dict:
    path = Path(path or config.AUDIT_DIR / "audit.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run experiments/00_audit.py first.")
    return json.loads(path.read_text())

"""Score a saved run and write everything to <run>/report/.

Everything here is computed from the run folder (forecasts and saved models)
plus the saved table; nothing is retrained.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import importance, metrics, misses
from .data import industry, raw
from .manifest import code_hash
from .runner import say
from .tasks import volatility

PANEL_COLS = ["row_id", "ticker", "exchcd", "siccd", "log_mcap", "log_price", "frac_valid_m", "mkt_log_rv_m"]


def first_listed_dates() -> pd.Series:
    return raw.load_msenames().groupby("permno")["namedt"].min()


def build_report(run_dir: Path, panel: pd.DataFrame | None = None, first_listed: pd.Series | None = None,
                 seed: int = 0, importance_rows: int = 20000, progress=say) -> Path:
    run_dir = Path(run_dir)
    man = json.loads((run_dir / "manifest.json").read_text())
    pred = pd.read_parquet(run_dir / "predictions.parquet")
    panel = panel if panel is not None else pd.read_parquet(volatility.panel_paths()[0])
    first_listed = first_listed if first_listed is not None else first_listed_dates()

    df = pred.merge(panel[PANEL_COLS], on="row_id", how="left", validate="many_to_one")
    df = misses.add_slices(metrics.add_losses(df), first_listed)
    progress("scores")
    skill = metrics.rank_skill(df)
    summary = metrics.model_summary(df, skill)
    comp = metrics.compare(df, seed=seed)
    years = metrics.by_year(df, seed=seed)
    progress("slices")
    slices = misses.slice_table(df, seed=seed)
    worst = misses.worst_cases(df)
    calib = misses.calibration(df)
    monthly = metrics.monthly_losses(df)
    progress("importance")
    imp = importance.permutation_importance(run_dir, panel.set_index("row_id"), pred, volatility.FEATURES,
                                            volatility.FEATURE_GROUPS, n_rows=importance_rows, seed=seed)

    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    comp.to_json(out / "comparisons.json", orient="records", indent=2)
    summary.to_json(out / "model_summary.json", orient="records", indent=2)
    years.to_parquet(out / "by_year.parquet", index=False)
    slices.to_parquet(out / "slices.parquet", index=False)
    worst.to_parquet(out / "worst_cases.parquet", index=False)
    calib.to_parquet(out / "calibration.parquet", index=False)
    monthly.to_parquet(out / "monthly_losses.parquet", index=False)
    skill.to_parquet(out / "rank_skill.parquet", index=False)
    imp.to_parquet(out / "importance_by_period.parquet", index=False)
    importance.summarise(imp).to_parquet(out / "importance.parquet", index=False)
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": man["run_id"],
        "run_type": man["run_type"],
        "evidence": man["evidence"],
        "test_years_inspected": man["test_years"],
        "report_code_hash": code_hash(),
        "seed": seed,
        "primary_metric": "QLIKE (lower is better); differences are model minus baseline, so negative favours the model",
        "weighting": "every month counts equally; examples within a month count equally",
        "interval": f"95 percent moving-block bootstrap over months, block {metrics.PRIMARY_BLOCK} months (sensitivity {metrics.SENSITIVITY_BLOCKS})",
        "trimmed_view": f"RETROSPECTIVE: drops examples whose answer is above the {metrics.TRIM_QUANTILE:.1%} point of all test answers",
        "slice_minimums": {"examples": misses.MIN_OBS, "firms": misses.MIN_FIRMS, "months": misses.MIN_MONTHS, "months_for_year_slices": misses.MIN_MONTHS_YEAR_SLICE},
        "industry_mapping_verified": industry.VERIFIED_AGAINST_OFFICIAL_FILE,
        "industry_mapping_source": industry.OFFICIAL_SOURCE,
        "industry_mapping_sha256": industry.OFFICIAL_TEXT_SHA256,
        "importance_rows_per_period": importance_rows,
    }
    (out / "report_info.json").write_text(json.dumps(info, indent=2))
    return out

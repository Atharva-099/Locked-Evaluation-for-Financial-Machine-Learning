"""Export compact, aggregate-only report artifacts for a hosted dashboard."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config


REPORT_FILES = {
    "volatility": ("report_info.json", "model_summary.json", "comparisons.json",
                   "monthly_losses.parquet", "importance.parquet", "by_year.parquet",
                   "calibration.parquet", "importance_by_period.parquet",
                   "rank_skill.parquet", "slices.parquet"),
    "returns": ("report_info.json", "model_summary.json", "comparisons.json",
                "monthly_metrics.parquet", "monthly_portfolios.parquet"),
    "delisting": ("report_info.json", "model_summary.json", "comparisons.json", "pairwise_comparisons.json",
                  "monthly_metrics.parquet", "calibration.parquet", "budget_metrics.parquet"),
}
CORE_REPORT_FILES = ("report_info.json", "model_summary.json", "comparisons.json")
SWEEP_FILES = ("manifest.json", "trials.parquet", "monthly.parquet", "candidates.parquet")
SENSITIVE_COLUMNS = {"permno", "ticker", "cusip", "ncusip", "comnam", "row_id"}


def _assert_aggregate_parquet(path: Path) -> None:
    """Refuse to publish a parquet file containing security identifiers."""
    columns = set(pq.read_schema(path).names)
    unsafe = sorted(columns & SENSITIVE_COLUMNS)
    if unsafe:
        raise ValueError(f"refusing to publish {path.name}: security identifiers {unsafe}")


def _monthly_volatility_series(predictions: Path) -> pd.DataFrame:
    """Create a small market-level chart series without retaining stock identifiers.

    Forecast and realized variances are averaged across stocks within each
    model-month and converted to annualized volatility. Reading in batches keeps
    the export bounded even when the source forecast file has millions of rows.
    """
    parts: list[pd.DataFrame] = []
    parquet = pq.ParquetFile(predictions)
    for batch in parquet.iter_batches(
        batch_size=250_000,
        columns=["yyyymm", "model", "forecast", "target_rv"],
    ):
        frame = batch.to_pandas()
        grouped = frame.groupby(["yyyymm", "model"], as_index=False, sort=False).agg(
            forecast_sum=("forecast", "sum"),
            actual_sum=("target_rv", "sum"),
            stocks=("forecast", "count"),
        )
        parts.append(grouped)
    totals = pd.concat(parts, ignore_index=True).groupby(
        ["yyyymm", "model"], as_index=False, sort=True
    )[["forecast_sum", "actual_sum", "stocks"]].sum()
    totals["forecast"] = totals["forecast_sum"] / totals["stocks"]
    totals["actual"] = totals["actual_sum"] / totals["stocks"]

    forecasts = totals.pivot(index="yyyymm", columns="model", values="forecast")
    forecasts = np.sqrt(12 * forecasts.clip(lower=0))
    reference = totals.sort_values(["yyyymm", "model"]).drop_duplicates("yyyymm")
    series = forecasts.join(reference.set_index("yyyymm")[["actual", "stocks"]])
    series["actual"] = np.sqrt(12 * series["actual"].clip(lower=0))
    return series.reset_index().sort_values("yyyymm")


def export_run(run_dir: Path, output_root: Path | None = None) -> Path:
    """Copy only small report aggregates; forecasts and fitted models are excluded."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    report_dir = run_dir / "report"
    manifest = json.loads(manifest_path.read_text())
    task = manifest.get("task")
    if task not in REPORT_FILES:
        raise ValueError(f"unsupported dashboard task {task!r}")
    missing = [name for name in CORE_REPORT_FILES if not (report_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"report is missing dashboard artifacts: {missing}")

    destination = Path(output_root or config.DASHBOARD_DATA_DIR) / task / manifest["run_id"]
    target_report = destination / "report"
    target_report.mkdir(parents=True, exist_ok=True)
    # Keep the provenance and model specification, but never package row-level data.
    shutil.copy2(manifest_path, destination / "manifest.json")
    for name in REPORT_FILES[task]:
        source = report_dir / name
        if source.exists():
            if source.suffix == ".parquet":
                _assert_aggregate_parquet(source)
            shutil.copy2(source, target_report / name)
    if task == "volatility" and (run_dir / "predictions.parquet").exists():
        # This is the only derived deployment artifact. It contains one row per
        # month, no security identifiers, and supports the hosted 2D/3D chart.
        _monthly_volatility_series(run_dir / "predictions.parquet").to_parquet(
            target_report / "monthly_series.parquet", index=False
        )
    return destination


def export_sweep(sweep_dir: Path, output_root: Path | None = None) -> Path:
    """Copy one completed sensitivity sweep as aggregate-only dashboard data."""
    sweep_dir = Path(sweep_dir)
    manifest_path = sweep_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") not in {"settings", "greedy", "data", "noise"}:
        raise ValueError(f"unsupported sweep kind {manifest.get('kind')!r}")
    missing = [name for name in ("trials.parquet", "monthly.parquet") if not (sweep_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"sweep is missing dashboard artifacts: {missing}")

    destination = Path(output_root or config.DASHBOARD_DATA_DIR) / "sweeps" / sweep_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    for name in SWEEP_FILES:
        source = sweep_dir / name
        if not source.exists():
            continue
        if source.suffix == ".parquet":
            _assert_aggregate_parquet(source)
        shutil.copy2(source, destination / name)
    return destination

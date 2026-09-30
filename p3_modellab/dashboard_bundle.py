"""Export compact, aggregate-only report artifacts for a hosted dashboard."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import config


REPORT_FILES = {
    "volatility": ("report_info.json", "model_summary.json", "comparisons.json",
                   "monthly_losses.parquet", "importance.parquet"),
    "returns": ("report_info.json", "model_summary.json", "comparisons.json", "monthly_metrics.parquet"),
    "delisting": ("report_info.json", "model_summary.json", "comparisons.json", "pairwise_comparisons.json",
                  "monthly_metrics.parquet", "calibration.parquet", "budget_metrics.parquet"),
}


def export_run(run_dir: Path, output_root: Path | None = None) -> Path:
    """Copy only small report aggregates; forecasts and fitted models are excluded."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    report_dir = run_dir / "report"
    manifest = json.loads(manifest_path.read_text())
    task = manifest.get("task")
    if task not in REPORT_FILES:
        raise ValueError(f"unsupported dashboard task {task!r}")
    missing = [name for name in REPORT_FILES[task] if not (report_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"report is missing dashboard artifacts: {missing}")

    destination = Path(output_root or config.DASHBOARD_DATA_DIR) / task / manifest["run_id"]
    target_report = destination / "report"
    target_report.mkdir(parents=True, exist_ok=True)
    # Keep the provenance and model specification, but never package row-level data.
    shutil.copy2(manifest_path, destination / "manifest.json")
    for name in REPORT_FILES[task]:
        shutil.copy2(report_dir / name, target_report / name)
    return destination

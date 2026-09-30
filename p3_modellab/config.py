"""Paths and project-wide settings. Task-specific settings live in each task module."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("P3_WRDS_CACHE", Path.home() / ".wrdslab" / "cache" / "wrds")).expanduser()

RESULTS_DIR = Path(os.environ.get("P3_RESULTS", PROJECT_DIR / "results")).expanduser()
DASHBOARD_DATA_DIR = Path(os.environ.get("P3_DASHBOARD_DATA", PROJECT_DIR / "dashboard_data")).expanduser()
AUDIT_DIR = RESULTS_DIR / "audit"
PANELS_DIR = RESULTS_DIR / "panels"
RUNS_DIR = RESULTS_DIR / "runs"
LOCKS_DIR = RESULTS_DIR / "locks"

# Each monthly/reference source is one parquet file whose name starts with this
# prefix followed by a hash. Loaders refuse to guess if zero or several match.
SOURCE_PREFIXES = {
    "msf": "msf_1990_2025_",
    "msenames": "msenames_",
    "msedelist": "msedelist_",
    "ccm": "ccm_lnkhist_",
    "fundq": "fundq_1990_2025_",
}
DAILY_PATTERN = r"^dsf_(\d{4})\.parquet$"

N_JOBS = max(1, (os.cpu_count() or 2) // 2)

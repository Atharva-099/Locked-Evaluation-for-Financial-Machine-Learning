"""Fingerprints of code, data and environment, so any saved result can be traced and re-created."""
from __future__ import annotations

import hashlib
import platform
import sys
from importlib import metadata
from pathlib import Path

from . import config
from .data import raw

PACKAGE_DIR = Path(__file__).resolve().parent


def code_hash(modules: list[str] | None = None) -> str:
    """sha256 over the given package modules (relative paths), or over every .py file in the package."""
    if modules is None:
        files = sorted(p for p in PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    else:
        files = [PACKAGE_DIR / m for m in sorted(modules)]
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(PACKAGE_DIR)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def data_quick_fingerprint(sources: list[str], daily: bool) -> dict:
    """Name, size, modification time and row count of each input file. Cheap; detects replaced files."""
    out = {name: raw.fingerprint(raw.find_source(name), full_hash=False) for name in sources}
    if daily:
        out["daily"] = {str(y): raw.fingerprint(p, full_hash=False) for y, p in raw.daily_files().items()}
    return out


def environment() -> dict:
    pkgs = ["pandas", "numpy", "pyarrow", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost",
            "joblib", "plotly", "streamlit"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": versions, "n_jobs": config.N_JOBS}

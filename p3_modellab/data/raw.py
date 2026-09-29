"""Read the saved WRDS parquet files. Read only; never writes to the data folder.

Every loader returns dates as pandas datetime64[ns] and numeric columns as
float64/int64, because the saved files mix string dates (monthly files) with
timestamp dates (daily files).
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import config


class DataFileError(RuntimeError):
    pass


def find_source(name: str, data_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir or config.DATA_DIR)
    prefix = config.SOURCE_PREFIXES[name]
    matches = sorted(p for p in data_dir.glob(f"{prefix}*.parquet") if p.name.startswith(prefix))
    if len(matches) != 1:
        raise DataFileError(f"{name}: expected exactly one file starting with '{prefix}' in {data_dir}, found {len(matches)}: {[m.name for m in matches]}")
    return matches[0]


def daily_files(data_dir: Path | None = None) -> dict[int, Path]:
    """All dsf_YYYY.parquet files by year, including empty ones."""
    data_dir = Path(data_dir or config.DATA_DIR)
    out = {}
    for p in data_dir.iterdir():
        m = re.match(config.DAILY_PATTERN, p.name)
        if m:
            out[int(m.group(1))] = p
    if not out:
        raise DataFileError(f"no daily files matching {config.DAILY_PATTERN} in {data_dir}")
    return dict(sorted(out.items()))


def row_count(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def usable_daily_years(data_dir: Path | None = None) -> tuple[list[int], list[int]]:
    """(years with rows, years whose file is empty). Empty files are reported, never read as data."""
    files = daily_files(data_dir)
    usable = [y for y, p in files.items() if row_count(p) > 0]
    empty = [y for y, p in files.items() if row_count(p) == 0]
    return usable, empty


def _to_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.astype("datetime64[ns]")
    return pd.to_datetime(s, format="%Y-%m-%d", errors="raise")


def load_daily_year(year: int, columns: list[str] | None = None, data_dir: Path | None = None) -> pd.DataFrame:
    path = daily_files(data_dir).get(year)
    if path is None:
        raise DataFileError(f"no daily file for {year}")
    if row_count(path) == 0:
        raise DataFileError(f"daily file for {year} is empty; callers must skip it explicitly")
    df = pd.read_parquet(path, columns=columns)
    if "date" in df:
        df["date"] = _to_datetime(df["date"])
    if "permno" in df:
        df["permno"] = df["permno"].astype("int64")
    for c in ("ret", "prc", "vol", "shrout"):
        if c in df:
            df[c] = df[c].astype("float64")
    return df


def load_msf(data_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(find_source("msf", data_dir))
    df["date"] = _to_datetime(df["date"])
    return df


def load_msenames(data_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(find_source("msenames", data_dir))
    df["namedt"] = _to_datetime(df["namedt"])
    df["nameendt"] = _to_datetime(df["nameendt"])
    return df


def load_msedelist(data_dir: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(find_source("msedelist", data_dir))
    df["dlstdt"] = _to_datetime(df["dlstdt"])
    return df


@lru_cache(maxsize=4)
def trading_calendar(data_dir: Path | None = None) -> pd.DatetimeIndex:
    """Every date on which at least one stock has a daily record, across all non-empty daily files."""
    usable, _ = usable_daily_years(data_dir)
    dates = [load_daily_year(y, columns=["date"], data_dir=data_dir)["date"].to_numpy() for y in usable]
    return pd.DatetimeIndex(np.unique(np.concatenate(dates)))


def fingerprint(path: Path, full_hash: bool = True) -> dict:
    st = path.stat()
    out = {"file": path.name, "bytes": st.st_size, "mtime": st.st_mtime, "rows": row_count(path)}
    if full_hash:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    return out

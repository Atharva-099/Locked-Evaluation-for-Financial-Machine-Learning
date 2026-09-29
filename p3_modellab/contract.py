"""The timing contract: every example carries four timestamps, and nothing may be
learned from an answer that was not yet known.

- prediction_time: when the forecast is made.
- feature_available_time: when the inputs were known (must be <= prediction_time).
- label_end_time: when the predicted outcome finishes (must be > prediction_time).
- label_available_time: when the outcome is known (must be >= label_end_time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIME_COLS = ["prediction_time", "feature_available_time", "label_end_time", "label_available_time"]


class ContractError(ValueError):
    pass


def _examples(df: pd.DataFrame, mask: pd.Series, n: int = 3) -> str:
    cols = [c for c in ["row_id", "permno", *TIME_COLS] if c in df.columns]
    return df.loc[mask, cols].head(n).to_string()


def validate_panel(df: pd.DataFrame) -> None:
    """Raise ContractError if any row breaks the timing contract."""
    missing = [c for c in ["row_id", *TIME_COLS] if c not in df.columns]
    if missing:
        raise ContractError(f"panel is missing columns {missing}")
    if not df["row_id"].is_unique:
        raise ContractError("row_id is not unique")
    for c in TIME_COLS:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            raise ContractError(f"{c} must be a datetime column, got {df[c].dtype}")
        if df[c].isna().any():
            raise ContractError(f"{c} has {int(df[c].isna().sum())} missing values")
    checks = {
        "feature_available_time > prediction_time": df["feature_available_time"] > df["prediction_time"],
        "label_end_time <= prediction_time": df["label_end_time"] <= df["prediction_time"],
        "label_available_time < label_end_time": df["label_available_time"] < df["label_end_time"],
    }
    for name, bad in checks.items():
        if bad.any():
            raise ContractError(f"{int(bad.sum())} rows violate the contract ({name}), e.g.\n{_examples(df, bad)}")


def assert_labels_known(df: pd.DataFrame, cutoff: pd.Timestamp, role: str) -> None:
    """Rows used for fitting or tuning at `cutoff` must have their answer known by then."""
    bad = df["label_available_time"] > cutoff
    if bad.any():
        raise ContractError(f"{int(bad.sum())} {role} rows have labels not available until after {cutoff.date()}, e.g.\n{_examples(df, bad)}")


def assert_predictable(df: pd.DataFrame, fit_cutoff: pd.Timestamp) -> None:
    """A model fitted at `fit_cutoff` may only forecast at or after that time."""
    bad = df["prediction_time"] < fit_cutoff
    if bad.any():
        raise ContractError(f"{int(bad.sum())} test rows are forecast before the model's fit time {fit_cutoff.date()}, e.g.\n{_examples(df, bad)}")


def assert_disjoint(**id_sets: np.ndarray) -> None:
    names = list(id_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = np.intersect1d(id_sets[a], id_sets[b])
            if overlap.size:
                raise ContractError(f"{overlap.size} rows are in both {a} and {b}")

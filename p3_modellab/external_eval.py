"""Small, dependency-light evaluation for user-supplied targets and forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .losses import qlike


TASKS = ("volatility", "returns", "delisting")


def _rank_correlation(target: np.ndarray, forecast: np.ndarray) -> float:
    if len(target) < 3 or np.unique(target).size < 2 or np.unique(forecast).size < 2:
        return np.nan
    return float(spearmanr(forecast, target).statistic)


def _average_precision(target: np.ndarray, forecast: np.ndarray) -> float:
    positives = int(target.sum())
    if positives == 0:
        return np.nan
    order = np.argsort(-forecast, kind="stable")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def evaluate_predictions(data: pd.DataFrame, task: str, target_col: str,
                         forecast_col: str, time_col: str | None = None) -> dict[str, float | int]:
    """Score one uploaded forecast column without fitting or executing model code."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    required = [target_col, forecast_col] + ([time_col] if time_col else [])
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    target = pd.to_numeric(data[target_col], errors="coerce").to_numpy(dtype="float64")
    forecast = pd.to_numeric(data[forecast_col], errors="coerce").to_numpy(dtype="float64")
    if len(target) == 0:
        raise ValueError("the uploaded dataset has no rows")
    if np.any(~np.isfinite(target)) or np.any(~np.isfinite(forecast)):
        raise ValueError("target and prediction columns must contain only finite numbers")

    result: dict[str, float | int] = {"examples": int(len(target))}
    if task == "volatility":
        if np.any(target < 0) or np.any(forecast <= 0):
            raise ValueError("volatility targets must be non-negative and predictions must be positive")
        result.update({
            "qlike": float(np.mean(qlike(target, forecast))),
            "mse": float(np.mean((target - forecast) ** 2)),
            "rank_correlation": _rank_correlation(target, forecast),
        })
    elif task == "returns":
        result.update({
            "rank_correlation": _rank_correlation(target, forecast),
            "mse": float(np.mean((target - forecast) ** 2)),
            "mae": float(np.mean(np.abs(target - forecast))),
        })
        if time_col:
            monthly = pd.DataFrame({"time": data[time_col].astype(str), "target": target, "forecast": forecast})
            correlations = [
                _rank_correlation(group["target"].to_numpy(), group["forecast"].to_numpy())
                for _, group in monthly.groupby("time", sort=True)
            ]
            valid = np.asarray(correlations, dtype="float64")
            valid = valid[np.isfinite(valid)]
            result["mean_period_rank_ic"] = float(valid.mean()) if len(valid) else np.nan
            result["periods"] = int(len(valid))
    else:
        if not np.isin(target, [0.0, 1.0]).all():
            raise ValueError("delisting targets must contain only 0 and 1")
        if np.any((forecast < 0) | (forecast > 1)):
            raise ValueError("delisting predictions must be probabilities between 0 and 1")
        probability = np.clip(forecast, 1e-12, 1 - 1e-12)
        result.update({
            "log_loss": float(np.mean(-(target * np.log(probability) + (1 - target) * np.log(1 - probability)))),
            "brier_score": float(np.mean((target - forecast) ** 2)),
            "pr_auc": _average_precision(target, forecast),
            "event_rate": float(target.mean()),
        })
    return result

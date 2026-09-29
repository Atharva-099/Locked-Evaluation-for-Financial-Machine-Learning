"""Error bars for differences between models.

Recipe: compute the loss difference for every example, average it within each
month (every month counts equally), then resample whole blocks of consecutive
months (a moving-block bootstrap). Blocks keep neighbouring months together,
because they are related; treating them as independent would make error bars
falsely narrow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Interval:
    estimate: float
    lo: float
    hi: float
    p_value: float
    n_months: int
    block: int

    def as_dict(self) -> dict:
        return {"estimate": self.estimate, "ci_lo": self.lo, "ci_hi": self.hi, "p_value": self.p_value,
                "n_months": self.n_months, "block": self.block}


def monthly_means(values: np.ndarray, months: np.ndarray) -> pd.Series:
    """Average of `values` within each month, in month order."""
    return pd.Series(values).groupby(np.asarray(months)).mean().sort_index()


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block resample of positions 0..n-1."""
    block = max(1, min(block, n))
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=k)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def block_bootstrap(series: np.ndarray, stat=np.mean, block: int = 12, n_boot: int = 2000, seed: int = 0,
                    level: float = 0.95) -> Interval:
    """Estimate, percentile interval and two-sided p-value (for 'true value is zero') of `stat` over a monthly series."""
    x = np.asarray(series, dtype="float64")
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return Interval(np.nan, np.nan, np.nan, np.nan, 0, block)
    est = float(stat(x))
    rng = np.random.default_rng(seed)
    boots = np.array([stat(x[_block_indices(n, block, rng)]) for _ in range(n_boot)])
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    centred = boots - est
    p = float(min(1.0, 2 * min(np.mean(centred >= abs(est)), np.mean(centred <= -abs(est)))) if n > 1 else np.nan)
    return Interval(est, float(lo), float(hi), p, n, block)


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values (controls the chance of any false 'significant' among the family)."""
    p = np.asarray(p_values, dtype="float64")
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()

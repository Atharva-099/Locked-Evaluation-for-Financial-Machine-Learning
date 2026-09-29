"""Loss functions. Lower is better."""
from __future__ import annotations

import numpy as np


def qlike(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Per-row QLIKE for variance forecast h of realised variance y: y/h + log(h).

    This form (Patton 2011) stays defined when y = 0. Differences between two
    forecasts are identical to the normalised form y/h - log(y/h) - 1.
    """
    y = np.asarray(y, dtype="float64")
    h = np.asarray(h, dtype="float64")
    if np.any(h <= 0) or np.any(~np.isfinite(h)):
        raise ValueError("variance forecasts must be positive and finite")
    return y / h + np.log(h)


def squared_error(y: np.ndarray, h: np.ndarray) -> np.ndarray:
    return (np.asarray(y, dtype="float64") - np.asarray(h, dtype="float64")) ** 2

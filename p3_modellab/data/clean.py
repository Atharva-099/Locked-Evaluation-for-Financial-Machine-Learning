"""Data cleaning rules for daily returns (chosen 2026-09-26).

On a day with no closing trade CRSP reports the bid/ask midpoint as a negative
price, and a bogus quote can make the return look like a huge jump.

Rule 1, same-day (known on the day): a move over 50 percent on a quote-price day
with zero or missing volume is set to missing. Applied when data is loaded.

Rule 2, reversal (known only the next trading day): a move over 50 percent on a
quote-price day with some volume, undone the next trading day, i.e.
(1 + r[d]) x (1 + r[d+1]) between 0.67 and 1.5. Because it needs the next day,
the task builder applies it only to days before the day the information is used
(see tasks/volatility.py), so no forecast uses next-day information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FAKE_JUMP_THRESHOLD = 0.5
REVERSAL_LOW, REVERSAL_HIGH = 0.67, 1.5


def zero_trade_jump_mask(d: pd.DataFrame) -> pd.Series:
    return (d["ret"].abs() > FAKE_JUMP_THRESHOLD) & (d["prc"] < 0) & ((d["vol"] == 0) | d["vol"].isna())


def clean_daily(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply rule 1. Returns a copy and counts, including possible leftovers the rule does not
    remove: a move over 50 percent on the stock's next record after a removed day."""
    mask = zero_trade_jump_mask(d)
    out = d.copy()
    out.loc[mask, "ret"] = np.nan
    s = d.sort_values(["permno", "date"])
    prev_removed = mask.loc[s.index].groupby(s["permno"]).shift(1, fill_value=False).astype(bool)
    leftover = prev_removed & ~mask.loc[s.index] & (s["ret"].abs() > FAKE_JUMP_THRESHOLD)
    return out, {"zero_trade_jumps_removed": int(mask.sum()), "leftovers_after_zero_trade": int(leftover.sum())}


def quote_reversal_flags(ret: np.ndarray, prc: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Rule 2 on calendar-aligned arrays (trading days x stocks). Row d+1 is the next trading day."""
    nxt = np.vstack([ret[1:], np.full((1, ret.shape[1]), np.nan)])
    with np.errstate(invalid="ignore"):
        prod = (1 + ret) * (1 + nxt)
        return (np.abs(ret) > FAKE_JUMP_THRESHOLD) & (prc < 0) & (vol > 0) & (prod >= REVERSAL_LOW) & (prod <= REVERSAL_HIGH)

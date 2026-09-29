"""SIC code to Fama-French 12 industries. Used only to break results down, never as a model input.

STATUS: typed from the published Fama-French 12-industry definition; not yet
checked line by line against the official Siccodes12 file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VERIFIED_AGAINST_OFFICIAL_FILE = False

FF12_RANGES = {
    "NoDur": [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)],
    "Durbl": [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)],
    "Manuf": [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)],
    "Enrgy": [(1200, 1399), (2900, 2999)],
    "Chems": [(2800, 2829), (2840, 2899)],
    "BusEq": [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)],
    "Telcm": [(4800, 4899)],
    "Utils": [(4900, 4949)],
    "Shops": [(5000, 5999), (7200, 7299), (7600, 7699)],
    "Hlth": [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)],
    "Money": [(6000, 6999)],
}
OTHER = "Other"


def ff12(sic: pd.Series) -> pd.Series:
    """Industry name for each SIC code; codes in no range (including missing or 0) are 'Other'."""
    s = pd.to_numeric(sic, errors="coerce").to_numpy()
    out = np.full(len(s), OTHER, dtype=object)
    for name, ranges in FF12_RANGES.items():
        for lo, hi in ranges:
            out[(s >= lo) & (s <= hi)] = name
    return pd.Series(out, index=sic.index)

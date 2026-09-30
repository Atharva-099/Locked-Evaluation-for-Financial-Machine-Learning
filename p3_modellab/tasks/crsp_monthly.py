"""Point-in-time CRSP monthly features shared by returns and delisting tasks."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


FEATURE_GROUPS = {
    "return_history": ["ret_1m", "ret_3m", "ret_6m", "momentum_12_2", "volatility_12m"],
    "size_liquidity": ["log_mcap", "log_price", "log_turnover", "log_dollar_volume"],
    "market": ["market_ret_1m"],
    "listing": ["is_nyse", "is_amex", "is_nasdaq"],
}
FEATURES = [name for names in FEATURE_GROUPS.values() for name in names]
FEATURE_DESCRIPTIONS = {
    "ret_1m": "stock return in the prediction month",
    "ret_3m": "compounded stock return over prediction month and the prior 2 months",
    "ret_6m": "compounded stock return over prediction month and the prior 5 months",
    "momentum_12_2": "compounded return from months t-12 through t-2",
    "volatility_12m": "standard deviation of monthly returns from t-11 through t, requiring at least 6 observations",
    "log_mcap": "log of absolute month-end price times shares outstanding",
    "log_price": "log of absolute month-end price",
    "log_turnover": "log of monthly volume divided by shares outstanding",
    "log_dollar_volume": "log of absolute price times monthly volume",
    "market_ret_1m": "equal-weighted common-stock return in the prediction month",
    "is_nyse": "one when the security is listed on NYSE",
    "is_amex": "one when the security is listed on AMEX",
    "is_nasdaq": "one when the security is listed on Nasdaq",
}


def _month_number(month: pd.Series) -> np.ndarray:
    return (month.dt.year.to_numpy(dtype="int64") * 12 + month.dt.month.to_numpy(dtype="int64"))


def _lagged(frame: pd.DataFrame, column: str, lag: int) -> np.ndarray:
    grouped = frame.groupby("permno", sort=False)
    values = grouped[column].shift(lag).to_numpy(dtype="float64")
    prior_month = grouped["month_number"].shift(lag).to_numpy(dtype="float64")
    valid = frame["month_number"].to_numpy() - prior_month == lag
    return np.where(valid, values, np.nan)


def _compound(values: list[np.ndarray], minimum: int | None = None) -> np.ndarray:
    a = np.column_stack(values)
    needed = minimum if minimum is not None else a.shape[1]
    count = np.isfinite(a).sum(axis=1)
    out = np.prod(np.where(np.isfinite(a), 1.0 + a, 1.0), axis=1) - 1.0
    out[count < needed] = np.nan
    return out


def build_features(msf: pd.DataFrame, first_month: str = "1991-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features using only records available by each month end.

    Common shares on NYSE, AMEX, and Nasdaq form the prediction universe. The
    feature calculation retains earlier non-eligible rows as lag history, then
    filters the final prediction rows to the common-share universe.
    """
    required = {"permno", "date", "ret", "prc", "shrout", "vol", "shrcd", "exchcd", "siccd", "ticker"}
    missing = sorted(required - set(msf.columns))
    if missing:
        raise ValueError(f"monthly CRSP data is missing columns {missing}")

    d = msf[list(required)].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["permno"] = d["permno"].astype("int64")
    d["month"] = d["date"].dt.to_period("M")
    if d.duplicated(["permno", "month"]).any():
        n = int(d.duplicated(["permno", "month"], keep=False).sum())
        raise ValueError(f"monthly CRSP data has {n} duplicate permno-month rows")
    d = d.sort_values(["permno", "month"]).reset_index(drop=True)
    d["month_number"] = _month_number(d["month"])

    lags = {k: _lagged(d, "ret", k) for k in range(13)}
    d["ret_1m"] = lags[0]
    d["ret_3m"] = _compound([lags[k] for k in range(3)])
    d["ret_6m"] = _compound([lags[k] for k in range(6)])
    d["momentum_12_2"] = _compound([lags[k] for k in range(2, 13)])
    trailing = np.column_stack([lags[k] for k in range(12)])
    count = np.isfinite(trailing).sum(axis=1)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        vol = np.nanstd(trailing, axis=1, ddof=1)
    d["volatility_12m"] = np.where(count >= 6, vol, np.nan)

    price = np.abs(pd.to_numeric(d["prc"], errors="coerce").to_numpy(dtype="float64"))
    shares = pd.to_numeric(d["shrout"], errors="coerce").to_numpy(dtype="float64")
    volume = pd.to_numeric(d["vol"], errors="coerce").to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        d["log_price"] = np.where(price > 0, np.log(price), np.nan)
        d["log_mcap"] = np.where((price > 0) & (shares > 0), np.log(price * shares), np.nan)
        d["log_turnover"] = np.where((volume >= 0) & (shares > 0), np.log(volume / (shares * 1000.0) + 1e-8), np.nan)
        d["log_dollar_volume"] = np.where((price > 0) & (volume > 0), np.log(price * volume), np.nan)

    common = d["shrcd"].isin([10, 11]) & d["exchcd"].isin([1, 2, 3])
    market = d.loc[common].groupby("month")["ret"].mean()
    d["market_ret_1m"] = d["month"].map(market).astype("float64")
    d["is_nyse"] = (d["exchcd"] == 1).astype("float64")
    d["is_amex"] = (d["exchcd"] == 2).astype("float64")
    d["is_nasdaq"] = (d["exchcd"] == 3).astype("float64")

    eligible_month = d["month"] >= pd.Period(first_month, freq="M")
    current_return = np.isfinite(d["ret_1m"])
    reason = np.select(
        [~common, ~eligible_month, ~current_return],
        ["outside_common_share_universe", "before_first_prediction_month", "missing_prediction_month_return"],
        default="",
    )
    excluded = d.loc[reason != "", ["permno", "month", "date", "shrcd", "exchcd"]].copy()
    excluded["reason"] = reason[reason != ""]
    out = d.loc[reason == ""].copy()
    out["yyyymm"] = out["month"].dt.year * 100 + out["month"].dt.month
    out["prediction_time"] = out["date"]
    out["feature_available_time"] = out["prediction_time"]
    out["row_id"] = out["permno"] * 1_000_000 + out["yyyymm"].astype("int64")
    keep = [
        "row_id", "permno", "month", "yyyymm", "ticker", "exchcd", "siccd",
        "prediction_time", "feature_available_time", *FEATURES,
    ]
    return out[keep].reset_index(drop=True), excluded.reset_index(drop=True)


def month_end_map(calendar: pd.DatetimeIndex) -> dict[pd.Period, pd.Timestamp]:
    calendar = pd.DatetimeIndex(calendar)
    return pd.Series(calendar, index=calendar.to_period("M")).groupby(level=0).max().to_dict()

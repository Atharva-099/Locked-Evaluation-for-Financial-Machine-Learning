import numpy as np
import pandas as pd
import pytest

from p3_modellab.data.clean import clean_daily, quote_reversal_flags


def mwrx_like():
    """Rows modelled on real cases: MWRX (bogus quotes, June 2014) and TPST (real jump with heavy trading, Oct 2023)."""
    rows = [
        # permno, date, ret, prc, vol
        (1, "2014-06-03", -0.0055, 8.15, 5119.0),        # normal traded day: keep
        (1, "2014-06-04", 4.4957, -44.79, 0.0),          # bogus quote jump, no trades: remove
        (1, "2014-06-05", -0.8178, -8.16, 0.0),          # reversal, no trades: remove
        (1, "2014-06-18", 0.0061, 8.25, 3900.0),         # normal: keep
        (1, "2014-06-19", 4.4927, -45.315, 7000.0),      # quote price but volume > 0: keep (rule needs zero volume)
        (1, "2014-06-20", -0.8190, 8.20, 261750.0),      # traded day after a kept day: keep, not a leftover
        (1, "2014-06-26", 0.60, -8.16, np.nan),          # quote day, missing volume, over 50%: remove
        (1, "2014-06-27", -0.70, 8.20, 1000.0),          # traded reversal right after a removed day: keep, count as leftover
        (1, "2014-06-30", 0.50, -8.20, 0.0),             # exactly 50%: keep (rule is "over 50%")
        (2, "2023-10-11", 39.7253, 9.77, 172998224.0),   # TPST real jump with heavy trading: keep
        (2, "2023-10-12", -0.5926, 3.98, 47496620.0),    # real fall with trading: keep
        (3, "2014-06-04", -0.55, -2.0, 0.0),             # large fall on a no-trade quote day: remove
    ]
    return pd.DataFrame(rows, columns=["permno", "date", "ret", "prc", "vol"]).assign(date=lambda d: pd.to_datetime(d["date"]), shrout=1000.0)


def test_fake_jumps_removed_real_jumps_kept():
    d = mwrx_like()
    out, counts = clean_daily(d)
    removed = out["ret"].isna() & d["ret"].notna()
    assert removed.tolist() == [False, True, True, False, False, False, True, False, False, False, False, True]
    assert counts == {"zero_trade_jumps_removed": 4, "leftovers_after_zero_trade": 1}
    assert out.loc[9, "ret"] == pytest.approx(39.7253)
    pd.testing.assert_frame_equal(d, mwrx_like())  # the input is not modified


def test_leftover_count_does_not_cross_between_stocks():
    d = pd.DataFrame({"permno": [1, 2], "date": pd.to_datetime(["2014-06-04", "2014-06-05"]),
                      "ret": [4.0, 3.0], "prc": [-40.0, 5.0], "vol": [0.0, 100.0], "shrout": 1.0})
    _, counts = clean_daily(d)
    assert counts == {"zero_trade_jumps_removed": 1, "leftovers_after_zero_trade": 0}


def test_panel_builder_uses_cleaned_returns(monkeypatch):
    from p3_modellab.data import raw
    from p3_modellab.tasks import volatility
    monkeypatch.setattr(raw, "load_daily_year", lambda y: mwrx_like())
    membership = pd.DataFrame({"permno": [1, 3], "month": pd.Period("2014-06", "M")})
    d, mkt, counts = volatility.load_clean_year(2014, membership)
    assert d["ret"].isna().sum() == 4 and counts["zero_trade_jumps_removed"] == 4
    assert np.isnan(mkt.loc[pd.Timestamp("2014-06-04")])  # both universe stocks' returns that day were removed
    assert mkt.loc[pd.Timestamp("2014-06-03")] == pytest.approx(-0.0055)


def test_reversal_flags_need_quote_price_volume_and_next_day_undo():
    # columns: flagged | quote with volume but not undone | traded jump undone (not a quote) | undone next day but zero volume
    ret = np.array([[4.0, 4.0, 4.0, 4.0], [-0.8, 0.01, -0.8, -0.8]])
    prc = np.array([[-50.0, -50.0, 50.0, -50.0], [10.0, 10.0, 10.0, 10.0]])
    vol = np.array([[100.0, 100.0, 100.0, 0.0], [1000.0, 1000.0, 1000.0, 1000.0]])
    flags = quote_reversal_flags(ret, prc, vol)
    assert flags[0].tolist() == [True, False, False, False]
    assert not flags[1].any()  # last day has no next day

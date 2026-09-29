import numpy as np
import pandas as pd
import pytest

from p3_modellab.contract import validate_panel
from p3_modellab.tasks.volatility import VolConfig, build_chunk, chunk_calendar

CAL = pd.bdate_range("2000-01-03", "2000-03-31")
FEB = pd.Period("2000-02", "M")


def daily_rows(permno, dates, rets, prc=10.0, vol=1000.0, shrout=100.0):
    return pd.DataFrame({"permno": permno, "date": dates, "ret": rets, "prc": prc, "vol": vol, "shrout": shrout})


def fixture():
    jan_feb = CAL[CAL < "2000-03-01"]
    mar = CAL[CAL >= "2000-03-01"]
    alt = np.where(np.arange(len(CAL)) % 2 == 0, 0.02, -0.02)
    e_rets = np.full(len(CAL), 0.01)
    e_rets[-3:] = 0.03
    daily = pd.concat([
        daily_rows(10001, CAL, 0.01, prc=-10.0),           # A: +1% every day, price stored negative
        daily_rows(10002, CAL, alt),                         # B: +2% / -2% alternating
        daily_rows(10003, jan_feb.append(mar[:10]), 0.01),   # C: only 10 days in March
        daily_rows(10004, jan_feb[-10:].append(mar), 0.01),  # D: only 10 days of history
        daily_rows(10005, CAL, e_rets),                      # E: last 3 March days at +3%
        daily_rows(10006, CAL, 0.01),                        # F: not in the universe
    ], ignore_index=True)
    universe = pd.DataFrame({
        "permno": [10001, 10002, 10003, 10004, 10005], "month": FEB,
        "prc": [-10.0, 10.0, 10.0, 10.0, 10.0], "shrout": 100.0, "exchcd": 1, "siccd": 1234, "ticker": list("ABCDE"),
    })
    mkt = pd.Series(0.005, index=CAL)
    return daily, universe, mkt


def build():
    daily, universe, mkt = fixture()
    return build_chunk(daily, universe, mkt, CAL, [FEB])[:2]


def row(panel, permno):
    r = panel[panel["permno"] == permno]
    assert len(r) == 1
    return r.iloc[0]


def test_calendar_is_as_assumed_in_hand_calculations():
    months = CAL.to_period("M").value_counts().sort_index()
    assert months.tolist() == [21, 21, 23]


def test_constant_return_stock_hand_values():
    panel, _ = build()
    a = row(panel, 10001)
    assert a["target_rv"] == pytest.approx(21 * 0.01**2)
    assert a["n_label_days"] == 23 and a["n_window_days"] == 22
    for w in ("d", "w", "m", "q"):
        assert a[f"rv_{w}"] == pytest.approx(0.0021)
    assert np.isnan(a["rv_y"]) and np.isnan(a["log_rv_y"])  # only 42 days of history, needs 150
    assert a["ret_m"] == pytest.approx(1.01**22 - 1)
    assert a["ret_q"] == pytest.approx(1.01**42 - 1)  # 63-day window, only 42 days exist
    assert a["log_turnover_m"] == pytest.approx(np.log(1000 / (100 * 1000) + 1e-6))
    assert a["log_amihud_m"] == pytest.approx(np.log(0.01 / (10 * 1000) * 1e6))
    assert a["log_price"] == pytest.approx(np.log(10.0))  # negative CRSP price made positive
    assert a["log_mcap"] == pytest.approx(np.log(10.0 * 100.0))
    assert a["frac_valid_m"] == 1.0
    assert a["mkt_log_rv_m"] == pytest.approx(np.log(21 * 0.005**2))
    assert a["prediction_time"] == pd.Timestamp("2000-02-29")
    assert a["feature_available_time"] == pd.Timestamp("2000-02-29")
    assert a["label_end_time"] == pd.Timestamp("2000-03-31")
    assert a["label_available_time"] == pd.Timestamp("2000-03-31")
    assert a["row_id"] == 10001 * 1_000_000 + 200002


def test_alternating_stock_hand_values():
    b = row(build()[0], 10002)
    assert b["target_rv"] == pytest.approx(21 * 0.02**2)
    assert b["rv_m"] == pytest.approx(0.0084)
    assert b["ret_m"] == pytest.approx((1.02 * 0.98) ** 11 - 1)


def test_uneven_label_month_hand_value():
    e = row(build()[0], 10005)
    assert e["target_rv"] == pytest.approx(21 * (20 * 0.01**2 + 3 * 0.03**2) / 23)


def test_exclusions_and_universe():
    panel, excl = build()
    assert sorted(panel["permno"]) == [10001, 10002, 10005]
    assert 10006 not in set(excl["permno"])
    reasons = dict(zip(excl["permno"], excl["reason"]))
    assert reasons == {10003: "label_month_lt_min_days", 10004: "window_lt_min_days"}
    c = excl[excl["permno"] == 10003].iloc[0]
    d = excl[excl["permno"] == 10004].iloc[0]
    assert c["n_label_days"] == 10 and d["n_window_days"] == 10


def test_output_obeys_timing_contract():
    validate_panel(build()[0])


def test_future_data_does_not_change_inputs():
    """Changing March (the answer month) must change the answer but not a single input."""
    daily, universe, mkt = fixture()
    base, *_ = build_chunk(daily, universe, mkt, CAL, [FEB])
    changed = daily.copy()
    changed.loc[changed["date"] >= "2000-03-01", ["ret", "prc", "vol"]] = [0.05, 99.0, 5.0]
    mkt2 = mkt.copy()
    mkt2[mkt2.index >= "2000-03-01"] = 0.3
    alt, *_ = build_chunk(changed, universe, mkt2, CAL, [FEB])
    inputs = [c for c in base.columns if c.startswith(("log_", "rv_", "ret_", "frac_", "mkt_", "n_window"))]
    pd.testing.assert_frame_equal(base[inputs], alt[inputs])
    assert row(alt, 10001)["target_rv"] == pytest.approx(21 * 0.05**2)


def test_missing_label_month_in_calendar_refused():
    daily, universe, mkt = fixture()
    with pytest.raises(ValueError, match="does not cover"):
        build_chunk(daily, universe, mkt, CAL[CAL < "2000-03-01"], [FEB])


def test_day_count_thresholds_are_exact():
    """15 days is enough, 14 is not, for both the answer month and the recent-history window."""
    jan_feb = CAL[CAL < "2000-03-01"]
    mar = CAL[CAL >= "2000-03-01"]
    daily = pd.concat([
        daily_rows(1, jan_feb.append(mar[:15]), 0.01),
        daily_rows(2, jan_feb.append(mar[:14]), 0.01),
        daily_rows(3, jan_feb[-15:].append(mar), 0.01),
        daily_rows(4, jan_feb[-14:].append(mar), 0.01),
    ], ignore_index=True)
    universe = pd.DataFrame({"permno": [1, 2, 3, 4], "month": FEB, "prc": 10.0, "shrout": 100.0, "exchcd": 1, "siccd": 1, "ticker": "X"})
    panel, excl, _ = build_chunk(daily, universe, pd.Series(0.0, index=CAL), CAL, [FEB])
    assert sorted(panel["permno"]) == [1, 3]
    assert dict(zip(excl["permno"], excl["reason"])) == {2: "label_month_lt_min_days", 4: "window_lt_min_days"}


def test_market_return_uses_only_universe_stocks():
    from p3_modellab.tasks.volatility import market_daily_returns
    d = pd.DataFrame({"permno": [1, 2, 3], "date": pd.Timestamp("2000-01-03"), "ret": [0.01, 0.03, 0.50]})
    membership = pd.DataFrame({"permno": [1, 2], "month": pd.Period("2000-01", "M")})
    assert market_daily_returns(d, membership).iloc[0] == pytest.approx(0.02)


def reversal_world():
    """X: fake quote jump on the month-end day, undone the next day (in the answer month).
    Y: fake quote jump the day before month end, undone on month end.
    Z: fake quote jump in the middle of the answer month, undone three days later."""
    rows = []
    for permno in (20001, 20002, 20003):
        rows.append(daily_rows(permno, CAL, 0.01))
    daily = pd.concat(rows, ignore_index=True)

    def setday(permno, day, ret, prc, vol):
        i = daily.index[(daily["permno"] == permno) & (daily["date"] == pd.Timestamp(day))]
        daily.loc[i, ["ret", "prc", "vol"]] = [ret, prc, vol]

    setday(20001, "2000-02-29", 4.0, -50.0, 100.0)
    setday(20001, "2000-03-01", -0.8, 10.0, 1000.0)
    setday(20002, "2000-02-28", 4.0, -50.0, 100.0)
    setday(20002, "2000-02-29", -0.8, 10.0, 1000.0)
    setday(20003, "2000-03-10", 4.0, -50.0, 100.0)
    setday(20003, "2000-03-13", -0.8, 10.0, 1000.0)
    universe = pd.DataFrame({"permno": [20001, 20002, 20003], "month": FEB, "prc": 10.0, "shrout": 100.0, "exchcd": 1, "siccd": 1, "ticker": list("XYZ")})
    return daily, universe, pd.Series(0.0, index=CAL)


def test_reversal_rule_never_uses_next_day_information():
    daily, universe, mkt = reversal_world()
    panel, _, info = build_chunk(daily, universe, mkt, CAL, [FEB])
    x, y, z = (row(panel, p) for p in (20001, 20002, 20003))
    # X: on 29 Feb nobody knows the jump will be undone on 1 Mar, so it stays in the inputs.
    assert x["rv_d"] == pytest.approx(21 * 4.0**2)
    assert x["rv_m"] == pytest.approx(21 * (21 * 0.01**2 + 4.0**2) / 22)
    assert x["n_window_days"] == 22
    # Y: the 28 Feb jump is undone on 29 Feb, which is known at the forecast, so it is removed.
    assert y["n_window_days"] == 21
    assert y["rv_d"] == pytest.approx(21 * 0.8**2)
    assert y["rv_m"] == pytest.approx(21 * (20 * 0.01**2 + 0.8**2) / 21)
    # Z: inside the answer month the jump's reversal is known by month end, so it is removed.
    assert z["n_label_days"] == 22
    assert z["target_rv"] == pytest.approx(21 * (21 * 0.01**2 + 0.8**2) / 22)
    # X's answer month keeps its traded -80 percent day (not a quote day).
    assert x["target_rv"] == pytest.approx(21 * (22 * 0.01**2 + 0.8**2) / 23)
    # Counted within February (this chunk's month): X on 29 Feb and Y on 28 Feb; Y's 29 Feb is a leftover.
    assert info == {"reversal_jumps_removed": 2, "leftovers_after_reversal": 1}


def test_changing_the_day_after_the_forecast_does_not_change_inputs():
    daily, universe, mkt = reversal_world()
    base, *_ = build_chunk(daily, universe, mkt, CAL, [FEB])
    i = daily.index[(daily["permno"] == 20001) & (daily["date"] == pd.Timestamp("2000-03-01"))]
    daily.loc[i, "ret"] = 0.01  # the jump is no longer undone
    alt, *_ = build_chunk(daily, universe, mkt, CAL, [FEB])
    inputs = [c for c in base.columns if c.startswith(("log_", "rv_", "ret_", "frac_", "n_window"))]
    pd.testing.assert_frame_equal(base[inputs], alt[inputs])


def random_world(seed=0):
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range("2000-01-03", "2002-12-31")
    rows = []
    for p in range(1, 16):
        keep = rng.random(len(cal)) > 0.05
        n = keep.sum()
        rows.append(pd.DataFrame({
            "permno": p, "date": cal[keep], "ret": rng.normal(0, 0.01 * p, n),
            "prc": rng.uniform(1, 50, n) * rng.choice([-1, 1], n), "vol": rng.integers(0, 5000, n).astype(float),
            "shrout": rng.uniform(10, 1000, n),
        }))
    daily = pd.concat(rows, ignore_index=True)
    months = list(pd.period_range("2000-02", "2002-11", freq="M"))
    univ = pd.DataFrame([(p, m) for p in range(1, 16) for m in months if rng.random() > 0.1], columns=["permno", "month"])
    univ = univ.assign(prc=rng.uniform(1, 50, len(univ)), shrout=rng.uniform(10, 1000, len(univ)), exchcd=1, siccd=1000, ticker="X")
    mkt = daily.groupby("date")["ret"].mean()
    return cal, daily, univ, mkt, months


def test_year_by_year_equals_all_at_once():
    cal, daily, univ, mkt, months = random_world()
    whole, whole_x, _ = build_chunk(daily, univ, mkt, cal, months)
    parts, parts_x = [], []
    for year in (2000, 2001, 2002):
        ms = [m for m in months if m.year == year]
        p, x, _ = build_chunk(daily, univ, mkt, chunk_calendar(cal, ms, 252), ms)
        parts.append(p)
        parts_x.append(x)
    chunked = pd.concat(parts, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    whole = whole.sort_values("row_id").reset_index(drop=True)
    assert len(whole) > 400
    pd.testing.assert_frame_equal(whole, chunked, check_exact=False, rtol=1e-9, atol=1e-12)
    wx = whole_x.sort_values("row_id").reset_index(drop=True)
    cx = pd.concat(parts_x, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(wx, cx)


def test_chunk_calendar_gives_full_history():
    cal = pd.bdate_range("2000-01-03", "2002-12-31")
    sl = chunk_calendar(cal, [pd.Period("2001-06", "M")], 252)
    e0 = sl.get_loc(cal[cal.to_period("M") == pd.Period("2001-06", "M")][-1])
    assert e0 == 251
    assert sl[-1] == cal[cal.to_period("M") == pd.Period("2001-07", "M")][-1]

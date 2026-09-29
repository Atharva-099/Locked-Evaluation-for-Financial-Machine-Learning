import json

import numpy as np
import pandas as pd
import pytest

from p3_modellab import config, metrics, misses, plots, runner
from p3_modellab.data.industry import ff12
from p3_modellab.report import build_report
from test_models_runner import AUDIT, synthetic_panel


def test_industry_spot_checks():
    sic = pd.Series([2834, 3571, 6021, 4911, 1311, 3714, 3715, 7372, 5812, 4813, 2011, 9999, 0, np.nan])
    assert ff12(sic).tolist() == ["Hlth", "BusEq", "Money", "Utils", "Enrgy", "Durbl", "Manuf", "BusEq", "Shops", "Telcm",
                                  "NoDur", "Other", "Other", "Other"]


@pytest.mark.parametrize("args,expected", [
    ((5000, 100, 30, -0.02, -0.01, 24), "better"),
    ((5000, 100, 30, 0.01, 0.02, 24), "worse"),
    ((5000, 100, 30, -0.01, 0.01, 24), "no detectable difference"),
    ((1999, 100, 30, -0.02, -0.01, 24), "too little data"),
    ((5000, 49, 30, -0.02, -0.01, 24), "too little data"),
    ((5000, 100, 23, -0.02, -0.01, 24), "too little data"),
    ((5000, 100, 11, -0.02, -0.01, 10), "better"),
])
def test_verdicts(args, expected):
    assert misses.verdict(*args) == expected


def test_sign_convention_and_equal_month_weights_by_hand():
    """Month 1: model loss 1 vs baseline 3 on two stocks. Month 2: 5 vs 4 on one stock.
    Monthly differences: -2 and +1, so the average difference is -0.5 (each month counts equally)."""
    rows = []
    for model, losses in (("lightgbm", [1, 1, 5]), ("har", [3, 3, 4])):
        for rid, ym, l in zip([1, 2, 3], [202001, 202001, 202002], losses):
            rows.append({"row_id": rid, "yyyymm": ym, "model": model, "qlike": float(l), "se": 0.0, "log_se": 0.0, "target_rv": 1.0})
    comp = metrics.compare(pd.DataFrame(rows), pairs=[("lightgbm", "har")])
    assert comp.loc[0, "mean_estimate"] == pytest.approx(-0.5)
    assert comp.loc[0, "months_model_better"] == 0.5


def test_models_must_share_examples():
    df = pd.DataFrame({"row_id": [1, 2, 1], "model": ["a", "a", "b"], "qlike": 1.0, "yyyymm": 202001, "target_rv": 1.0})
    with pytest.raises(ValueError, match="identical examples"):
        metrics.wide(df, "qlike")


def test_slice_columns():
    df = pd.DataFrame({
        "row_id": range(10), "permno": range(10), "yyyymm": 202001,
        "prediction_time": pd.Timestamp("2020-01-31"), "siccd": 2834, "exchcd": [1, 2, 3, 1, 2, 3, 1, 2, 3, 4],
        "log_mcap": np.arange(10.0), "log_price": np.log([1, 4.99, 5, 19.99, 20, 100, 3, 7, 50, 2]),
        "frac_valid_m": [1, 0.95, 0.8, 1, 1, 1, 1, 1, 1, 1], "mkt_log_rv_m": np.log(0.2**2 / 12),
        "target_rv": np.arange(10.0),
    })
    first = pd.Series(pd.to_datetime(["2019-06-30", "2016-01-31", "2012-01-31", "2001-01-31"] + ["1990-01-01"] * 6), index=range(10))
    s = misses.add_slices(df, first)
    assert s["size"].value_counts().to_dict() == {"Q1": 2, "Q2": 2, "Q3": 2, "Q4": 2, "Q5": 2}
    assert s.loc[0, "size"] == "Q1" and s.loc[9, "size"] == "Q5"
    assert s["price"].tolist()[:6] == ["under $5", "under $5", "$5 to $20", "$5 to $20", "$20 and up", "$20 and up"]
    assert s["exchange"].tolist()[:4] == ["NYSE", "AMEX", "Nasdaq", "NYSE"] and s.loc[9, "exchange"] == "other"
    assert s["age"].tolist()[:4] == ["under 2y", "2 to 5y", "5 to 10y", "10 to 20y"]
    assert s["missing_days"].tolist()[:3] == ["none missing", "1 or 2 missing", "3 or more missing"]
    assert set(s["market_regime"]) == {"normal (15 to 25%)"}
    assert s.loc[9, "extreme_answer"] == "top 1% answer" and s.loc[0, "extreme_answer"] == "other"


def test_scramble_keeps_values_within_their_month():
    from p3_modellab.importance import _shuffle_within_month
    X = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0) * 10})
    months = np.repeat([1, 2, 3, 4], 25)
    s = _shuffle_within_month(X, ["a", "b"], months, np.random.default_rng(0))
    for m in (1, 2, 3, 4):
        assert sorted(s.loc[months == m, "a"]) == sorted(X.loc[months == m, "a"])
    assert (s["b"] == s["a"] * 10).all()  # a group is scrambled together
    assert not s["a"].equals(X["a"])


def test_month_level_inputs_are_swapped_between_months():
    from p3_modellab.importance import scramble
    months = np.repeat([1, 2, 3, 4, 5, 6], 10)
    X = pd.DataFrame({"mkt": np.repeat([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 10), "stock": np.arange(60.0)})
    s, how = scramble(X, ["mkt"], months, np.random.default_rng(1))
    assert how == "between months"
    assert (s.groupby(months)["mkt"].nunique() == 1).all()  # still one value per month
    assert sorted(s.groupby(months)["mkt"].first()) == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    assert not s["mkt"].equals(X["mkt"])
    assert scramble(X, ["stock"], months, np.random.default_rng(1))[1] == "within month"


def with_meta(panel):
    return panel.assign(ticker="T" + panel["permno"].astype(str), exchcd=1 + panel["permno"] % 3, siccd=2834)


FIRST = pd.Series(pd.Timestamp("1990-01-01"), index=range(1, 1001))


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "LOCKS_DIR", tmp_path / "locks")
    return tmp_path


def run_and_report(panel):
    out = runner.run(panel=panel, audit=AUDIT, test_years=[2004, 2005], progress=lambda *_: None)
    rep = build_report(out, panel=panel, first_listed=FIRST, importance_rows=3000, progress=lambda *_: None)
    return out, rep


def test_positive_signal_is_found_with_an_interval_excluding_zero(tmp_results):
    panel = with_meta(synthetic_panel(firms=150, nonlinear=True))
    _, rep = run_and_report(panel)
    comp = pd.read_json(rep / "comparisons.json")
    row = comp[(comp["model"] == "lightgbm") & (comp["baseline"] == "persistence")].iloc[0]
    assert row["mean_ci_hi"] < 0
    imp = pd.read_parquet(rep / "importance.parquet")
    lg = imp[(imp["model"] == "lightgbm") & (imp["kind"] == "feature")].set_index("input")["mean"]
    noise = ["log_amihud_m", "log_turnover_m", "log_price", "frac_valid_m", "ret_q"]
    assert lg["ret_m"] > 5 * lg[noise].abs().max()  # the planted input matters, pure-noise inputs barely do
    paths = plots.write_all(rep)
    assert all(p.exists() and p.stat().st_size > 1000 for p in paths)


def test_scrambled_answers_show_no_ranking_skill(tmp_results):
    """Answers shuffled among stocks within each month: no model can rank stocks, unless future data leaks in."""
    panel = with_meta(synthetic_panel(firms=150))
    rng = np.random.default_rng(0)
    perm = panel.groupby("yyyymm")["row_id"].transform(lambda s: rng.permutation(s.index.to_numpy()))
    panel[["target_rv", "target_log_rv"]] = panel.loc[perm.to_numpy(), ["target_rv", "target_log_rv"]].to_numpy()
    _, rep = run_and_report(panel)
    summ = pd.read_json(rep / "model_summary.json").set_index("model")
    for m in ("lightgbm", "ridge", "har"):
        ci = summ.loc[m, "rank_corr_ci"]
        assert ci["ci_lo"] < 0 < ci["ci_hi"], m
        assert abs(summ.loc[m, "rank_corr"]) < 0.05


def test_report_is_reproducible(tmp_results):
    panel = with_meta(synthetic_panel(firms=100, nonlinear=True))
    out, rep = run_and_report(panel)
    first = (pd.read_json(rep / "comparisons.json"), pd.read_parquet(rep / "importance_by_period.parquet"))
    build_report(out, panel=panel, first_listed=FIRST, importance_rows=3000, progress=lambda *_: None)
    pd.testing.assert_frame_equal(first[0], pd.read_json(rep / "comparisons.json"))
    pd.testing.assert_frame_equal(first[1], pd.read_parquet(rep / "importance_by_period.parquet"))
    info = json.loads((rep / "report_info.json").read_text())
    assert info["industry_mapping_verified"] is False

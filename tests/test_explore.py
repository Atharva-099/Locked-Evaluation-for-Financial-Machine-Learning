import numpy as np
import pandas as pd
import pytest

from p3_modellab import config, explore, metrics, runner
from test_models_runner import AUDIT, synthetic_panel
from test_report import FIRST, with_meta


@pytest.fixture(scope="module")
def run_and_table(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "RUNS_DIR", root)
    panel = with_meta(synthetic_panel(firms=120, nonlinear=True))
    out = runner.run(panel=panel, audit=AUDIT, test_years=[2004, 2005], progress=lambda *_: None)
    t = explore.explore_table(out, panel, FIRST)
    yield out, t
    mp.undo()


def test_table_has_one_row_per_example_with_every_model(run_and_table):
    out, t = run_and_table
    pred = pd.read_parquet(out / "predictions.parquet")
    assert len(t) == pred["row_id"].nunique() and t["row_id"].is_unique
    assert set(explore.models_in(t)) == set(pred["model"].unique())
    row = pred[(pred["model"] == "har")].iloc[0]
    assert t.set_index("row_id").loc[row["row_id"], "f_har"] == row["forecast"]


def test_custom_comparison_matches_the_official_one_without_filters(run_and_table):
    out, t = run_and_table
    pred = metrics.add_losses(pd.read_parquet(out / "predictions.parquet"))
    official = metrics.compare(pred, pairs=[("lightgbm", "har")]).iloc[0]
    mine = explore.custom_compare(t, "lightgbm", "har", (2004, 2005))
    assert mine["estimate"] == pytest.approx(official["mean_estimate"])
    assert mine["months_a_better"] == pytest.approx(official["months_model_better"])
    assert mine["n_obs"] == len(t)


def test_filters_and_years_narrow_the_data(run_and_table):
    _, t = run_and_table
    full = explore.custom_compare(t, "ridge", "har", (2004, 2005))
    one_year = explore.custom_compare(t, "ridge", "har", (2005, 2005))
    small = explore.custom_compare(t, "ridge", "har", (2004, 2005), {"size": ["Q1"]})
    assert one_year["n_months"] == 11 and one_year["n_obs"] < full["n_obs"]  # Dec 2005's answer falls in the sealed 2006
    assert 0 < small["n_obs"] < full["n_obs"]
    assert explore.custom_compare(t, "ridge", "har", (2004, 2005), {"industry": ["Nothing"]}) == {"n_obs": 0}
    assert explore.verdict({"n_obs": 0}) == "no examples"


def test_stock_history_and_monthly_losses(run_and_table):
    _, t = run_and_table
    h = explore.stock_history(t, 5)
    assert (h["permno"] == 5).all() and h["yyyymm"].is_monotonic_increasing
    ml = explore.monthly_losses(t)
    assert ml.index.nunique() == t["yyyymm"].nunique()
    m = ml.index[0]
    assert ml.loc[m, "har"] == pytest.approx(t[t["yyyymm"] == m]["q_har"].mean())
    assert explore.to_vol(np.array([1 / 12]))[0] == pytest.approx(1.0)


def test_names_and_search():
    ms = pd.DataFrame({"permno": [1, 1, 2, 3], "ticker": ["OLD", "AAPL", "MSFT", "AAPLX"],
                       "comnam": ["APPLE COMPUTER", "APPLE INC", "MICROSOFT CORP", "APPLE LOOKALIKE"],
                       "nameendt": pd.to_datetime(["2000-01-01", "2024-12-31", "2024-12-31", "2024-12-31"])})
    nm = explore.names(ms)
    assert dict(zip(nm["permno"], nm["ticker"])) == {1: "AAPL", 2: "MSFT", 3: "AAPLX"}
    hits = explore.search(nm, "aapl", have={1, 2, 3})
    assert hits["permno"].iloc[0] == 1  # exact ticker first
    assert set(explore.search(nm, "apple", have={1, 2})["permno"]) == {1}  # only scored stocks
    assert explore.search(nm, "  ", have={1}).empty


def test_accuracy_hand_examples():
    base = pd.DataFrame({"yyyymm": [202001, 202001, 202002, 202002], "target_rv": [0.01, 0.04, 0.01, 0.04]})
    exact = base.assign(f_m=base["target_rv"], q_m=1.0)
    a = explore.accuracy(exact, "m")
    assert a["typical_off"] == pytest.approx(0.0) and a["within"] == 1.0 and a["too_low"] == 0.0
    double_var = base.assign(f_m=base["target_rv"] * 2, q_m=1.0)  # twice the variance = 1.414 x the swing
    b = explore.accuracy(double_var, "m")
    assert b["typical_off"] == pytest.approx(np.sqrt(2) - 1) and b["within"] == 0.0
    half_var = base.assign(f_m=base["target_rv"] / 2, q_m=1.0)
    assert explore.accuracy(half_var, "m")["too_low"] == 1.0
    assert explore.accuracy(base.iloc[0:0].assign(f_m=[], q_m=[]), "m") == {"n": 0}


def test_market_and_stock_series(run_and_table):
    _, t = run_and_table
    ms = explore.market_series(t, ["har"])
    m = ms.index[0]
    rows = t[t["yyyymm"] == m]
    assert ms.loc[m, "actual"] == pytest.approx(np.sqrt(12 * rows["target_rv"].median()))
    assert ms.loc[m, "har"] == pytest.approx(np.sqrt(12 * rows["f_har"].median()))
    assert ms.loc[m, "stocks"] == len(rows)
    ss = explore.stock_series(explore.filtered(t, (2004, 2005), 7), ["ridge"])
    one = t[(t["permno"] == 7)].sort_values("yyyymm")
    assert list(ss.index) == list(one["yyyymm"])
    assert ss["share_price"].iloc[0] == one["share_price"].iloc[0]
    assert list(ms.index) == sorted(t["yyyymm"].unique())  # month labels kept


def test_scorecard_picks_the_best_average_rank():
    report = {
        "summary": pd.DataFrame({"model": ["a", "b", "c"], "qlike": [-3.0, -2.9, -2.0], "rank_corr": [0.70, 0.80, 0.60]}),
        "monthly": pd.DataFrame({"model": ["a", "a", "b", "b", "c", "c"], "yyyymm": [1, 2, 1, 2, 1, 2], "qlike": [-3, -3, -2.8, -2.8, -2, -2]}),
        "by_year": pd.DataFrame({"model": ["a", "b"], "baseline": ["persistence", "persistence"], "ci_hi": [-0.1, -0.1], "year": [2000, 2000]}),
    }
    sc = explore.scorecard(report)
    assert sc.index[0] == "a"  # ranks: a = 1,1,2,1 ; b = 2,2,1,1 ; c = 3,3,3,3
    assert sc.loc["a", "average rank"] == pytest.approx(1.25)
    assert sc.loc["c", "years beating persistence"] == 0

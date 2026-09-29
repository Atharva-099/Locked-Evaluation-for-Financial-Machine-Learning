import json

import numpy as np
import pandas as pd
import pytest

from p3_modellab import config, runner
from p3_modellab.losses import qlike
from p3_modellab.models import HAR, FLOOR, Persistence, RidgeModel, Winsorizer, _scale, make_model
from p3_modellab.tasks.volatility import FEATURES


def synthetic_panel(firms=60, start="2000-01", end="2008-11", seed=0, nonlinear=False):
    """Monthly panel with the real column names. The answer depends on the inputs in a known way."""
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    n = firms * len(months)
    permno = np.repeat(np.arange(1, firms + 1), len(months))
    month = np.tile(months, firms)
    level = np.repeat(rng.normal(-4, 0.7, firms), len(months))
    df = pd.DataFrame({"permno": permno, "yyyymm": [m.year * 100 + m.month for m in month]})
    for f in FEATURES:
        df[f] = rng.normal(0, 1, n)
    df["log_rv_m"] = level + rng.normal(0, 0.3, n)
    df["log_rv_w"] = df["log_rv_m"] + rng.normal(0, 0.3, n)
    df["log_rv_d"] = df["log_rv_w"] + rng.normal(0, 0.5, n)
    signal = 0.8 * (np.abs(df["ret_m"]) > 1) if nonlinear else 0.0
    df["target_log_rv"] = 0.7 * df["log_rv_m"] - 1.2 + signal + rng.normal(0, 0.3, n)
    for w in ("d", "w", "m"):
        df[f"rv_{w}"] = np.exp(df[f"log_rv_{w}"])
    df["target_rv"] = np.exp(df["target_log_rv"])
    df["row_id"] = df["permno"] * 1_000_000 + df["yyyymm"]
    pt = pd.PeriodIndex(month, freq="M").to_timestamp(how="end").normalize()
    le = (pd.PeriodIndex(month, freq="M") + 1).to_timestamp(how="end").normalize()
    df["prediction_time"] = df["feature_available_time"] = pt
    df["label_end_time"] = df["label_available_time"] = le
    return df


AUDIT = {"created_utc": "test", "cutoffs": {"one_month_ahead_last_prediction_month": "2008-11"}, "sources": {}}


def split(df, year=2004):
    tr = df[df["yyyymm"] < (year - 2) * 100]
    va = df[(df["yyyymm"] >= (year - 2) * 100) & (df["yyyymm"] < year * 100)]
    te = df[df["yyyymm"] // 100 == year]
    return tr, va, te


def test_persistence_is_last_22_day_variance_floored():
    df = pd.DataFrame({"rv_m": [0.01, 0.0, 2.5]})
    np.testing.assert_allclose(Persistence().predict(df), [0.01, FLOOR, 2.5])


def test_lognormal_multiplier_matches_formula_and_resists_one_extreme_row():
    rng = np.random.default_rng(1)
    g = rng.normal(-4, 1, 100_000)
    y_log = g + rng.normal(0.3, 0.5, 100_000)
    s = _scale(y_log, g)
    assert s == pytest.approx(np.exp(0.3 + 0.5**2 / 2), rel=0.01)
    # One MWRX-like row (answer 136,000 times the forecast): the new multiplier barely moves,
    # the old plain average of answer/forecast would have jumped.
    y_bad, g_bad = np.append(y_log, g[0] + np.log(136_000)), np.append(g, g[0])
    assert _scale(y_bad, g_bad) / s < 1.01
    plain = lambda yl, gg: np.mean(np.exp(yl - gg))
    assert plain(y_bad, g_bad) / plain(y_log, g) > 1.5


def test_har_recovers_exact_log_relationship_and_picks_log_form():
    rng = np.random.default_rng(2)
    n = 3000
    X = rng.normal(-4, 0.8, (n, 3))
    df = pd.DataFrame(np.exp(X), columns=["rv_d", "rv_w", "rv_m"])
    df["target_log_rv"] = 0.1 + 0.2 * X[:, 0] + 0.3 * X[:, 1] + 0.4 * X[:, 2]
    df["target_rv"] = np.exp(df["target_log_rv"])
    m = HAR()
    rec = m.fit(df.iloc[:2000], df.iloc[2000:])
    assert rec["chosen"]["form"] == "log"
    np.testing.assert_allclose(m.coef_, [0.1, 0.2, 0.3, 0.4], atol=1e-8)
    assert m.scale_ == pytest.approx(1.0)
    np.testing.assert_allclose(m.predict(df.iloc[2000:]), df["target_rv"].iloc[2000:], rtol=1e-8)
    assert {t["form"] for t in rec["trials"]} == {"log", "level"}


def test_har_missing_daily_falls_back_to_weekly_then_monthly():
    from p3_modellab.models import har_inputs
    df = pd.DataFrame({"rv_d": [np.nan, np.nan, 1.0], "rv_w": [2.0, np.nan, 2.0], "rv_m": [3.0, 3.0, 3.0]})
    np.testing.assert_array_equal(har_inputs(df), [[2, 2, 3], [3, 3, 3], [1, 2, 3]])


def test_winsorizer_bounds_come_from_training_rows_only():
    train = np.arange(1001, dtype=float).reshape(-1, 1)
    w = Winsorizer(0.001, 0.999).fit(train)
    out = w.transform(np.array([[-1e9], [500.0], [1e9], [np.nan]]))
    assert out[0, 0] == pytest.approx(np.quantile(train, 0.001))
    assert out[1, 0] == 500.0
    assert out[2, 0] == pytest.approx(np.quantile(train, 0.999))
    assert np.isnan(out[3, 0])


def test_ridge_records_every_trial_and_forecasts_positive():
    tr, va, te = split(synthetic_panel())
    m = RidgeModel(FEATURES)
    rec = m.fit(tr, va)
    assert [t["alpha"] for t in rec["trials"]] == list(m.alphas)
    h = m.predict(te)
    assert np.all(h > 0) and np.all(np.isfinite(h))
    assert np.corrcoef(np.log(h), te["target_log_rv"])[0, 1] > 0.5


def test_lightgbm_is_reproducible_with_the_same_seed():
    tr, va, te = split(synthetic_panel())
    a, b = make_model("lightgbm", FEATURES, seed=3, quick=True), make_model("lightgbm", FEATURES, seed=3, quick=True)
    a.fit(tr, va)
    b.fit(tr, va)
    np.testing.assert_array_equal(a.predict(te), b.predict(te))


def test_lightgbm_finds_a_planted_nonlinear_signal_that_persistence_misses():
    tr, va, te = split(synthetic_panel(firms=150, nonlinear=True))
    lgbm = make_model("lightgbm", FEATURES, quick=True)
    lgbm.fit(tr, va)
    y = te["target_rv"].to_numpy()
    assert qlike(y, lgbm.predict(te)).mean() < qlike(y, Persistence().predict(te)).mean()


@pytest.mark.parametrize("name,key", [("har", "form"), ("ridge", "alpha"), ("lightgbm", "num_leaves")])
def test_each_model_keeps_its_best_validation_setting(name, key):
    tr, va, _ = split(synthetic_panel(firms=100, nonlinear=True))
    m = make_model(name, FEATURES, quick=False)
    if name == "lightgbm":
        m.max_rounds, m.leaves = 200, (4, 63)
    rec = m.fit(tr, va)
    best = min(rec["trials"], key=lambda t: t["val_qlike"])
    assert rec["chosen"][key] == best[key]
    assert len({t["val_qlike"] for t in rec["trials"]}) > 1  # the choice was a real choice


def test_ridge_learns_from_training_rows_only():
    """The fitted model must equal one refitted on the training rows alone with the chosen penalty."""
    tr, va, te = split(synthetic_panel())
    m = RidgeModel(FEATURES)
    m.fit(tr, va)
    ref = m._pipe(m.alpha_).fit(tr[FEATURES], tr["target_log_rv"].to_numpy())
    np.testing.assert_allclose(m.pipe_.predict(te[FEATURES]), ref.predict(te[FEATURES]))


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "LOCKS_DIR", tmp_path / "locks")
    return tmp_path


def run_small(panel, **kw):
    kw.setdefault("models", ["persistence", "har", "ridge"])
    return runner.run(panel=panel, audit=AUDIT, progress=lambda *_: None, **kw)


def test_runner_end_to_end_obeys_timing_and_is_reproducible(tmp_results):
    panel = synthetic_panel()
    out = run_small(panel, test_years=[2004, 2005])
    man = json.loads((out / "manifest.json").read_text())
    pred = pd.read_parquet(out / "predictions.parquet")
    assert man["reserved_start"] == "2006-01-01" and man["evidence"] is True
    assert len(man["folds"]) == 2
    per_model = pred.groupby("model").size()
    assert set(per_model.index) == {"persistence", "har", "ridge"} and per_model.nunique() == 1
    assert per_model.iloc[0] == sum(f["n_test"] for f in man["folds"])
    for f in man["folds"]:
        rows = pred[pred["fold"] == f["fold"]]
        assert (rows["prediction_time"] >= pd.Timestamp(f["fit_cutoff"])).all()
        assert set(rows["yyyymm"] // 100) == set(f["test_years"])
        assert set(f["models"]) == {"persistence", "har", "ridge"}
    assert (pred["label_available_time"] < pd.Timestamp("2006-01-01")).all()
    assert len(list((out / "models").glob("*.joblib"))) == 3 * 2
    again = pd.read_parquet(run_small(panel, test_years=[2004, 2005]) / "predictions.parquet")
    pd.testing.assert_frame_equal(pred, again)


def test_saved_models_were_trained_on_their_fold_rows_only(tmp_results):
    import joblib
    from p3_modellab.splits import walk_forward
    panel = synthetic_panel()
    out = run_small(panel, models=["ridge"], test_years=[2004, 2005])
    folds = walk_forward(panel, [2004, 2005], pd.Timestamp("2006-01-01"), "exploratory", validation_months=runner.VALIDATION_MONTHS)
    by_id = panel.set_index("row_id")
    for f in folds:
        saved = joblib.load(out / "models" / f"ridge_fold{f.fold:02d}.joblib")
        train = by_id.loc[f.train_ids]
        ref = saved._pipe(saved.alpha_).fit(train[FEATURES], train["target_log_rv"].to_numpy())
        test = by_id.loc[f.test_ids]
        np.testing.assert_allclose(saved.pipe_.predict(test[FEATURES]), ref.predict(test[FEATURES]))


def test_code_fingerprint_is_taken_at_start_and_mid_run_edits_are_flagged(tmp_results, monkeypatch):
    calls = iter(["hash-at-start"] + ["hash-after-edit"] * 20)
    monkeypatch.setattr(runner, "code_hash", lambda *a, **k: next(calls))
    out = run_small(synthetic_panel(), models=["persistence"], test_years=[2004])
    man = json.loads((out / "manifest.json").read_text())
    assert man["code_hash"] == "hash-at-start"
    assert man["code_changed_during_run"] is True


def test_quick_runs_are_marked_as_not_evidence(tmp_results):
    out = run_small(synthetic_panel(firms=400), test_years=[2004, 2005], quick=True)
    man = json.loads((out / "manifest.json").read_text())
    assert man["quick"] is True and man["evidence"] is False and out.name.endswith("-quick")
    assert man["spec"]["firm_sample_percent"] == runner.QUICK_FIRM_SHARE
    pred = pd.read_parquet(out / "predictions.parquet")
    firms = pred["permno"].unique()
    assert runner.quick_sample(pd.Series(firms)).all()
    assert 40 <= len(firms) <= 120  # about 20 percent of 400


def test_locked_run_rules(tmp_results):
    panel = synthetic_panel()
    models = ["persistence", "har"]
    with pytest.raises(runner.RunError, match="lock file"):
        run_small(panel, models=models, run_type="locked")
    with pytest.raises(runner.RunError, match="quick"):
        run_small(panel, models=models, run_type="locked", quick=True)
    runner.create_lock(models)
    with pytest.raises(runner.RunError, match="already exists"):
        runner.create_lock(models)
    with pytest.raises(runner.RunError, match="does not match"):
        run_small(panel, models=["persistence"], run_type="locked")
    out = run_small(panel, models=models, run_type="locked")
    pred = pd.read_parquet(out / "predictions.parquet")
    assert (pred["prediction_time"] >= pd.Timestamp("2006-01-01")).all()
    with pytest.raises(runner.RunError, match="already ran"):
        run_small(panel, models=models, run_type="locked")
    again = run_small(panel, models=models, run_type="locked", allow_rerun=True)
    assert json.loads((again / "manifest.json").read_text())["lock"]["previous_locked_runs"] == [out.name]


def test_exploratory_cannot_test_sealed_years(tmp_results):
    from p3_modellab.splits import SplitError
    with pytest.raises(SplitError, match="reserved"):
        run_small(synthetic_panel(), test_years=[2005, 2006])


NEW = ["xgboost", "catboost", "extratrees", "mlp"]


def small(name, seed=0):
    from p3_modellab.models import make_model
    m = make_model(name, FEATURES, seed=seed, quick=True)
    return m


@pytest.mark.parametrize("name", NEW)
def test_new_models_forecast_positive_and_repeat_with_same_seed(name):
    tr, va, te = split(synthetic_panel(firms=80))
    a, b = small(name, seed=4), small(name, seed=4)
    rec = a.fit(tr, va)
    b.fit(tr, va)
    h = a.predict(te)
    assert np.all(h > 0) and np.all(np.isfinite(h)) and len(h) == len(te)
    np.testing.assert_allclose(h, b.predict(te), rtol=1e-12)
    assert rec["trials"] and all("val_qlike" in t for t in rec["trials"])


@pytest.mark.parametrize("name", ["xgboost", "catboost", "extratrees"])
def test_tree_models_find_a_planted_pattern_persistence_misses(name):
    tr, va, te = split(synthetic_panel(firms=150, nonlinear=True))
    m = small(name)
    m.fit(tr, va)
    y = te["target_rv"].to_numpy()
    assert qlike(y, m.predict(te)).mean() < qlike(y, Persistence().predict(te)).mean()


@pytest.mark.parametrize("name,key,values", [("xgboost", "max_depth", (2, 8)), ("extratrees", "min_samples_leaf", (5, 2000))])
def test_new_models_keep_their_best_setting(name, key, values):
    from p3_modellab.models import ExtraTreesModel, XGBoostModel
    tr, va, _ = split(synthetic_panel(firms=100, nonlinear=True))
    m = XGBoostModel(FEATURES, depths=values, max_rounds=100, patience=10) if name == "xgboost" else ExtraTreesModel(FEATURES, leaf_sizes=values, n_trees=20)
    rec = m.fit(tr, va)
    best = min(rec["trials"], key=lambda t: t["val_qlike"])
    assert rec["chosen"][key] == best[key] and len({t["val_qlike"] for t in rec["trials"]}) == 2


def test_neural_net_stops_early_and_row_caps_hold():
    from p3_modellab.models import ExtraTreesModel, MLPModel
    tr, va, te = split(synthetic_panel(firms=100))
    m = MLPModel(FEATURES, max_epochs=40, patience=2, max_train_rows=500)
    t = m.fit(tr, va)["trials"][0]
    assert t["epochs_run"] <= 40 and 1 <= t["best_epoch"] <= t["epochs_run"] and t["train_rows_used"] == 500
    assert t["val_mse_by_epoch"][t["best_epoch"] - 1] == min(t["val_mse_by_epoch"])
    et = ExtraTreesModel(FEATURES, leaf_sizes=(50,), n_trees=10, max_train_rows=700).fit(tr, va)
    assert et["chosen"]["train_rows_used"] == 700


def test_runner_handles_all_eight_models(tmp_results):
    from p3_modellab.models import MODEL_NAMES
    out = runner.run(panel=synthetic_panel(firms=60), audit=AUDIT, test_years=[2005], quick=False, models=list(MODEL_NAMES),
                     progress=lambda *_: None)
    pred = pd.read_parquet(out / "predictions.parquet")
    assert set(pred["model"]) == set(MODEL_NAMES) and pred.groupby("model").size().nunique() == 1
    assert (pred["forecast"] > 0).all()


def test_extratrees_predicts_on_one_core_so_forecasts_reproduce_exactly():
    from p3_modellab.models import ExtraTreesModel
    tr, va, te = split(synthetic_panel(firms=80))
    m = ExtraTreesModel(FEATURES, leaf_sizes=(50,), n_trees=30, n_jobs=3)
    m.fit(tr, va)
    assert m.model_.n_jobs == 1
    first = m.predict(te)
    assert all(np.array_equal(first, m.predict(te)) for _ in range(5))

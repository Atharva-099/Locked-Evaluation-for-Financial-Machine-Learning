import json

import numpy as np
import pandas as pd
import pytest

from p3_modellab import delisting_runner, return_runner
from p3_modellab.delisting_models import stratified_training_sample
from p3_modellab.delisting_report import build_report as build_delisting_report
from p3_modellab.return_report import build_report
from p3_modellab.tasks import delisting, returns
from p3_modellab.tasks.crsp_monthly import build_features


def monthly(rows):
    out = pd.DataFrame(rows, columns=["permno", "date", "ret", "prc", "shrout", "vol", "shrcd", "exchcd"])
    out["date"] = pd.to_datetime(out["date"])
    out["siccd"] = 1000
    out["ticker"] = "T" + out["permno"].astype(str)
    return out


def regular_msf(permnos=(1, 2, 3), start="1999-01", end="2001-12"):
    rows = []
    for permno in permnos:
        for i, month in enumerate(pd.period_range(start, end, freq="M")):
            rows.append((permno, month.to_timestamp(how="end").normalize(), 0.01 * permno + i / 10_000,
                         10.0 + permno, 1000.0, 100_000.0, 10, 1 + (permno % 3)))
    return monthly(rows)


def audit(one="2000-11", twelve="2000-12"):
    return {"created_utc": "test", "cutoffs": {"one_month_ahead_last_prediction_month": one,
                                                   "twelve_month_ahead_last_prediction_month": twelve}, "sources": {}}


def calendar(start="1999-01", end="2002-12"):
    return pd.date_range(pd.Period(start).start_time, pd.Period(end).end_time, freq="B")


def test_monthly_features_require_consecutive_months_for_compounding():
    data = regular_msf(permnos=(1,), start="1999-01", end="2000-03")
    data = data[data["date"].dt.to_period("M") != pd.Period("2000-01")]
    features, _ = build_features(data, first_month="1999-01")
    feb = features[features["yyyymm"] == 200002].iloc[0]
    assert np.isnan(feb["ret_3m"])
    assert feb["ret_1m"] == data.loc[data["date"].dt.to_period("M") == pd.Period("2000-02"), "ret"].iloc[0]


def test_return_label_combines_ret_and_dlret_and_keeps_delisting_only_row():
    data = regular_msf(permnos=(1,), start="1999-01", end="2000-03")
    jan = data["date"].dt.to_period("M") == pd.Period("2000-01")
    feb = data["date"].dt.to_period("M") == pd.Period("2000-02")
    data.loc[jan, "ret"] = 0.02
    data.loc[feb, "ret"] = 0.10
    dl = pd.DataFrame({"permno": [1], "dlstdt": pd.to_datetime(["2000-02-15"]), "dlret": [-0.5], "dlstcd": [574]})
    panel, _, _ = returns.build_panel(returns.ReturnConfig("1999-01"), audit("2000-02"), data, dl, calendar("1999-01", "2000-03"),
                                      progress=lambda *_: None)
    row = panel[panel["yyyymm"] == 200001].iloc[0]
    assert row["target_ret"] == np.float64((1.1 * 0.5) - 1.0)
    assert bool(row["target_includes_dlret"])
    assert row["ret_1m"] == 0.02

    no_feb = data[~feb]
    panel, _, _ = returns.build_panel(returns.ReturnConfig("1999-01"), audit("2000-01"), no_feb, dl, calendar("1999-01", "2000-03"),
                                      progress=lambda *_: None)
    row = panel[panel["yyyymm"] == 200001].iloc[0]
    assert row["target_ret"] == -0.5


def test_future_return_changes_label_but_not_prediction_features():
    data = regular_msf(permnos=(1,), start="1999-01", end="2000-03")
    empty_dl = pd.DataFrame(columns=["permno", "dlstdt", "dlret", "dlstcd"])
    kwargs = dict(cfg=returns.ReturnConfig("1999-01"), audit=audit("2000-02"), delist=empty_dl,
                  calendar=calendar("1999-01", "2000-03"), progress=lambda *_: None)
    base, _, _ = returns.build_panel(msf=data, **kwargs)
    changed = data.copy()
    changed.loc[changed["date"].dt.to_period("M") == pd.Period("2000-02"), "ret"] = 9.0
    altered, _, _ = returns.build_panel(msf=changed, **kwargs)
    a = base[base["yyyymm"] == 200001].iloc[0]
    b = altered[altered["yyyymm"] == 200001].iloc[0]
    for feature in returns.FEATURES:
        assert (pd.isna(a[feature]) and pd.isna(b[feature])) or a[feature] == b[feature]
    assert a["target_ret"] != b["target_ret"]


def synthetic_return_panel(firms=100, start="1999-01", end="2007-11", seed=1):
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    rows = []
    for permno in range(1, firms + 1):
        signal = rng.normal(0, 0.02)
        for month in months:
            y = signal + rng.normal(0, 0.08)
            row = {"permno": permno, "yyyymm": month.year * 100 + month.month, "target_ret": y,
                   "target_includes_dlret": False}
            for feature in returns.FEATURES:
                row[feature] = rng.normal()
            row["ret_1m"] = -signal + rng.normal(0, 0.02)
            row["row_id"] = permno * 1_000_000 + row["yyyymm"]
            row["prediction_time"] = row["feature_available_time"] = month.to_timestamp(how="end").normalize()
            row["label_end_time"] = row["label_available_time"] = (month + 1).to_timestamp(how="end").normalize()
            rows.append(row)
    return pd.DataFrame(rows)


def test_return_runner_and_report_end_to_end(tmp_path):
    out = return_runner.run(models=["zero", "reversal", "ridge"], test_years=[2004], quick=False,
                            panel=synthetic_return_panel(), audit=audit("2007-11"), out_root=tmp_path,
                            progress=lambda *_: None)
    manifest = json.loads((out / "manifest.json").read_text())
    pred = pd.read_parquet(out / "predictions.parquet")
    assert manifest["task"] == "returns" and manifest["code_changed_during_run"] is False
    assert set(pred["model"]) == {"zero", "reversal", "ridge"}
    assert pred.groupby("model")["row_id"].apply(frozenset).nunique() == 1
    report = build_report(out)
    assert (report / "REPORT.md").exists()
    assert {x["model"] for x in json.loads((report / "model_summary.json").read_text())} == {"zero", "reversal", "ridge"}
    assert {x["model"] for x in json.loads((report / "comparisons.json").read_text())} == {"ridge"}


def test_adverse_code_definition_excludes_mergers_active_and_fund_conversion():
    assert {400, 520, 552, 574, 584, 591} <= delisting.ADVERSE_CODES
    assert not ({100, 233, 300, 588} & delisting.ADVERSE_CODES)


def test_delisting_horizon_is_forward_complete_and_uses_adverse_codes_only():
    data = regular_msf(start="1999-01", end="2001-12")
    events = pd.DataFrame({
        "permno": [1, 2, 3], "dlstdt": pd.to_datetime(["2001-01-15", "2000-06-15", "2001-02-01"]),
        "dlret": [-1.0, 0.2, -0.8], "dlstcd": [574, 233, 552],
    })
    panel, _, meta = delisting.build_panel(delisting.DelistingConfig("1999-01"), audit("2000-11", "2000-12"),
                                           data, events, calendar("1999-01", "2001-12"), progress=lambda *_: None)
    jan = panel[panel["yyyymm"] == 200001].set_index("permno")
    assert jan.loc[1, "target_adverse_delisting"] == 1
    assert jan.loc[2, "target_adverse_delisting"] == 0
    assert jan.loc[3, "target_adverse_delisting"] == 0
    assert jan.loc[1, "next_adverse_code"] == 574
    assert jan.loc[1, "label_available_time"].to_period("M") == pd.Period("2001-01")
    assert panel["yyyymm"].max() == 200012
    assert meta["summary"]["unique_positive_events"] >= 1


def synthetic_delisting_panel(firms=100, start="1995-01", end="2007-12", seed=3):
    rng = np.random.default_rng(seed)
    months = pd.period_range(start, end, freq="M")
    rows = []
    for permno in range(1, firms + 1):
        firm_risk = rng.normal()
        for month in months:
            signal = firm_risk + rng.normal(scale=0.8)
            event = int(signal > 1.45)
            row = {"permno": permno, "yyyymm": month.year * 100 + month.month,
                   "target_adverse_delisting": event}
            for feature in delisting.FEATURES:
                row[feature] = rng.normal()
            row["log_price"] = -signal + rng.normal(scale=0.15)
            row["log_mcap"] = -0.7 * signal + rng.normal(scale=0.2)
            row["row_id"] = permno * 1_000_000 + row["yyyymm"]
            row["prediction_time"] = row["feature_available_time"] = month.to_timestamp(how="end").normalize()
            row["label_end_time"] = row["label_available_time"] = (month + 12).to_timestamp(how="end").normalize()
            row["next_adverse_date"] = ((month + 6).to_timestamp(how="end").normalize() if event else pd.NaT)
            row["next_adverse_code"] = 574.0 if event else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def test_delisting_training_cap_preserves_full_class_weight():
    panel = synthetic_delisting_panel(firms=20)
    sampled, weights = stratified_training_sample(panel, max_rows=400, seed=7)
    assert len(sampled) == 400
    for label in (0, 1):
        full = int((panel["target_adverse_delisting"] == label).sum())
        weighted = float(weights[sampled["target_adverse_delisting"].to_numpy() == label].sum())
        assert weighted == pytest.approx(full)


def test_delisting_runner_and_event_report_end_to_end(tmp_path):
    panel = synthetic_delisting_panel()
    out = delisting_runner.run(models=["base_rate", "logit"], test_years=[2004], quick=False,
                                panel=panel, audit=audit("2007-11", "2007-12"), out_root=tmp_path,
                                progress=lambda *_: None)
    manifest = json.loads((out / "manifest.json").read_text())
    pred = pd.read_parquet(out / "predictions.parquet")
    assert manifest["task"] == "delisting" and manifest["reserved_start"] == "2006-01-01"
    assert set(pred["model"]) == {"base_rate", "logit"}
    assert pred.groupby("model")["row_id"].apply(frozenset).nunique() == 1
    assert pred["forecast"].between(0, 1, inclusive="neither").all()

    report = build_delisting_report(out)
    summary = pd.read_json(report / "model_summary.json").set_index("model")
    budgets = pd.read_parquet(report / "budget_metrics.parquet")
    assert summary.loc["logit", "mean_monthly_log_loss"] < summary.loc["base_rate", "mean_monthly_log_loss"]
    assert set(budgets["budget"]) == {0.005, 0.01, 0.05}
    ranked = budgets[budgets["ranking_available"]]
    assert ranked["event_recall"].between(0, 1).all()
    assert budgets.loc[budgets["model"] == "base_rate", "event_recall"].isna().all()
    assert (report / "pairwise_comparisons.json").exists()

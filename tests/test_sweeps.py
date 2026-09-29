import json

import numpy as np
import pandas as pd
import pytest

from p3_modellab import sweep_plots, sweeps
from test_models_runner import synthetic_panel

AUDIT_LONG = {"created_utc": "test", "cutoffs": {"one_month_ahead_last_prediction_month": "2024-11"}, "sources": {}}
FAST = {
    "settings": {"leaves": [4, 31], "learning_rate": [0.1, 0.3], "max_rounds": 100, "patience": 20, "ridge_alphas": [1.0, 1e6]},
    "greedy": {"models": ["ridge", "lightgbm"], "lightgbm": {"leaves": 7, "learning_rate": 0.2, "max_rounds": 60, "patience": 10}},
    "data": {"train_years": [2, 5], "firm_percent": [50, 100], "models": ["har", "ridge", "lightgbm"]},
    "noise": {"noise_sd": [0.0, 0.5, 2.0], "blank_share": [0.0, 0.5], "models": ["ridge", "lightgbm"]},
}


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel(firms=80, start="1995-01", end="2024-11", nonlinear=True)


def run(kind, panel, tmp_path, seeds=(0, 1)):
    return sweeps.run_sweep(kind, seeds=list(seeds), spec_override=FAST[kind], panel=panel, audit=AUDIT_LONG,
                            out_root=tmp_path, progress=lambda *_: None)


def test_fixed_split_obeys_timing_and_never_touches_sealed_years(panel):
    d = sweeps.fixed_split(panel, AUDIT_LONG, 25, [2015, 2016, 2017, 2018, 2019], 24)
    assert d.fit_cutoff == pd.Timestamp("2015-01-31")
    assert (d.train_all["label_available_time"] <= d.train_end).all()
    assert (d.val["label_available_time"] <= d.fit_cutoff).all() and (d.val["label_available_time"] > d.train_end).all()
    assert set(d.eval["yyyymm"] // 100) == {2015, 2016, 2017, 2018, 2019}
    assert (d.eval["label_available_time"] < pd.Timestamp("2022-01-01")).all()
    assert d.eval["permno"].isin(d.train["permno"]).all()  # same fixed firm sample
    assert 0.1 < d.eval["permno"].nunique() / 80 < 0.45


def test_firm_sample_is_deterministic_and_about_the_right_size():
    p = pd.Series(np.arange(10000, 20000))
    a, b = sweeps.firm_sample(p, 25), sweeps.firm_sample(p, 25)
    assert a.equals(b) and 0.23 < a.mean() < 0.27
    assert not a.equals(sweeps.firm_sample(p, 25, salt=1))
    assert sweeps.firm_sample(p, 100).all()


def test_corrupt_has_the_stated_size():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": np.zeros(100_000), "b": np.ones(100_000), "c": np.nan})
    stds = pd.Series({"a": 2.0, "b": 1.0, "c": 1.0})
    noisy = sweeps.corrupt(X, ["a", "b", "c"], stds, 0.5, 0.0, rng)
    assert noisy["a"].std() == pytest.approx(1.0, rel=0.02)  # 0.5 x spread of 2
    assert noisy["c"].isna().all()  # missing stays missing
    blank = sweeps.corrupt(X, ["a", "b"], stds, 0.0, 0.25, rng)
    assert blank["a"].isna().mean() == pytest.approx(0.25, abs=0.01)
    assert (sweeps.corrupt(X, ["a"], stds, 0.0, 0.0, rng)["a"] == 0).all()


def check_common(out):
    man = json.loads((out / "manifest.json").read_text())
    t, m = pd.read_parquet(out / "trials.parquet"), pd.read_parquet(out / "monthly.parquet")
    assert (m.groupby("trial")["qlike"].mean() - t.set_index("trial")["eval_qlike"]).abs().max() < 1e-12
    assert set(t[t["role"] == "reference"]["model"]) == {"har", "persistence"}
    assert man["code_changed_during_run"] is False
    assert all(p.exists() and p.stat().st_size > 1000 for p in sweep_plots.write_figures(out))
    return man, t


def test_settings_sweep(panel, tmp_path):
    out = run("settings", panel, tmp_path)
    man, t = check_common(out)
    lg = t[t["model"] == "lightgbm"]
    assert len(lg) == 2 * 2 * 2 and set(lg["num_leaves"]) == {4, 31} and set(lg["seed"]) == {0, 1}
    assert len(t[(t["model"] == "ridge") & t["role"].isna()]) == 2
    # The interval against HAR must be paired month by month: recompute it independently.
    from p3_modellab.uncertainty import block_bootstrap
    m = pd.read_parquet(out / "monthly.parquet").pivot(index="yyyymm", columns="trial", values="qlike")
    har = t[(t["role"] == "reference") & (t["model"] == "har")]["trial"].item()
    row = lg.iloc[0]
    iv = block_bootstrap((m[row["trial"]] - m[har]).to_numpy(), block=12)
    assert (row["diff_vs_har"], row["diff_vs_har_lo"], row["diff_vs_har_hi"]) == pytest.approx((iv.estimate, iv.lo, iv.hi))


def test_greedy_picks_by_validation_only(panel, tmp_path):
    out = run("greedy", panel, tmp_path, seeds=(0,))
    man, t = check_common(out)
    c = pd.read_parquet(out / "candidates.parquet")
    path = t[t["role"].isna() & (t["step"] > 0)]
    for (model, seed, step), g in c.groupby(["model", "seed", "step"]):
        chosen = path[(path["model"] == model) & (path["seed"] == seed) & (path["step"] == step)]["added"].item()
        assert chosen == g.loc[g["val_qlike"].idxmin(), "candidate"]
    ridge = path[path["model"] == "ridge"].sort_values("step")
    assert ridge["added"].iloc[0] == "log_rv_m"  # the planted strongest input comes first
    assert ridge["step"].max() == 13 and ridge["inputs"].iloc[-1].count(",") == 12
    zero = t[(t["step"] == 0) & (t["model"] == "ridge")]["eval_qlike"].item()
    assert ridge["eval_qlike"].iloc[0] < zero  # one good input beats no inputs


def test_data_sweep_uses_the_stated_windows(panel, tmp_path):
    out = run("data", panel, tmp_path)
    man, t = check_common(out)
    body = t[t["role"].isna()]
    train_end = pd.Timestamp(man["split"]["train_labels_known_by"])
    for years, g in body.groupby("train_years"):
        assert (pd.to_datetime(g["train_labels_from"]) > train_end - pd.DateOffset(years=int(years))).all()
    n = body[(body["model"] == "ridge") & (body["seed"] == 0)].set_index(["train_years", "firm_percent"])["n_train"]
    assert n.loc[(5, 100)] > n.loc[(2, 100)] > n.loc[(2, 50)]
    assert len(body[(body["model"] == "har") & (body["firm_percent"] == 100)]) == 2  # deterministic: one run per window


def test_noise_sweep(panel, tmp_path):
    out = run("noise", panel, tmp_path)
    man, t = check_common(out)
    b = t[t["role"].isna()]
    for (model, seed), g in b.groupby(["model", "seed"]):
        clean_noise = g[(g["corruption"] == "noise") & (g["level"] == 0)]["eval_qlike"].item()
        clean_blank = g[(g["corruption"] == "blank") & (g["level"] == 0)]["eval_qlike"].item()
        assert clean_noise == clean_blank
    lg = b[(b["model"] == "lightgbm") & (b["corruption"] == "noise")].groupby("level")["eval_qlike"].mean()
    assert lg.loc[2.0] > lg.loc[0.0]


def test_demo_size_is_small_and_labelled(panel, tmp_path):
    out = sweeps.run_sweep("greedy", demo=True, panel=panel, audit=AUDIT_LONG, out_root=tmp_path, progress=lambda *_: None)
    man = json.loads((out / "manifest.json").read_text())
    t = pd.read_parquet(out / "trials.parquet")
    assert out.name.endswith("-greedy-demo") and man["size"] == "demo" and man["seeds"] == [0] and man["firm_percent"] == 10
    assert t["step"].max() == 3


def test_unknown_kind_refused(panel, tmp_path):
    with pytest.raises(ValueError, match="unknown sweep kind"):
        sweeps.run_sweep("colour", panel=panel, audit=AUDIT_LONG, out_root=tmp_path)

"""Sweeps: keep the data fixed, change one or two things, record how the score moves.

Fixed data for every sweep:
- evaluation: forecasts made in the evaluation years (default 2015 to 2019) for a
  fixed sample of firms (default 25 percent, chosen by a fixed rule on permno);
- fit time c = the first evaluation forecast; validation = answers known in the
  24 months before c; training = answers known before that;
- the sealed years are never used (the split refuses them).

Rules: every choice (best setting, next greedy input, early stopping, multiplier)
uses validation rows only. The evaluation score is recorded for every point but
never used to choose. Each point is repeated over seeds.

Kinds:
- settings: LightGBM tree size x learning rate grid; Ridge penalty line
- greedy: add inputs one at a time, choosing the one that helps validation most
- data: training years x share of firms used for training
- noise: add noise to, or blank out, inputs at evaluation time (models fit on clean data)
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .contract import assert_disjoint, assert_labels_known, assert_predictable, validate_panel
from .data.audit import load_audit
from .losses import qlike
from .manifest import code_hash, environment
from .models import FLOOR, HAR, LightGBMModel, Persistence, RidgeModel, _floor, _scale, make_model
from .runner import reserved_start, say
from .splits import walk_forward
from .tasks import volatility
from .uncertainty import block_bootstrap, monthly_means

KINDS = ("settings", "greedy", "data", "noise")

DEFAULTS = {
    "firm_percent": 25,
    "eval_years": [2015, 2016, 2017, 2018, 2019],
    "validation_months": 24,
    "seeds": [0, 1, 2],
    "settings": {"leaves": [7, 15, 31, 63, 127, 255], "learning_rate": [0.02, 0.05, 0.1, 0.2], "max_rounds": 3000, "patience": 100,
                 "ridge_alphas": [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]},
    "greedy": {"models": ["ridge", "lightgbm"], "max_steps": None, "ridge_alphas": None,
               "lightgbm": {"leaves": 31, "learning_rate": 0.1, "max_rounds": 500, "patience": 50}},
    "data": {"train_years": [2, 5, 10, 20], "firm_percent": [10, 25, 50, 100], "models": ["har", "ridge", "lightgbm"]},
    "noise": {"noise_sd": [0.0, 0.1, 0.25, 0.5, 1.0], "blank_share": [0.0, 0.1, 0.25, 0.5], "models": ["ridge", "lightgbm"]},
}


# "Demo" size: only checks that the calculations run end to end (about a minute in total).
DEMO = {
    "firm_percent": 10,
    "seeds": [0],
    "settings": {"leaves": [15, 63], "learning_rate": [0.1, 0.2], "max_rounds": 200, "patience": 20, "ridge_alphas": [1e3, 1e5, 1e7]},
    "greedy": {"models": ["ridge", "lightgbm"], "max_steps": 3, "ridge_alphas": [1e5],
               "lightgbm": {"leaves": 15, "learning_rate": 0.2, "max_rounds": 100, "patience": 10}},
    "data": {"train_years": [2, 10], "firm_percent": [25, 100], "models": ["har", "ridge"]},
    "noise": {"noise_sd": [0.0, 0.5], "blank_share": [0.0, 0.25], "models": ["ridge"]},
}


def firm_sample(permno: pd.Series, percent: float, salt: int = 0) -> pd.Series:
    """Deterministic firm sample: the same firms for the same (percent, salt)."""
    if percent >= 100:
        return pd.Series(True, index=permno.index)
    return ((permno.astype("int64") * 2654435761 + salt * 40503) % 1000) < percent * 10


@dataclass
class SweepData:
    train_all: pd.DataFrame = field(repr=False)
    train: pd.DataFrame = field(repr=False)
    val: pd.DataFrame = field(repr=False)
    eval: pd.DataFrame = field(repr=False)
    fit_cutoff: pd.Timestamp = None
    train_end: pd.Timestamp = None

    def info(self) -> dict:
        return {"fit_cutoff": str(self.fit_cutoff.date()), "train_labels_known_by": str(self.train_end.date()),
                "n_train_all_firms": len(self.train_all), "n_train": len(self.train), "n_val": len(self.val), "n_eval": len(self.eval),
                "eval_months": int(self.eval["yyyymm"].nunique()), "eval_firms": int(self.eval["permno"].nunique())}


def fixed_split(panel: pd.DataFrame, audit: dict, firm_percent: float, eval_years: list[int], validation_months: int) -> SweepData:
    validate_panel(panel)
    (f,) = walk_forward(panel, eval_years, reserved_start(audit), "exploratory", validation_months=validation_months, refit_every=len(eval_years))
    by_id = panel.set_index("row_id")
    train_all = by_id.loc[f.train_ids].reset_index()
    val, ev = by_id.loc[f.val_ids].reset_index(), by_id.loc[f.test_ids].reset_index()
    keep = lambda d: d[firm_sample(d["permno"], firm_percent)].reset_index(drop=True)
    data = SweepData(train_all, keep(train_all), keep(val), keep(ev), f.fit_cutoff, f.train_end)
    assert_labels_known(data.train_all, f.train_end, "train")
    assert_labels_known(data.val, f.fit_cutoff, "validation")
    assert_predictable(data.eval, f.fit_cutoff)
    assert_disjoint(train=data.train_all["row_id"].to_numpy(), val=data.val["row_id"].to_numpy(), eval=data.eval["row_id"].to_numpy())
    return data


class Constant:
    """No inputs at all: one forecast for everyone (the training average on the log scale, rescaled)."""
    name = "constant"
    kind = "baseline"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        self.g_ = float(train["target_log_rv"].mean())
        self.scale_ = _scale(val["target_log_rv"].to_numpy(), np.full(len(val), self.g_))
        h = _floor(np.full(len(val), self.scale_ * np.exp(self.g_)))
        return {"trials": [{"val_qlike": float(qlike(val["target_rv"].to_numpy(), h).mean())}], "chosen": {}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _floor(np.full(len(df), self.scale_ * np.exp(self.g_)))


def _val_score(rec: dict) -> float:
    return min(t["val_qlike"] for t in rec["trials"]) if rec["trials"] else np.nan


class Recorder:
    """Collects one row per trial plus its monthly evaluation losses."""

    def __init__(self, data: SweepData):
        self.data = data
        self.trials: list[dict] = []
        self.monthly: list[pd.DataFrame] = []
        self.months = data.eval["yyyymm"].to_numpy()
        self.y = data.eval["target_rv"].to_numpy()
        self.ref_monthly: pd.Series | None = None

    def add(self, params: dict, model, rec: dict | None, X_eval: pd.DataFrame | None = None, seconds: float = 0.0) -> dict:
        h = model.predict(self.data.eval if X_eval is None else X_eval)
        m = monthly_means(qlike(self.y, h), self.months)
        tid = len(self.trials)
        row = {"trial": tid, **params, "val_qlike": _val_score(rec) if rec else np.nan, "eval_qlike": float(m.mean()), "seconds": round(seconds, 2)}
        if rec and rec.get("chosen"):
            row.update({f"chosen_{k}": v for k, v in rec["chosen"].items()})
        if self.ref_monthly is not None:
            iv = block_bootstrap((m - self.ref_monthly).to_numpy(), block=12)
            row.update(diff_vs_har=iv.estimate, diff_vs_har_lo=iv.lo, diff_vs_har_hi=iv.hi)
        self.trials.append(row)
        self.monthly.append(pd.DataFrame({"trial": tid, "yyyymm": m.index, "qlike": m.to_numpy()}))
        return row

    def reference(self) -> None:
        """Clean HAR and persistence on the fixed data, for comparison lines."""
        har = HAR()
        rec = har.fit(self.data.train, self.data.val)
        row = self.add({"model": "har", "role": "reference"}, har, rec)
        self.ref_monthly = self.monthly[row["trial"]].set_index("yyyymm")["qlike"]
        self.add({"model": "persistence", "role": "reference"}, Persistence(), None)


def _fit(model, train, val):
    t0 = time.time()
    rec = model.fit(train, val)
    return rec, time.time() - t0


def sweep_settings(r: Recorder, spec: dict, seeds: list[int], progress) -> None:
    d = r.data
    for leaves in spec["leaves"]:
        for lr in spec["learning_rate"]:
            for seed in seeds:
                m = LightGBMModel(volatility.FEATURES, seed=seed, leaves=(leaves,), learning_rate=lr, max_rounds=spec["max_rounds"], patience=spec["patience"])
                rec, sec = _fit(m, d.train, d.val)
                row = r.add({"model": "lightgbm", "num_leaves": leaves, "learning_rate": lr, "seed": seed}, m, rec, seconds=sec)
                progress(f"settings lightgbm leaves={leaves} lr={lr} seed={seed}: eval {row['eval_qlike']:.4f} ({sec:.0f}s)")
    for a in spec["ridge_alphas"]:
        m = RidgeModel(volatility.FEATURES, alphas=(a,))
        rec, sec = _fit(m, d.train, d.val)
        r.add({"model": "ridge", "alpha": a, "seed": 0}, m, rec, seconds=sec)
    progress("settings ridge done (deterministic, one seed)")


def _greedy_model(name: str, features: list[str], seed: int, spec: dict):
    if name == "ridge":
        return RidgeModel(features, alphas=tuple(spec["ridge_alphas"])) if spec.get("ridge_alphas") else RidgeModel(features)
    g = spec["lightgbm"]
    return LightGBMModel(features, seed=seed, leaves=(g["leaves"],), learning_rate=g["learning_rate"], max_rounds=g["max_rounds"], patience=g["patience"])


def sweep_greedy(r: Recorder, spec: dict, seeds: list[int], progress) -> pd.DataFrame:
    d = r.data
    candidates = []
    for name in spec["models"]:
        for seed in (seeds if name == "lightgbm" else seeds[:1]):
            const = Constant()
            rec, sec = _fit(const, d.train, d.val)
            r.add({"model": name, "seed": seed, "step": 0, "added": "(no inputs)", "inputs": ""}, const, rec, seconds=sec)
            chosen, remaining = [], list(volatility.FEATURES)
            step = 0
            limit = spec.get("max_steps") or len(remaining)
            while remaining and step < limit:
                step += 1
                best = None
                for f in remaining:
                    m = _greedy_model(name, chosen + [f], seed, spec)
                    rec, sec = _fit(m, d.train, d.val)
                    v = _val_score(rec)
                    candidates.append({"model": name, "seed": seed, "step": step, "candidate": f, "val_qlike": v, "seconds": round(sec, 2)})
                    if best is None or v < best[0]:
                        best = (v, f, m, rec, sec)
                _, f, m, rec, sec = best
                chosen.append(f)
                remaining.remove(f)
                row = r.add({"model": name, "seed": seed, "step": step, "added": f, "inputs": ",".join(chosen)}, m, rec, seconds=sec)
                progress(f"greedy {name} seed={seed} step {step}: +{f} val {row['val_qlike']:.4f} eval {row['eval_qlike']:.4f}")
    return pd.DataFrame(candidates)


def sweep_data(r: Recorder, spec: dict, seeds: list[int], progress) -> None:
    d = r.data
    for years in spec["train_years"]:
        window = d.train_all[d.train_all["label_available_time"] > d.train_end - pd.DateOffset(years=years)]
        for pct in spec["firm_percent"]:
            for seed in seeds:
                tr = window[firm_sample(window["permno"], pct, salt=seed + 1)].reset_index(drop=True)
                for name in spec["models"]:
                    if pct >= 100 and name != "lightgbm" and seed != seeds[0]:
                        continue  # same firms and no randomness: repeats would be identical
                    m = make_model(name, volatility.FEATURES, seed=seed)
                    rec, sec = _fit(m, tr, d.val)
                    row = r.add({"model": name, "train_years": years, "firm_percent": pct, "seed": seed, "n_train": len(tr),
                                 "train_labels_from": tr["label_available_time"].min()}, m, rec, seconds=sec)
                    progress(f"data {name} years={years} firms={pct}% seed={seed} n={len(tr):,}: eval {row['eval_qlike']:.4f} ({sec:.0f}s)")


def corrupt(X: pd.DataFrame, cols: list[str], stds: pd.Series, noise_sd: float, blank_share: float, rng: np.random.Generator) -> pd.DataFrame:
    """Add Gaussian noise (in units of each input's training spread) and/or blank a random share of input values."""
    out = X.copy()
    vals = out[cols].to_numpy(dtype="float64")
    if noise_sd > 0:
        vals = vals + rng.normal(0.0, 1.0, vals.shape) * stds[cols].to_numpy() * noise_sd
    if blank_share > 0:
        vals[rng.random(vals.shape) < blank_share] = np.nan
    out[cols] = vals
    return out


def sweep_noise(r: Recorder, spec: dict, seeds: list[int], progress) -> None:
    d = r.data
    cols = volatility.FEATURES
    stds = d.train[cols].std()
    for seed in seeds:
        for name in spec["models"]:
            m = make_model(name, cols, seed=seed)
            rec, sec = _fit(m, d.train, d.val)
            for s in spec["noise_sd"]:
                X = corrupt(d.eval, cols, stds, s, 0.0, np.random.default_rng([seed, 1, int(s * 1000)]))
                r.add({"model": name, "seed": seed, "corruption": "noise", "level": s}, m, rec, X_eval=X, seconds=sec)
            for b in spec["blank_share"]:
                X = corrupt(d.eval, cols, stds, 0.0, b, np.random.default_rng([seed, 2, int(b * 1000)]))
                r.add({"model": name, "seed": seed, "corruption": "blank", "level": b}, m, rec, X_eval=X, seconds=sec)
            progress(f"noise {name} seed={seed} done")


def run_sweep(kind: str, seeds: list[int] | None = None, firm_percent: float | None = None, spec_override: dict | None = None,
              panel: pd.DataFrame | None = None, audit: dict | None = None, out_root: Path | None = None, progress=say,
              demo: bool = False) -> Path:
    started_code = code_hash()
    if kind not in KINDS:
        raise ValueError(f"unknown sweep kind {kind!r}; choose from {KINDS}")
    if demo:
        seeds = seeds if seeds is not None else DEMO["seeds"]
        firm_percent = firm_percent if firm_percent is not None else DEMO["firm_percent"]
        spec_override = {**DEMO[kind], **(spec_override or {})}
    audit = audit or load_audit()
    if panel is None:
        panel, meta = volatility.load_or_build(progress=progress)
        panel_info = {"source": "results/panels", "code_hash": meta["code_hash"], "created_utc": meta["created_utc"]}
    else:
        panel_info = {"source": "supplied by caller", "rows": int(len(panel))}
    seeds = seeds if seeds is not None else DEFAULTS["seeds"]
    fp = firm_percent if firm_percent is not None else DEFAULTS["firm_percent"]
    spec = json.loads(json.dumps({**DEFAULTS[kind], **(spec_override or {})}))
    data = fixed_split(panel, audit, fp, DEFAULTS["eval_years"], DEFAULTS["validation_months"])
    progress(f"fixed data: {data.info()}")

    sweep_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}-{kind}{'-demo' if demo else ''}"
    out = Path(out_root or config.RESULTS_DIR / "sweeps") / sweep_id
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    r = Recorder(data)
    r.reference()
    extra = None
    if kind == "settings":
        sweep_settings(r, spec, seeds, progress)
    elif kind == "greedy":
        extra = sweep_greedy(r, spec, seeds, progress)
    elif kind == "data":
        sweep_data(r, spec, seeds, progress)
    else:
        sweep_noise(r, spec, seeds, progress)

    trials = pd.DataFrame(r.trials)
    trials.to_parquet(out / "trials.parquet", index=False)
    pd.concat(r.monthly, ignore_index=True).to_parquet(out / "monthly.parquet", index=False)
    if extra is not None:
        extra.to_parquet(out / "candidates.parquet", index=False)
    manifest = {
        "sweep_id": sweep_id, "kind": kind, "size": "demo" if demo else "standard",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidence": "exploratory: evaluation years have been used by earlier exploratory runs",
        "rule": "every choice uses validation rows only; the evaluation score is recorded for every point but never used to choose",
        "firm_percent": fp, "eval_years": DEFAULTS["eval_years"], "validation_months": DEFAULTS["validation_months"],
        "seeds": seeds, "spec": spec, "split": data.info(), "features": volatility.FEATURES,
        "code_hash": started_code, "code_changed_during_run": code_hash() != started_code,
        "panel": panel_info, "audit_created_utc": audit.get("created_utc"),
        "n_trials": len(trials), "total_seconds": round(time.time() - t0, 1), "environment": environment(),
        "reference_note": "HAR and persistence fitted on the same fixed data; diff_vs_har is paired by month with a 12-month block bootstrap interval",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out

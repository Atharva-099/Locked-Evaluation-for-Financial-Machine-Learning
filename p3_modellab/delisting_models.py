"""Probability models for twelve-month adverse-delisting risk.

The learned models use a deterministic, class-stratified cap on training rows.
Sampling weights restore the full training distribution, so the probabilities
still target the observed event rate instead of the sampled event rate.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import MODEL_CORES, Winsorizer


DELISTING_MODEL_NAMES = ("base_rate", "logit", "lightgbm", "xgboost")
DELISTING_BASELINES = ("base_rate",)
PROBABILITY_FLOOR = 1e-6


def _probability(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype="float64"), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)


def _validation_loss(val: pd.DataFrame, forecast: np.ndarray) -> float:
    return float(log_loss(val["target_adverse_delisting"].to_numpy(), _probability(forecast), labels=[0, 1]))


def stratified_training_sample(
    train: pd.DataFrame, max_rows: int | None, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep both classes and return inverse-sampling weights for calibration."""
    if max_rows is None or len(train) <= max_rows:
        return train, np.ones(len(train), dtype="float64")
    if max_rows < 2:
        raise ValueError("max_rows must allow both outcome classes")
    y = train["target_adverse_delisting"].to_numpy(dtype="int8")
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    if any(len(idx) == 0 for idx in class_indices):
        raise ValueError("delisting training rows must contain both outcome classes")

    # Keep every member of the rare class when possible, but never let one
    # class consume the entire cap. This also behaves sensibly on test fixtures.
    rare_label = int(len(class_indices[1]) < len(class_indices[0]))
    rare = class_indices[rare_label]
    common = class_indices[1 - rare_label]
    n_rare = min(len(rare), max_rows // 2)
    n_common = min(len(common), max_rows - n_rare)
    if n_rare + n_common < max_rows:
        n_rare = min(len(rare), max_rows - n_common)
    rng = np.random.default_rng(seed)
    picked_rare = rng.choice(rare, n_rare, replace=False)
    picked_common = rng.choice(common, n_common, replace=False)
    positions = np.sort(np.concatenate([picked_rare, picked_common]))
    sampled = train.iloc[positions]
    counts = {rare_label: (len(rare), n_rare), 1 - rare_label: (len(common), n_common)}
    weights = np.array([counts[int(label)][0] / counts[int(label)][1] for label in y[positions]])
    return sampled, weights


class HistoricalBaseRate:
    name = "base_rate"
    kind = "baseline"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        positives = float(train["target_adverse_delisting"].sum())
        self.rate_ = (positives + 0.5) / (len(train) + 1.0)
        return {"trials": [], "chosen": {"training_event_rate": self.rate_}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.rate_, dtype="float64")


class DelistingLogit:
    name = "logit"
    kind = "model"

    def __init__(self, features: list[str], seed: int = 0, cs: tuple[float, ...] = (0.01, 1.0, 100.0),
                 max_train_rows: int | None = 400_000):
        self.features, self.seed, self.cs = list(features), seed, cs
        self.max_train_rows = max_train_rows

    def _pipe(self, c: float) -> Pipeline:
        return Pipeline([
            ("winsorize", Winsorizer()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=c, solver="lbfgs", max_iter=500, random_state=self.seed)),
        ])

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        sampled, weights = stratified_training_sample(train, self.max_train_rows, self.seed)
        y = sampled["target_adverse_delisting"].to_numpy(dtype="int8")
        trials, best = [], None
        for c in self.cs:
            t0 = time.time()
            pipe = self._pipe(c)
            pipe.fit(sampled[self.features], y, logit__sample_weight=weights)
            score = _validation_loss(val, pipe.predict_proba(val[self.features])[:, 1])
            record = {"C": c, "val_log_loss": score, "seconds": round(time.time() - t0, 2)}
            trials.append(record)
            if best is None or score < best[0]:
                best = (score, pipe, c)
        _, self.pipe_, self.c_ = best
        return {"trials": trials, "chosen": {"C": self.c_}, "training_rows_used": len(sampled)}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _probability(self.pipe_.predict_proba(df[self.features])[:, 1])


class DelistingLightGBM:
    name = "lightgbm"
    kind = "model"

    def __init__(self, features: list[str], seed: int = 0, leaves: tuple[int, ...] = (15, 63),
                 learning_rate: float = 0.05, max_rounds: int = 1200, patience: int = 75,
                 max_train_rows: int | None = 500_000, n_jobs: int | None = None):
        self.features, self.seed, self.leaves = list(features), seed, leaves
        self.learning_rate, self.max_rounds, self.patience = learning_rate, max_rounds, patience
        self.max_train_rows, self.n_jobs = max_train_rows, n_jobs or MODEL_CORES

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        import lightgbm as lgb

        sampled, weights = stratified_training_sample(train, self.max_train_rows, self.seed)
        y = sampled["target_adverse_delisting"].to_numpy(dtype="int8")
        y_val = val["target_adverse_delisting"].to_numpy(dtype="int8")
        trials, best = [], None
        for leaves in self.leaves:
            t0 = time.time()
            model = lgb.LGBMClassifier(
                objective="binary", n_estimators=self.max_rounds, learning_rate=self.learning_rate,
                num_leaves=leaves, min_child_samples=200, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=self.seed,
                n_jobs=self.n_jobs, deterministic=True, force_row_wise=True, verbose=-1,
            )
            model.fit(
                sampled[self.features], y, sample_weight=weights,
                eval_X=(val[self.features],), eval_y=(y_val,), eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(self.patience, verbose=False)],
            )
            forecast = model.predict_proba(val[self.features], num_iteration=model.best_iteration_)[:, 1]
            score = _validation_loss(val, forecast)
            record = {"num_leaves": leaves, "best_iteration": int(model.best_iteration_),
                      "val_log_loss": score, "seconds": round(time.time() - t0, 2)}
            trials.append(record)
            if best is None or score < best[0]:
                best = (score, model, leaves)
        _, self.model_, self.leaves_ = best
        return {"trials": trials,
                "chosen": {"num_leaves": self.leaves_, "best_iteration": int(self.model_.best_iteration_)},
                "training_rows_used": len(sampled)}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _probability(self.model_.predict_proba(
            df[self.features], num_iteration=self.model_.best_iteration_
        )[:, 1])


class DelistingXGBoost:
    name = "xgboost"
    kind = "model"

    def __init__(self, features: list[str], seed: int = 0, depths: tuple[int, ...] = (3, 6),
                 learning_rate: float = 0.05, max_rounds: int = 1200, patience: int = 75,
                 max_train_rows: int | None = 500_000, n_jobs: int | None = None):
        self.features, self.seed, self.depths = list(features), seed, depths
        self.learning_rate, self.max_rounds, self.patience = learning_rate, max_rounds, patience
        self.max_train_rows, self.n_jobs = max_train_rows, n_jobs or MODEL_CORES

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        import xgboost as xgb

        sampled, weights = stratified_training_sample(train, self.max_train_rows, self.seed)
        y = sampled["target_adverse_delisting"].to_numpy(dtype="int8")
        y_val = val["target_adverse_delisting"].to_numpy(dtype="int8")
        trials, best = [], None
        for depth in self.depths:
            t0 = time.time()
            model = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                n_estimators=self.max_rounds, learning_rate=self.learning_rate, max_depth=depth,
                min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                early_stopping_rounds=self.patience, random_state=self.seed, n_jobs=self.n_jobs,
            )
            model.fit(sampled[self.features], y, sample_weight=weights,
                      eval_set=[(val[self.features], y_val)], verbose=False)
            forecast = model.predict_proba(val[self.features], iteration_range=(0, model.best_iteration + 1))[:, 1]
            score = _validation_loss(val, forecast)
            record = {"max_depth": depth, "best_iteration": int(model.best_iteration),
                      "val_log_loss": score, "seconds": round(time.time() - t0, 2)}
            trials.append(record)
            if best is None or score < best[0]:
                best = (score, model, depth)
        _, self.model_, self.depth_ = best
        return {"trials": trials,
                "chosen": {"max_depth": self.depth_, "best_iteration": int(self.model_.best_iteration)},
                "training_rows_used": len(sampled)}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _probability(self.model_.predict_proba(
            df[self.features], iteration_range=(0, self.model_.best_iteration + 1)
        )[:, 1])


def make_delisting_model(name: str, features: list[str], seed: int = 0, quick: bool = False):
    if name == "base_rate":
        return HistoricalBaseRate()
    if name == "logit":
        return DelistingLogit(features, seed=seed, cs=(1.0,) if quick else (0.01, 1.0, 100.0),
                              max_train_rows=80_000 if quick else 400_000)
    if name == "lightgbm":
        return DelistingLightGBM(features, seed=seed, leaves=(31,) if quick else (15, 63),
                                 max_rounds=250 if quick else 1200,
                                 max_train_rows=100_000 if quick else 500_000)
    if name == "xgboost":
        return DelistingXGBoost(features, seed=seed, depths=(4,) if quick else (3, 6),
                                max_rounds=250 if quick else 1200,
                                max_train_rows=100_000 if quick else 500_000)
    raise ValueError(f"unknown delisting model {name!r}; choose from {DELISTING_MODEL_NAMES}")


def delisting_model_spec(name: str, features: list[str], seed: int = 0, quick: bool = False) -> dict:
    model = make_delisting_model(name, features, seed, quick)
    out = {"name": name, "kind": model.kind}
    for key in ("cs", "leaves", "depths", "learning_rate", "max_rounds", "patience",
                "max_train_rows", "seed", "n_jobs"):
        if hasattr(model, key):
            out[key] = getattr(model, key)
    if name not in DELISTING_BASELINES:
        out["features"] = list(features)
    return out

"""Forecasters for the volatility task. Each one:
- fit(train, val): learns from train rows only; uses val rows only for choosing
  settings, early stopping and the final scale; returns a record of every trial;
- predict(df): variance forecasts (21-day units), always positive.

Learned models predict log variance. Turning a log forecast g back into a
variance forecast needs a multiplier. We use the log-normal formula
s = exp(mean(r) + var(r) / 2), with r = log answer - g on the validation rows
(chosen 2026-09-26). The plain average of answer / forecast was rejected because
a single extreme row could move it by 74 percent.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .losses import qlike

FLOOR = 1e-6
MODEL_CORES = min(3, config.N_JOBS)  # keep the laptop usable while models train


def _floor(h: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(h, dtype="float64"), FLOOR)


def _scale(y_log: np.ndarray, g: np.ndarray) -> float:
    """Log-normal multiplier for exp(g) forecasts, from log answers y_log."""
    r = np.asarray(y_log, dtype="float64") - np.asarray(g, dtype="float64")
    return float(np.exp(r.mean() + r.var() / 2))


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned from the training rows (missing values ignored and kept)."""

    def __init__(self, lower: float = 0.001, upper: float = 0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype="float64")
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype="float64"), self.lo_, self.hi_)


class Persistence:
    name = "persistence"
    kind = "baseline"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        return {"trials": [], "chosen": {}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _floor(df["rv_m"].to_numpy())


def har_inputs(df: pd.DataFrame) -> np.ndarray:
    """Daily, weekly, monthly variance; a missing daily value falls back to weekly, weekly to monthly."""
    m = df["rv_m"].to_numpy(dtype="float64")
    w = df["rv_w"].fillna(df["rv_m"]).to_numpy(dtype="float64")
    d = df["rv_d"].fillna(df["rv_w"]).fillna(df["rv_m"]).to_numpy(dtype="float64")
    return np.column_stack([d, w, m])


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


class HAR:
    """HAR benchmark, fitted by least squares on training rows. Log or level form, chosen on validation QLIKE."""

    name = "har"
    kind = "baseline"

    def __init__(self, forms: tuple[str, ...] = ("log", "level")):
        self.forms = forms

    def _design(self, df: pd.DataFrame, form: str) -> np.ndarray:
        X = har_inputs(df)
        return np.log(_floor(X)) if form == "log" else X

    def _raw(self, df: pd.DataFrame, form: str, coef: np.ndarray) -> np.ndarray:
        return coef[0] + self._design(df, form) @ coef[1:]

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        trials = []
        y_val = val["target_rv"].to_numpy()
        for form in self.forms:
            t0 = time.time()
            if form == "log":
                coef = _ols(self._design(train, form), train["target_log_rv"].to_numpy())
                g = self._raw(val, form, coef)
                s = _scale(val["target_log_rv"].to_numpy(), g)
                h = _floor(s * np.exp(g))
            else:
                coef = _ols(self._design(train, form), train["target_rv"].to_numpy())
                s = 1.0
                h = _floor(self._raw(val, form, coef))
            trials.append({"form": form, "coef": coef.tolist(), "scale": s, "val_qlike": float(qlike(y_val, h).mean()), "seconds": round(time.time() - t0, 3)})
        best = min(trials, key=lambda t: t["val_qlike"])
        self.form_, self.coef_, self.scale_ = best["form"], np.array(best["coef"]), best["scale"]
        return {"trials": trials, "chosen": {"form": self.form_}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self._raw(df, self.form_, self.coef_)
        return _floor(self.scale_ * np.exp(raw)) if self.form_ == "log" else _floor(raw)


class RidgeModel:
    name = "ridge"
    kind = "model"

    def __init__(self, features: list[str], alphas: tuple[float, ...] = (1e2, 1e4, 1e5, 1e6, 1e7, 1e8)):
        self.features = list(features)
        self.alphas = alphas

    def _pipe(self, alpha: float) -> Pipeline:
        return Pipeline([
            ("winsorize", Winsorizer()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        Xtr, ytr = train[self.features], train["target_log_rv"].to_numpy()
        Xva, y_val = val[self.features], val["target_rv"].to_numpy()
        trials, best = [], None
        for a in self.alphas:
            t0 = time.time()
            pipe = self._pipe(a).fit(Xtr, ytr)
            g = pipe.predict(Xva)
            s = _scale(val["target_log_rv"].to_numpy(), g)
            score = float(qlike(y_val, _floor(s * np.exp(g))).mean())
            trials.append({"alpha": a, "scale": s, "val_qlike": score, "seconds": round(time.time() - t0, 3)})
            if best is None or score < best[0]:
                best = (score, pipe, s, a)
        _, self.pipe_, self.scale_, self.alpha_ = best
        return {"trials": trials, "chosen": {"alpha": self.alpha_}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _floor(self.scale_ * np.exp(self.pipe_.predict(df[self.features])))


class LightGBMModel:
    name = "lightgbm"
    kind = "model"

    def __init__(self, features: list[str], seed: int = 0, leaves: tuple[int, ...] = (31, 127),
                 learning_rate: float = 0.05, max_rounds: int = 2000, patience: int = 100, n_jobs: int | None = None):
        self.features = list(features)
        self.seed = seed
        self.leaves = leaves
        self.learning_rate = learning_rate
        self.max_rounds = max_rounds
        self.patience = patience
        self.n_jobs = n_jobs or MODEL_CORES

    def _params(self, leaves: int) -> dict:
        return dict(objective="l2", n_estimators=self.max_rounds, learning_rate=self.learning_rate, num_leaves=leaves,
                    min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                    random_state=self.seed, n_jobs=self.n_jobs, deterministic=True, force_row_wise=True, verbose=-1)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        import lightgbm as lgb

        Xtr, ytr = train[self.features], train["target_log_rv"].to_numpy()
        Xva, y_val = val[self.features], val["target_rv"].to_numpy()
        trials, best = [], None
        for leaves in self.leaves:
            t0 = time.time()
            m = lgb.LGBMRegressor(**self._params(leaves))
            m.fit(Xtr, ytr, eval_X=(Xva,), eval_y=(val["target_log_rv"].to_numpy(),),
                  callbacks=[lgb.early_stopping(self.patience, verbose=False)])
            g = m.predict(Xva, num_iteration=m.best_iteration_)
            s = _scale(val["target_log_rv"].to_numpy(), g)
            score = float(qlike(y_val, _floor(s * np.exp(g))).mean())
            trials.append({"num_leaves": leaves, "best_iteration": int(m.best_iteration_), "scale": s,
                           "val_qlike": score, "seconds": round(time.time() - t0, 1)})
            if best is None or score < best[0]:
                best = (score, m, s, leaves)
        _, self.model_, self.scale_, self.leaves_ = best
        return {"trials": trials, "chosen": {"num_leaves": self.leaves_, "best_iteration": int(self.model_.best_iteration_)}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        g = self.model_.predict(df[self.features], num_iteration=self.model_.best_iteration_)
        return _floor(self.scale_ * np.exp(g))


def _sample_rows(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    """A fixed random subset of training rows (for models that are slow on the full set)."""
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=seed)


def _score(val: pd.DataFrame, g: np.ndarray) -> tuple[float, float]:
    s = _scale(val["target_log_rv"].to_numpy(), g)
    return s, float(qlike(val["target_rv"].to_numpy(), _floor(s * np.exp(g))).mean())


class XGBoostModel:
    """Gradient-boosted trees (another implementation of the LightGBM idea), grown level by level."""
    name = "xgboost"
    kind = "model"

    def __init__(self, features: list[str], seed: int = 0, depths: tuple[int, ...] = (6, 10), learning_rate: float = 0.05,
                 max_rounds: int = 2000, patience: int = 100, n_jobs: int | None = None):
        self.features, self.seed, self.depths = list(features), seed, depths
        self.learning_rate, self.max_rounds, self.patience = learning_rate, max_rounds, patience
        self.n_jobs = n_jobs or MODEL_CORES

    def _params(self, depth: int) -> dict:
        return dict(n_estimators=self.max_rounds, learning_rate=self.learning_rate, max_depth=depth, min_child_weight=50,
                    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist", objective="reg:squarederror",
                    eval_metric="rmse", early_stopping_rounds=self.patience, random_state=self.seed, n_jobs=self.n_jobs)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        import xgboost as xgb
        trials, best = [], None
        for depth in self.depths:
            t0 = time.time()
            m = xgb.XGBRegressor(**self._params(depth))
            m.fit(train[self.features], train["target_log_rv"].to_numpy(),
                  eval_set=[(val[self.features], val["target_log_rv"].to_numpy())], verbose=False)
            g = m.predict(val[self.features], iteration_range=(0, m.best_iteration + 1))
            s, score = _score(val, g)
            trials.append({"max_depth": depth, "best_iteration": int(m.best_iteration), "scale": s, "val_qlike": score, "seconds": round(time.time() - t0, 1)})
            if best is None or score < best[0]:
                best = (score, m, s, depth)
        _, self.model_, self.scale_, self.depth_ = best
        return {"trials": trials, "chosen": {"max_depth": self.depth_, "best_iteration": int(self.model_.best_iteration)}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        g = self.model_.predict(df[self.features], iteration_range=(0, self.model_.best_iteration + 1))
        return _floor(self.scale_ * np.exp(g))


class CatBoostModel:
    """Gradient-boosted trees with symmetric (balanced) trees; a third implementation of the boosting idea."""
    name = "catboost"
    kind = "model"

    def __init__(self, features: list[str], seed: int = 0, depths: tuple[int, ...] = (6,), learning_rate: float = 0.08,
                 max_rounds: int = 1500, patience: int = 100, n_jobs: int | None = None):
        self.features, self.seed, self.depths = list(features), seed, depths
        self.learning_rate, self.max_rounds, self.patience = learning_rate, max_rounds, patience
        self.n_jobs = n_jobs or MODEL_CORES

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        from catboost import CatBoostRegressor
        trials, best = [], None
        for depth in self.depths:
            t0 = time.time()
            m = CatBoostRegressor(iterations=self.max_rounds, learning_rate=self.learning_rate, depth=depth, loss_function="RMSE",
                                  random_seed=self.seed, thread_count=self.n_jobs, od_type="Iter", od_wait=self.patience,
                                  use_best_model=True, verbose=False, allow_writing_files=False)
            m.fit(train[self.features], train["target_log_rv"].to_numpy(), eval_set=(val[self.features], val["target_log_rv"].to_numpy()))
            g = m.predict(val[self.features])
            s, score = _score(val, g)
            trials.append({"depth": depth, "best_iteration": int(m.get_best_iteration()), "scale": s, "val_qlike": score, "seconds": round(time.time() - t0, 1)})
            if best is None or score < best[0]:
                best = (score, m, s, depth)
        _, self.model_, self.scale_, self.depth_ = best
        return {"trials": trials, "chosen": {"depth": self.depth_, "best_iteration": int(self.model_.get_best_iteration())}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _floor(self.scale_ * np.exp(self.model_.predict(df[self.features])))


class ExtraTreesModel:
    """A random forest variant: many independent, extra-randomised decision trees whose forecasts are averaged."""
    name = "extratrees"
    kind = "model"

    def __init__(self, features: list[str], seed: int = 0, leaf_sizes: tuple[int, ...] = (100, 400), n_trees: int = 100,
                 max_features: float = 0.6, max_train_rows: int | None = 400_000, n_jobs: int | None = None):
        self.features, self.seed, self.leaf_sizes, self.n_trees = list(features), seed, leaf_sizes, n_trees
        self.max_features, self.max_train_rows = max_features, max_train_rows
        self.n_jobs = n_jobs or MODEL_CORES

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        from sklearn.ensemble import ExtraTreesRegressor
        tr = _sample_rows(train, self.max_train_rows, self.seed)
        trials, best = [], None
        for leaf in self.leaf_sizes:
            t0 = time.time()
            m = ExtraTreesRegressor(n_estimators=self.n_trees, min_samples_leaf=leaf, max_features=self.max_features,
                                    n_jobs=self.n_jobs, random_state=self.seed)
            m.fit(tr[self.features], tr["target_log_rv"].to_numpy())
            s, score = _score(val, m.predict(val[self.features]))
            trials.append({"min_samples_leaf": leaf, "train_rows_used": len(tr), "scale": s, "val_qlike": score, "seconds": round(time.time() - t0, 1)})
            if best is None or score < best[0]:
                best = (score, m, s, leaf)
        _, self.model_, self.scale_, self.leaf_ = best
        # Predict on one core: with several, trees are summed in varying order and the last digits change between runs.
        self.model_.set_params(n_jobs=1)
        return {"trials": trials, "chosen": {"min_samples_leaf": self.leaf_, "train_rows_used": len(tr)}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _floor(self.scale_ * np.exp(self.model_.predict(df[self.features])))


class MLPModel:
    """A small neural network (two hidden layers). Stops training when the tuning score stops improving."""
    name = "mlp"
    kind = "model"

    def __init__(self, features: list[str], seed: int = 0, hidden: tuple[int, ...] = (64, 32), alpha: float = 1e-4,
                 learning_rate: float = 1e-3, batch_size: int = 2048, max_epochs: int = 25, patience: int = 3,
                 max_train_rows: int | None = 200_000):
        self.features, self.seed, self.hidden, self.alpha = list(features), seed, hidden, alpha
        self.learning_rate, self.batch_size, self.max_epochs, self.patience = learning_rate, batch_size, max_epochs, patience
        self.max_train_rows = max_train_rows

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        import copy
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor

        t0 = time.time()
        tr = _sample_rows(train, self.max_train_rows, self.seed)
        self.prep_ = Pipeline([("winsorize", Winsorizer()), ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                               ("scale", StandardScaler())]).fit(tr[self.features])
        X, y = self.prep_.transform(tr[self.features]), tr["target_log_rv"].to_numpy()
        Xv, yv = self.prep_.transform(val[self.features]), val["target_log_rv"].to_numpy()
        net = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha, batch_size=min(self.batch_size, len(X)),
                           learning_rate_init=self.learning_rate, random_state=self.seed, shuffle=False)
        rng = np.random.default_rng(self.seed)
        best, best_epoch, waited, history = None, 0, 0, []
        for epoch in range(1, self.max_epochs + 1):
            order = rng.permutation(len(X))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                net.partial_fit(X[order], y[order])
            err = float(np.mean((net.predict(Xv) - yv) ** 2))
            history.append(round(err, 5))
            if best is None or err < best[0]:
                best, best_epoch, waited = (err, copy.deepcopy(net)), epoch, 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        self.net_ = best[1]
        self.scale_, score = _score(val, self.net_.predict(Xv))
        trial = {"hidden": list(self.hidden), "alpha": self.alpha, "best_epoch": best_epoch, "epochs_run": len(history),
                 "val_mse_by_epoch": history, "train_rows_used": len(tr), "scale": self.scale_, "val_qlike": score,
                 "seconds": round(time.time() - t0, 1)}
        return {"trials": [trial], "chosen": {"best_epoch": best_epoch, "train_rows_used": len(tr)}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return _floor(self.scale_ * np.exp(self.net_.predict(self.prep_.transform(df[self.features]))))


MODEL_NAMES = ("persistence", "har", "ridge", "lightgbm", "xgboost", "catboost", "extratrees", "mlp")
BASELINES = ("persistence", "har")


def make_model(name: str, features: list[str], seed: int = 0, quick: bool = False):
    if name == "persistence":
        return Persistence()
    if name == "har":
        return HAR()
    if name == "ridge":
        return RidgeModel(features)
    if name == "lightgbm":
        return LightGBMModel(features, seed=seed, leaves=(31,) if quick else (31, 127), max_rounds=300 if quick else 2000)
    if name == "xgboost":
        return XGBoostModel(features, seed=seed, depths=(6,) if quick else (6, 10), max_rounds=300 if quick else 2000)
    if name == "catboost":
        return CatBoostModel(features, seed=seed, max_rounds=300 if quick else 1500)
    if name == "extratrees":
        return ExtraTreesModel(features, seed=seed, leaf_sizes=(200,) if quick else (100, 400), n_trees=50 if quick else 100)
    if name == "mlp":
        return MLPModel(features, seed=seed, max_epochs=8 if quick else 25)
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


SPEC_KEYS = {
    "har": ["forms"], "ridge": ["alphas"],
    "lightgbm": ["leaves", "learning_rate", "max_rounds", "patience", "seed"],
    "xgboost": ["depths", "learning_rate", "max_rounds", "patience", "seed"],
    "catboost": ["depths", "learning_rate", "max_rounds", "patience", "seed"],
    "extratrees": ["leaf_sizes", "n_trees", "max_features", "max_train_rows", "seed"],
    "mlp": ["hidden", "alpha", "learning_rate", "batch_size", "max_epochs", "patience", "max_train_rows", "seed"],
}


def spec(name: str, features: list[str], seed: int = 0, quick: bool = False) -> dict:
    """The settings a model will be built with, for manifests and locks."""
    m = make_model(name, features, seed, quick)
    out = {"name": name, "kind": m.kind, **{k: getattr(m, k) for k in SPEC_KEYS.get(name, [])}}
    if name not in BASELINES:
        out["features"] = list(features)
    if name == "lightgbm":
        out["fixed_params"] = {k: v for k, v in m._params(0).items() if k not in ("num_leaves", "n_jobs", "random_state", "n_estimators", "learning_rate")}
    if name == "xgboost":
        out["fixed_params"] = {k: v for k, v in m._params(0).items() if k not in ("max_depth", "n_jobs", "random_state", "n_estimators", "learning_rate", "early_stopping_rounds")}
    return out

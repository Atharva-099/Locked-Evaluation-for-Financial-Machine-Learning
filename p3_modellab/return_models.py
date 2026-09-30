"""Baselines and learned models for one-month-ahead stock returns."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import MODEL_CORES, Winsorizer


RETURN_MODEL_NAMES = ("zero", "reversal", "ridge", "lightgbm")
RETURN_BASELINES = ("zero", "reversal")


def mean_monthly_rank_ic(y: np.ndarray, forecast: np.ndarray, months: np.ndarray) -> float:
    values = []
    data = pd.DataFrame({"y": y, "forecast": forecast, "month": months})
    for _, g in data.groupby("month"):
        if len(g) < 3 or g["forecast"].nunique() < 2 or g["y"].nunique() < 2:
            continue
        values.append(float(spearmanr(g["forecast"], g["y"]).statistic))
    return float(np.mean(values)) if values else float("nan")


def _selection_score(y: np.ndarray, forecast: np.ndarray, months: np.ndarray) -> tuple[float, float]:
    ic = mean_monthly_rank_ic(y, forecast, months)
    return (-np.inf if not np.isfinite(ic) else ic, -float(np.mean((y - forecast) ** 2)))


def _clip_target(train: pd.DataFrame) -> tuple[np.ndarray, tuple[float, float]]:
    y = train["target_ret"].to_numpy(dtype="float64")
    lo, hi = np.quantile(y, [0.001, 0.999])
    return np.clip(y, lo, hi), (float(lo), float(hi))


class ZeroReturn:
    name = "zero"
    kind = "baseline"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        return {"trials": [], "chosen": {}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df), dtype="float64")


class ReversalReturn:
    name = "reversal"
    kind = "baseline"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        return {"trials": [], "chosen": {"rule": "negative prediction-month return"}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return -df["ret_1m"].to_numpy(dtype="float64")


class ReturnRidge:
    name = "ridge"
    kind = "model"

    def __init__(self, features: list[str], alphas: tuple[float, ...] = (1e2, 1e4, 1e6, 1e8)):
        self.features, self.alphas = list(features), alphas

    def _pipe(self, alpha: float) -> Pipeline:
        return Pipeline([
            ("winsorize", Winsorizer()),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        ytr, bounds = _clip_target(train)
        yv = val["target_ret"].to_numpy(dtype="float64")
        months = val["yyyymm"].to_numpy()
        trials, best = [], None
        for alpha in self.alphas:
            t0 = time.time()
            pipe = self._pipe(alpha).fit(train[self.features], ytr)
            pred = pipe.predict(val[self.features])
            score = _selection_score(yv, pred, months)
            rec = {"alpha": alpha, "val_rank_ic": score[0], "val_mse": -score[1], "seconds": round(time.time() - t0, 3)}
            trials.append(rec)
            if best is None or score > best[0]:
                best = (score, pipe, alpha)
        _, self.pipe_, self.alpha_ = best
        self.target_clip_ = bounds
        return {"trials": trials, "chosen": {"alpha": self.alpha_, "target_clip": list(bounds)}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict(df[self.features]).astype("float64")


class ReturnLightGBM:
    name = "lightgbm"
    kind = "model"

    def __init__(self, features: list[str], seed: int = 0, leaves: tuple[int, ...] = (15, 63),
                 learning_rate: float = 0.05, max_rounds: int = 1200, patience: int = 75, n_jobs: int | None = None):
        self.features, self.seed, self.leaves = list(features), seed, leaves
        self.learning_rate, self.max_rounds, self.patience = learning_rate, max_rounds, patience
        self.n_jobs = n_jobs or MODEL_CORES

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict:
        import lightgbm as lgb

        ytr, bounds = _clip_target(train)
        yv = val["target_ret"].to_numpy(dtype="float64")
        months = val["yyyymm"].to_numpy()
        trials, best = [], None
        for leaves in self.leaves:
            t0 = time.time()
            model = lgb.LGBMRegressor(
                objective="regression", n_estimators=self.max_rounds, learning_rate=self.learning_rate,
                num_leaves=leaves, min_child_samples=200, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=self.seed,
                n_jobs=self.n_jobs, deterministic=True, force_row_wise=True, verbose=-1,
            )
            model.fit(
                train[self.features], ytr,
                eval_X=(val[self.features],), eval_y=(np.clip(yv, *bounds),),
                callbacks=[lgb.early_stopping(self.patience, verbose=False)],
            )
            pred = model.predict(val[self.features], num_iteration=model.best_iteration_)
            score = _selection_score(yv, pred, months)
            rec = {
                "num_leaves": leaves, "best_iteration": int(model.best_iteration_),
                "val_rank_ic": score[0], "val_mse": -score[1], "seconds": round(time.time() - t0, 2),
            }
            trials.append(rec)
            if best is None or score > best[0]:
                best = (score, model, leaves)
        _, self.model_, self.leaves_ = best
        self.target_clip_ = bounds
        return {"trials": trials, "chosen": {"num_leaves": self.leaves_, "best_iteration": int(self.model_.best_iteration_),
                                               "target_clip": list(bounds)}}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(df[self.features], num_iteration=self.model_.best_iteration_).astype("float64")


def make_return_model(name: str, features: list[str], seed: int = 0, quick: bool = False):
    if name == "zero":
        return ZeroReturn()
    if name == "reversal":
        return ReversalReturn()
    if name == "ridge":
        return ReturnRidge(features, alphas=(1e4, 1e6) if quick else (1e2, 1e4, 1e6, 1e8))
    if name == "lightgbm":
        return ReturnLightGBM(features, seed=seed, leaves=(31,) if quick else (15, 63), max_rounds=200 if quick else 1200)
    raise ValueError(f"unknown return model {name!r}; choose from {RETURN_MODEL_NAMES}")


def return_model_spec(name: str, features: list[str], seed: int = 0, quick: bool = False) -> dict:
    model = make_return_model(name, features, seed, quick)
    out = {"name": name, "kind": model.kind}
    for key in ("alphas", "leaves", "learning_rate", "max_rounds", "patience", "seed"):
        if hasattr(model, key):
            out[key] = getattr(model, key)
    if name not in RETURN_BASELINES:
        out["features"] = list(features)
    return out

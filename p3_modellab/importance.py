"""What does each model rely on? Permutation importance from the saved models.

For a sample of each period's test examples: score the model, then scramble one
input (or one group of inputs, scrambled together) and score again. The increase
in QLIKE is how much the model relies on that input. It measures reliance, not
cause, and related inputs share credit, which is why groups are also scrambled
together.

How inputs are scrambled:
- stock-level inputs: swapped among stocks in the same month;
- month-level inputs (the same value for every stock in a month, such as market
  volatility): swapped between months, because swapping within a month would
  change nothing and always show zero reliance.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .losses import qlike


def _shuffle_within_month(X: pd.DataFrame, cols: list[str], months: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    out = X.copy()
    pos = np.arange(len(X))
    perm = pos.copy()
    for m in np.unique(months):
        idx = pos[months == m]
        perm[idx] = rng.permutation(idx)
    out[cols] = X[cols].to_numpy()[perm]
    return out


def _is_month_level(X: pd.DataFrame, cols: list[str], months: np.ndarray) -> bool:
    return bool((X[cols].groupby(months).nunique(dropna=False) <= 1).all().all())


def _shuffle_between_months(X: pd.DataFrame, cols: list[str], months: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    out = X.copy()
    per_month = X[cols].groupby(months).first()
    swapped = per_month.set_axis(rng.permutation(per_month.index.to_numpy()))
    out[cols] = swapped.loc[months].to_numpy()
    return out


def scramble(X: pd.DataFrame, cols: list[str], months: np.ndarray, rng: np.random.Generator) -> tuple[pd.DataFrame, str]:
    if _is_month_level(X, cols, months):
        return _shuffle_between_months(X, cols, months, rng), "between months"
    return _shuffle_within_month(X, cols, months, rng), "within month"


def permutation_importance(run_dir: Path, panel_by_id: pd.DataFrame, pred: pd.DataFrame, features: list[str],
                           groups: dict[str, list[str]], models=None, n_rows: int = 20000, seed: int = 0) -> pd.DataFrame:
    models = models if models is not None else [m for m in pred["model"].unique() if m not in ("persistence", "har")]
    rows = []
    items = [("group", g, cols) for g, cols in groups.items()] + [("feature", f, [f]) for f in features]
    for fold in sorted(pred["fold"].unique()):
        ids = pred.loc[(pred["fold"] == fold) & (pred["model"] == pred["model"].iloc[0]), "row_id"].to_numpy()
        rng = np.random.default_rng(seed + int(fold))
        ids = rng.choice(ids, size=min(n_rows, len(ids)), replace=False)
        X = panel_by_id.loc[ids].reset_index()
        y, months = X["target_rv"].to_numpy(), X["yyyymm"].to_numpy()
        for name in models:
            path = Path(run_dir) / "models" / f"{name}_fold{int(fold):02d}.joblib"
            if not path.exists():
                continue
            model = joblib.load(path)
            base = qlike(y, model.predict(X)).mean()
            for kind, label, cols in items:
                Xs, how = scramble(X, cols, months, np.random.default_rng([seed, int(fold), zlib.crc32(label.encode())]))
                rows.append({"model": name, "fold": int(fold), "kind": kind, "input": label, "scrambled": how,
                             "qlike_increase": qlike(y, model.predict(Xs)).mean() - base})
    return pd.DataFrame(rows)


def summarise(imp: pd.DataFrame) -> pd.DataFrame:
    g = imp.groupby(["model", "kind", "input", "scrambled"])["qlike_increase"]
    out = g.agg(mean="mean", std="std", folds="count").reset_index()
    out["share_of_periods_positive"] = g.apply(lambda s: float((s > 0).mean())).to_numpy()
    return out.sort_values(["model", "kind", "mean"], ascending=[True, True, False])

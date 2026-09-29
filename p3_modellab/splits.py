"""Walk-forward splits that obey the timing contract.

For each refit, the fit time c is the first forecast time in the test period.
- train: label known by c minus the validation span
- validation: label known after that, but by c
- test: forecasts made in the test years
Exploratory runs never touch a row whose label becomes known on or after the
start of the reserved (locked) period. Locked runs test only on forecasts made
inside the reserved period.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .contract import assert_disjoint, assert_labels_known, assert_predictable


class SplitError(ValueError):
    pass


@dataclass
class Fold:
    fold: int
    test_years: list[int]
    fit_cutoff: pd.Timestamp
    train_end: pd.Timestamp
    train_ids: np.ndarray = field(repr=False)
    val_ids: np.ndarray = field(repr=False)
    test_ids: np.ndarray = field(repr=False)

    def summary(self) -> dict:
        def h(a: np.ndarray) -> str:
            return hashlib.sha256(np.sort(a).astype("int64").tobytes()).hexdigest()[:16]
        return {
            "fold": self.fold,
            "test_years": self.test_years,
            "fit_cutoff": str(self.fit_cutoff.date()),
            "train_label_known_by": str(self.train_end.date()),
            "n_train": int(self.train_ids.size),
            "n_val": int(self.val_ids.size),
            "n_test": int(self.test_ids.size),
            "train_ids_sha256": h(self.train_ids),
            "val_ids_sha256": h(self.val_ids),
            "test_ids_sha256": h(self.test_ids),
        }


def walk_forward(
    panel: pd.DataFrame,
    test_years: list[int],
    reserved_start: pd.Timestamp,
    run_type: str,
    validation_months: int = 24,
    refit_every: int = 1,
    min_train: int = 1000,
) -> list[Fold]:
    if run_type not in ("exploratory", "locked"):
        raise SplitError(f"unknown run_type {run_type!r}")
    reserved_start = pd.Timestamp(reserved_start)
    pt, la = panel["prediction_time"], panel["label_available_time"]

    if run_type == "exploratory":
        if any(pd.Timestamp(year=y, month=1, day=1) >= reserved_start for y in test_years):
            raise SplitError(f"exploratory test years {test_years} reach into the reserved period starting {reserved_start.date()}")
        usable = la < reserved_start
    else:
        if any(pd.Timestamp(year=y, month=12, day=31) < reserved_start for y in test_years):
            raise SplitError(f"locked test years {test_years} must lie inside the reserved period starting {reserved_start.date()}")
        usable = pd.Series(True, index=panel.index)

    years = sorted(test_years)
    groups = [years[i:i + refit_every] for i in range(0, len(years), refit_every)]
    folds = []
    for k, group in enumerate(groups):
        in_test = pt.dt.year.isin(group) & usable
        if run_type == "locked":
            in_test &= pt >= reserved_start
        if not in_test.any():
            raise SplitError(f"no test rows for years {group}")
        c = pt[in_test].min()
        train_end = c - pd.DateOffset(months=validation_months)
        tr = usable & (la <= train_end)
        va = usable & (la > train_end) & (la <= c)
        if tr.sum() < min_train:
            raise SplitError(f"fold for {group}: only {int(tr.sum())} training rows (minimum {min_train})")
        assert_labels_known(panel[tr], train_end, "train")
        assert_labels_known(panel[va], c, "validation")
        assert_predictable(panel[in_test], c)
        ids = panel["row_id"].to_numpy()
        f = Fold(k, group, c, train_end, ids[tr.to_numpy()], ids[va.to_numpy()], ids[in_test.to_numpy()])
        assert_disjoint(train=f.train_ids, val=f.val_ids, test=f.test_ids)
        folds.append(f)
    return folds

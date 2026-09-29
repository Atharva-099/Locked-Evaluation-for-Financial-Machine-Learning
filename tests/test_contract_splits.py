import numpy as np
import pandas as pd
import pytest

from p3_modellab.contract import ContractError, assert_disjoint, assert_labels_known, assert_predictable, validate_panel
from p3_modellab.splits import SplitError, walk_forward


def monthly_panel(start="2000-01", end="2010-12", firms=3):
    """One-month-ahead panel: forecast at month end, answer known at next month end."""
    months = pd.period_range(start, end, freq="M")
    rows = []
    for p in range(1, firms + 1):
        for m in months:
            pt = m.to_timestamp(how="end").normalize()
            le = (m + 1).to_timestamp(how="end").normalize()
            rows.append({"row_id": p * 10**6 + m.year * 100 + m.month, "permno": p, "month": m,
                         "prediction_time": pt, "feature_available_time": pt,
                         "label_end_time": le, "label_available_time": le})
    return pd.DataFrame(rows)


def test_valid_panel_passes():
    validate_panel(monthly_panel())


@pytest.mark.parametrize("col,shift,msg", [
    ("feature_available_time", pd.Timedelta(days=1), "feature_available_time > prediction_time"),
    ("label_end_time", -pd.Timedelta(days=40), "label_end_time <= prediction_time"),
    ("label_available_time", -pd.Timedelta(days=1), "label_available_time < label_end_time"),
])
def test_each_violation_is_caught(col, shift, msg):
    df = monthly_panel()
    df.loc[5, col] = df.loc[5, col] + shift
    with pytest.raises(ContractError, match=msg):
        validate_panel(df)


def test_missing_time_and_duplicate_ids_caught():
    df = monthly_panel()
    df.loc[0, "label_available_time"] = pd.NaT
    with pytest.raises(ContractError, match="missing values"):
        validate_panel(df)
    df = monthly_panel()
    df.loc[1, "row_id"] = df.loc[0, "row_id"]
    with pytest.raises(ContractError, match="not unique"):
        validate_panel(df)


def test_unavailable_label_rejected():
    df = monthly_panel()
    with pytest.raises(ContractError, match="labels not available"):
        assert_labels_known(df, pd.Timestamp("2005-01-31"), "train")


def test_label_guard_is_exact_at_the_boundary():
    df = monthly_panel().iloc[[0]].copy()
    known = df["label_available_time"].iloc[0]
    assert_labels_known(df, known, "train")
    with pytest.raises(ContractError):
        assert_labels_known(df, known - pd.Timedelta(days=1), "train")


def test_predictable_guard_is_exact_at_the_boundary():
    df = monthly_panel().iloc[[0]].copy()
    t = df["prediction_time"].iloc[0]
    assert_predictable(df, t)
    with pytest.raises(ContractError):
        assert_predictable(df, t + pd.Timedelta(days=1))


def test_disjoint_check_catches_overlap():
    assert_disjoint(a=np.array([1, 2]), b=np.array([3]))
    with pytest.raises(ContractError, match="1 rows are in both a and b"):
        assert_disjoint(a=np.array([1, 2]), b=np.array([2, 3]))


def months_of(df, ids):
    return sorted(df.set_index("row_id").loc[ids, "month"].unique())


def test_hand_worked_fold():
    """Test year 2005, 24-month validation span, reserved period from 2008.
    Fit time c = 2005-01-31 (first forecast in 2005). Train labels known by 2003-01-31,
    so train forecasts are 2000-01..2002-12. Validation forecasts 2003-01..2004-12."""
    df = monthly_panel()
    (f,) = walk_forward(df, [2005], pd.Timestamp("2008-01-01"), "exploratory", validation_months=24, min_train=1)
    assert f.fit_cutoff == pd.Timestamp("2005-01-31")
    assert months_of(df, f.train_ids)[0] == pd.Period("2000-01") and months_of(df, f.train_ids)[-1] == pd.Period("2002-12")
    assert months_of(df, f.val_ids)[0] == pd.Period("2003-01") and months_of(df, f.val_ids)[-1] == pd.Period("2004-12")
    assert months_of(df, f.test_ids) == list(pd.period_range("2005-01", "2005-12", freq="M"))
    assert f.train_ids.size == 36 * 3 and f.val_ids.size == 24 * 3 and f.test_ids.size == 12 * 3


def test_exploratory_never_sees_reserved_labels():
    df = monthly_panel()
    folds = walk_forward(df, [2005, 2006, 2007], pd.Timestamp("2008-01-01"), "exploratory", min_train=1)
    la = df.set_index("row_id")["label_available_time"]
    for f in folds:
        for ids in (f.train_ids, f.val_ids, f.test_ids):
            assert (la.loc[ids] < pd.Timestamp("2008-01-01")).all()
    # December 2007's answer arrives in January 2008 (reserved), so it is left out.
    assert pd.Period("2007-12") not in months_of(df, folds[-1].test_ids)
    assert pd.Period("2007-11") in months_of(df, folds[-1].test_ids)


def test_exploratory_refuses_reserved_years():
    with pytest.raises(SplitError, match="reserved"):
        walk_forward(monthly_panel(), [2007, 2008], pd.Timestamp("2008-01-01"), "exploratory", min_train=1)


def test_locked_tests_only_reserved_and_fits_on_known_labels():
    df = monthly_panel()
    (f,) = walk_forward(df, [2008], pd.Timestamp("2008-01-01"), "locked", min_train=1)
    assert f.fit_cutoff == pd.Timestamp("2008-01-31")
    assert months_of(df, f.test_ids)[0] == pd.Period("2008-01")
    la = df.set_index("row_id")["label_available_time"]
    assert (la.loc[f.val_ids] <= f.fit_cutoff).all()


def test_locked_with_mid_year_reserved_start_tests_only_after_it():
    df = monthly_panel()
    (f,) = walk_forward(df, [2008], pd.Timestamp("2008-07-01"), "locked", min_train=1)
    assert months_of(df, f.test_ids) == list(pd.period_range("2008-07", "2008-12", freq="M"))
    assert f.fit_cutoff == pd.Timestamp("2008-07-31")


def test_empty_panel_refused():
    df = monthly_panel().iloc[0:0]
    validate_panel(df)
    with pytest.raises(SplitError, match="no test rows"):
        walk_forward(df, [2005], pd.Timestamp("2008-01-01"), "exploratory", min_train=1)


def test_refit_every_groups_years_and_fit_time_precedes_all_tests():
    df = monthly_panel()
    folds = walk_forward(df, [2004, 2005, 2006, 2007], pd.Timestamp("2009-01-01"), "exploratory", refit_every=3, min_train=1)
    assert [f.test_years for f in folds] == [[2004, 2005, 2006], [2007]]
    pt = df.set_index("row_id")["prediction_time"]
    for f in folds:
        assert (pt.loc[f.test_ids] >= f.fit_cutoff).all()
        assert np.intersect1d(f.train_ids, f.test_ids).size == 0


def test_too_little_training_data_refused():
    with pytest.raises(SplitError, match="training rows"):
        walk_forward(monthly_panel(), [2002], pd.Timestamp("2009-01-01"), "exploratory", min_train=10_000)

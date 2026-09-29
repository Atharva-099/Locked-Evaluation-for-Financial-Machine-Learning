import numpy as np
import pandas as pd
import pytest

from p3_modellab.uncertainty import _block_indices, block_bootstrap, holm, monthly_means


def test_monthly_means_give_each_month_equal_weight():
    values = np.array([1.0, 3.0, 10.0, 10.0, 10.0, 10.0])
    months = np.array([202001, 202001, 202002, 202002, 202002, 202002])
    m = monthly_means(values, months)
    assert m.tolist() == [2.0, 10.0]
    assert m.mean() == 6.0  # not the row average 7.33


def test_block_indices_are_whole_consecutive_blocks():
    rng = np.random.default_rng(0)
    idx = _block_indices(10, 4, rng)
    assert len(idx) == 10
    for start in range(0, 8, 4):
        b = idx[start:start + 4]
        assert all((b[i + 1] - b[i]) % 10 == 1 for i in range(3))


def test_interval_covers_known_mean_about_95_percent_of_the_time():
    rng = np.random.default_rng(1)
    hits = 0
    for s in range(200):
        x = rng.normal(0.5, 1.0, 120)
        iv = block_bootstrap(x, block=1, n_boot=400, seed=s)
        hits += iv.lo <= 0.5 <= iv.hi
    assert 0.88 <= hits / 200 <= 0.99


def test_blocks_widen_intervals_for_related_months():
    """Strongly related months carry less information; block resampling must show that."""
    rng = np.random.default_rng(2)
    e = rng.normal(0, 1, 300)
    x = np.zeros(300)
    for t in range(1, 300):
        x[t] = 0.9 * x[t - 1] + e[t]
    narrow = block_bootstrap(x, block=1, seed=0)
    wide = block_bootstrap(x, block=24, seed=0)
    assert (wide.hi - wide.lo) > 1.8 * (narrow.hi - narrow.lo)


def test_p_value_small_for_clear_effect_and_large_for_none():
    rng = np.random.default_rng(3)
    assert block_bootstrap(rng.normal(1.0, 1.0, 200), seed=0).p_value < 0.01
    assert block_bootstrap(rng.normal(0.0, 1.0, 200), seed=0).p_value > 0.05


def test_reproducible_with_seed_and_handles_empty():
    x = np.arange(50.0)
    assert block_bootstrap(x, seed=5).as_dict() == block_bootstrap(x, seed=5).as_dict()
    assert np.isnan(block_bootstrap(np.array([np.nan])).estimate)


def test_holm_matches_hand_calculation():
    # sorted p: 0.01, 0.02, 0.04 -> 3*0.01=0.03, 2*0.02=0.04, 1*0.04=0.04 (kept non-decreasing)
    assert holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])

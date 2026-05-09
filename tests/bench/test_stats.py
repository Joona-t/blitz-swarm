"""Unit tests for bench/stats.py."""

from __future__ import annotations

import math

import pytest

from bench.stats import (
    bootstrap_ci,
    cohens_d,
    paired_t_test,
    welch_t_test,
)


def test_paired_t_test_zero_diffs():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [1.0, 2.0, 3.0, 4.0]
    t, p, df = paired_t_test(a, b)
    assert t == 0.0
    assert p == 1.0
    assert df == 3


def test_paired_t_test_positive_difference():
    a = [10.0, 11.0, 9.5, 12.0, 10.5]
    b = [8.0, 9.5, 7.0, 10.5, 8.0]
    # Diffs: 2.0, 1.5, 2.5, 1.5, 2.5  (positive, with variance)
    t, p, df = paired_t_test(a, b)
    assert t > 0  # a > b consistently
    assert p < 0.05  # significant
    assert df == 4


def test_paired_t_test_unequal_lengths_raises():
    with pytest.raises(ValueError):
        paired_t_test([1, 2, 3], [1, 2])


def test_paired_t_test_too_few_samples():
    with pytest.raises(ValueError):
        paired_t_test([1.0], [2.0])


def test_cohens_d_zero_when_no_difference():
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    assert cohens_d(a, b) == 0.0


def test_cohens_d_positive():
    a = [10.0, 11.0, 12.0, 13.0, 14.0]
    b = [8.5, 9.5, 10.0, 10.5, 12.0]
    # Diffs: 1.5, 1.5, 2.0, 2.5, 2.0  (positive, non-zero variance)
    d = cohens_d(a, b)
    assert d > 0  # paired diff strictly positive on average


def test_bootstrap_ci_single_value():
    lo, hi = bootstrap_ci([5.0])
    assert lo == hi == 5.0


def test_bootstrap_ci_contains_mean():
    values = [10.0, 11.0, 9.0, 12.0, 8.0, 11.5, 10.5]
    lo, hi = bootstrap_ci(values, ci=95.0, n_resamples=2000, seed=42)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi


def test_bootstrap_ci_deterministic_with_seed():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    a_lo, a_hi = bootstrap_ci(values, n_resamples=500, seed=42)
    b_lo, b_hi = bootstrap_ci(values, n_resamples=500, seed=42)
    assert math.isclose(a_lo, b_lo)
    assert math.isclose(a_hi, b_hi)


def test_welch_t_test_equal_means():
    a = [10.0, 11.0, 9.0, 10.5, 9.5]
    b = [10.0, 11.0, 9.0, 10.5, 9.5]
    t, p = welch_t_test(a, b)
    assert abs(t) < 0.001
    assert p > 0.99


def test_welch_t_test_different_means():
    a = [10.0, 11.0, 9.0, 10.5, 9.5]
    b = [5.0, 6.0, 4.0, 5.5, 4.5]
    t, p = welch_t_test(a, b)
    assert abs(t) > 5.0
    assert p < 0.01

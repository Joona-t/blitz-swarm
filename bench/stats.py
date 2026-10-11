"""Statistical analysis for bench results.

Provides:
    paired_t_test(a, b)              -> (t_stat, p_value, df)
    cohens_d(a, b)                   -> float (paired d_z)
    bootstrap_ci(values, *, ci=95, n_resamples=10000, seed=42) -> (lo, hi)
    welch_t_test(a, b)               -> (t_stat, p_value)
    mann_whitney_u(a, b)             -> (u_stat, p_value)

All functions accept Python lists of floats. They use scipy.stats when
available; otherwise fall back to a pure-stdlib implementation. The
fallback is deterministic and seed-able.

scipy-free p-values (bench-03): the Student-t survival function is
computed exactly through the regularized incomplete beta function for
df < 30; the normal approximation is used only for df >= 30, where it is
within ~1e-3 of the t distribution. The old code applied the normal
approximation at every df, which under-states p for small samples (the
only sample sizes the bench ever sees).

This module never imports scipy at module load; it lazy-imports inside each
function so the bench package can be imported on systems without scipy.
"""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable


def _ensure_lists(a: Iterable[float], b: Iterable[float]) -> tuple[list[float], list[float]]:
    a_list = list(a)
    b_list = list(b)
    if len(a_list) != len(b_list):
        raise ValueError(f"paired tests require equal-length samples: {len(a_list)} vs {len(b_list)}")
    if len(a_list) < 2:
        raise ValueError("paired tests require at least 2 samples")
    return a_list, b_list


def paired_t_test(a: Iterable[float], b: Iterable[float]) -> tuple[float, float, int]:
    """Paired t-test. Returns (t_stat, two_sided_p, df)."""
    a_list, b_list = _ensure_lists(a, b)
    diffs = [ai - bi for ai, bi in zip(a_list, b_list)]
    n = len(diffs)
    mean_d = statistics.mean(diffs)
    if all(d == 0 for d in diffs):
        return 0.0, 1.0, n - 1
    sd_d = statistics.stdev(diffs)
    if sd_d == 0:
        return 0.0, 1.0, n - 1
    t_stat = mean_d / (sd_d / math.sqrt(n))
    df = n - 1

    try:
        from scipy import stats as scipy_stats

        p_value = float(scipy_stats.t.sf(abs(t_stat), df) * 2)
    except ImportError:
        p_value = t_two_sided_p(t_stat, df)
    return t_stat, p_value, df


def cohens_d(a: Iterable[float], b: Iterable[float]) -> float:
    """Cohen's d_z for paired data."""
    a_list, b_list = _ensure_lists(a, b)
    diffs = [ai - bi for ai, bi in zip(a_list, b_list)]
    if len(diffs) < 2:
        return 0.0
    sd_d = statistics.stdev(diffs)
    if sd_d == 0:
        return 0.0
    return statistics.mean(diffs) / sd_d


def bootstrap_ci(
    values: Iterable[float],
    *,
    ci: float = 95.0,
    n_resamples: int = 10000,
    seed: int = 42,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean (or median)."""
    vals = list(values)
    if len(vals) < 2:
        return (vals[0], vals[0]) if vals else (0.0, 0.0)
    rng = random.Random(seed)
    stat_fn = statistics.mean if statistic == "mean" else statistics.median
    samples: list[float] = []
    n = len(vals)
    for _ in range(n_resamples):
        resample = [vals[rng.randrange(n)] for _ in range(n)]
        samples.append(stat_fn(resample))
    samples.sort()
    alpha = (100.0 - ci) / 200.0
    lo_idx = int(alpha * n_resamples)
    hi_idx = int((1.0 - alpha) * n_resamples) - 1
    return samples[lo_idx], samples[hi_idx]


def welch_t_test(a: Iterable[float], b: Iterable[float]) -> tuple[float, float]:
    """Welch's t-test for unequal variances. Returns (t_stat, two_sided_p)."""
    a_list = list(a)
    b_list = list(b)
    if len(a_list) < 2 or len(b_list) < 2:
        raise ValueError("Welch's t-test requires at least 2 samples per group")
    m1, m2 = statistics.mean(a_list), statistics.mean(b_list)
    v1, v2 = statistics.variance(a_list), statistics.variance(b_list)
    n1, n2 = len(a_list), len(b_list)
    if v1 == 0 and v2 == 0:
        return 0.0, 1.0
    se = math.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se if se else 0.0
    # Welch-Satterthwaite df
    if v1 == 0 or v2 == 0:
        df = n1 + n2 - 2
    else:
        df = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )
    try:
        from scipy import stats as scipy_stats

        p_value = float(scipy_stats.t.sf(abs(t_stat), df) * 2)
    except ImportError:
        p_value = t_two_sided_p(t_stat, df)
    return t_stat, p_value


def mann_whitney_u(a: Iterable[float], b: Iterable[float]) -> tuple[float, float]:
    """Mann-Whitney U test for two independent samples (skewed distributions)."""
    try:
        from scipy import stats as scipy_stats

        result = scipy_stats.mannwhitneyu(list(a), list(b), alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except ImportError as e:
        raise NotImplementedError("mann_whitney_u requires scipy") from e


# Below this df the normal approximation is NOT used (bench-03).
NORMAL_APPROX_MIN_DF = 30


def t_two_sided_p(t_stat: float, df: float) -> float:
    """Two-sided p-value for Student's t without scipy.

    df < NORMAL_APPROX_MIN_DF -> exact t CDF via the regularized incomplete
    beta function I_x(df/2, 1/2) with x = df / (df + t^2).
    df >= NORMAL_APPROX_MIN_DF -> normal approximation (error < ~1e-3).
    """
    if df <= 0 or math.isnan(t_stat):
        return 1.0
    if math.isinf(t_stat):
        return 0.0
    if df >= NORMAL_APPROX_MIN_DF:
        return 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    x = df / (df + t_stat * t_stat)
    p = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    return min(1.0, max(0.0, p))


def _normal_cdf(x: float) -> float:
    """Standard-normal CDF — scipy-free fallback (df >= 30 only)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b) via Lentz's continued fraction (Numerical Recipes betacf)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(ln_beta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, *, max_iter: int = 300, eps: float = 3e-14) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) >= tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= tiny else tiny)
        c = 1.0 + aa / (c if abs(c) >= tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) >= tiny else tiny)
        c = 1.0 + aa / (c if abs(c) >= tiny else tiny)
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h

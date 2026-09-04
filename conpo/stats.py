"""Evaluation statistics (locked decision (d), 2026-09-02): mean/std over
seeds, and a paired bootstrap over prompts for every arm difference named in
conpo.arms.HYPOTHESES.

Minimal recipe (always computed, by scripts/analyze.py): mean_std over each
arm's per-seed headline metric, plus paired_bootstrap_diff over per-prompt
verdicts for every HYPOTHESES row. wilson_interval, mcnemar and holm are
optional extras, only run when named (scripts/analyze.py's --extras flag):
wilson_interval reports one arm's own prompt-accuracy confidence band,
mcnemar tests the headline pair's discordant prompts, holm corrects the
per-hypothesis p-values for testing every HYPOTHESES row at once.
"""

import math
import random
import statistics
from collections.abc import Sequence
from typing import NamedTuple


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1) of `values`.

    One value per seed for the same arm/metric -- the per-benchmark table's
    mean +/- std columns. len(values) < 2 returns std 0.0 (sample std is
    undefined below n=2); an empty `values` returns (0.0, 0.0).
    """
    vals = [float(v) for v in values]
    if not vals:
        return 0.0, 0.0
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return mean, 0.0
    return mean, statistics.stdev(vals)


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0, 1]) of a pre-sorted sequence."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = p * (n - 1)
    lower = int(idx)
    upper = min(lower + 1, n - 1)
    frac = idx - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def paired_bootstrap_diff(
    a: Sequence[bool], b: Sequence[bool], n_resamples: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """95 percent paired bootstrap interval for mean(a) - mean(b).

    `a` and `b` are the same prompt set's per-prompt verdicts for the
    treatment and control arm respectively: index i must be the same prompt
    in both. Each resample draws len(a) prompt indices with replacement
    once and applies that SAME index set to both `a` and `b` (paired, not
    two independent resamples), then records mean(a[idx]) - mean(b[idx]).

    Returns (point_diff, lo, hi): point_diff is mean(a) - mean(b) on the
    full (unresampled) data; (lo, hi) is the 2.5th/97.5th percentile of the
    resampled diffs. Raises ValueError if len(a) != len(b).
    """
    if len(a) != len(b):
        raise ValueError(f"paired arrays must have equal length, got {len(a)} and {len(b)}")
    n = len(a)
    if n == 0:
        return 0.0, 0.0, 0.0
    a_vals = [float(x) for x in a]
    b_vals = [float(x) for x in b]
    point_diff = sum(a_vals) / n - sum(b_vals) / n
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(a_vals[i] for i in idx) / n - sum(b_vals[i] for i in idx) / n)
    diffs.sort()
    return point_diff, _percentile(diffs, 0.025), _percentile(diffs, 0.975)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (default z=1.96,
    approximately 95 percent).

    Optional extra: one arm's own prompt-accuracy confidence band, not a
    comparison between arms. Raises ValueError if n <= 0 or k not in [0, n].
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")
    phat = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = phat + z2 / (2 * n)
    adj = z * math.sqrt(phat * (1 - phat) / n + z2 / (4 * n * n))
    return (center - adj) / denom, (center + adj) / denom


class McNemarResult(NamedTuple):
    """n01: control-only-passed count (a False, b True); n10: treatment-only
    (a True, b False); p_value: exact two-sided binomial test over the
    discordant pairs, p=0.5."""

    n01: int
    n10: int
    p_value: float


def mcnemar(a: Sequence[bool], b: Sequence[bool]) -> McNemarResult:
    """Exact two-sided McNemar test on paired binary outcomes, no scipy: the
    binomial test on the discordant pairs (n10 vs n01), p=0.5.

    Optional extra: tests whether the headline arm pair's discordant
    prompts favour one arm over the other. Raises ValueError if
    len(a) != len(b).
    """
    if len(a) != len(b):
        raise ValueError(f"paired arrays must have equal length, got {len(a)} and {len(b)}")
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n = n10 + n01
    if n == 0:
        return McNemarResult(n01=0, n10=0, p_value=1.0)
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
    return McNemarResult(n01=n01, n10=n10, p_value=min(1.0, 2 * tail))


def verdict_from_ci(lo: float, hi: float, expected_sign: str) -> str:
    """"supports" when the interval excludes zero on the expected side,
    "null (opposite sign)" when it excludes zero on the other side, else
    "null" (the interval includes zero).

    Same logic and return strings as scripts/dashboard.py's
    ``_verdict_from_ci`` (design choice, builder, 2026-09-04: the shared
    home for both callers -- scripts/analyze.py's hypothesis_table treats
    "no matched seed pairs" as a fourth state, "pending", on top of this).
    """
    if lo > 0:
        return "supports" if expected_sign == ">" else "null (opposite sign)"
    if hi < 0:
        return "supports" if expected_sign == "<" else "null (opposite sign)"
    return "null"


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, returned in the same
    order as `pvalues` (not sorted).

    Optional extra: corrects for testing every conpo.arms.HYPOTHESES row at
    once instead of each at an unadjusted alpha.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        factor = m - rank
        running_max = max(running_max, factor * pvalues[i])
        adjusted[i] = min(1.0, running_max)
    return adjusted

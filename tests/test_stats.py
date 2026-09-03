"""Tests for conpo/stats.py on hand-built vectors."""

from conpo import stats


def test_mean_std_multiple_values() -> None:
    mean, std = stats.mean_std([1, 2, 3])
    assert mean == 2.0
    assert std == 1.0  # sample stdev, ddof=1


def test_mean_std_single_value_has_zero_std() -> None:
    assert stats.mean_std([5.0]) == (5.0, 0.0)


def test_mean_std_empty() -> None:
    assert stats.mean_std([]) == (0.0, 0.0)


def test_paired_bootstrap_diff_identical_arms_contains_zero() -> None:
    a = [True, False, True, True, False, True, False, False, True, True]
    diff, lo, hi = stats.paired_bootstrap_diff(a, a)
    assert diff == 0.0
    assert lo <= 0.0 <= hi


def test_paired_bootstrap_diff_clear_improvement_excludes_zero() -> None:
    a = [True] * 45 + [False] * 5  # 90% pass
    b = [False] * 45 + [True] * 5  # 10% pass
    diff, lo, hi = stats.paired_bootstrap_diff(a, b)
    assert diff == 0.8
    assert lo > 0.0  # interval excludes 0 in the expected (a > b) direction


def test_paired_bootstrap_diff_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        stats.paired_bootstrap_diff([True, False], [True])


def test_wilson_interval_k90_n300_within_hand_computed_range() -> None:
    lo, hi = stats.wilson_interval(90, 300)
    # hand-computed via the Wilson score formula: phat=0.3, z=1.96
    assert 0.245 < lo < 0.256
    assert 0.349 < hi < 0.359


def test_mcnemar_ten_vs_two_discordant() -> None:
    a = [True] * 10 + [False] * 2 + [True] * 5  # 5 concordant passes, irrelevant to the count
    b = [False] * 10 + [True] * 2 + [True] * 5
    result = stats.mcnemar(a, b)
    assert result.n10 == 10
    assert result.n01 == 2
    # hand-computed: 2 * sum_{i=0..2} C(12, i) * 0.5**12 = 2 * 79 / 4096
    assert abs(result.p_value - 2 * 79 / 4096) < 1e-9


def test_holm_ordering() -> None:
    adjusted = stats.holm([0.01, 0.02, 0.03, 0.5])
    assert adjusted == [0.04, 0.06, 0.06, 0.5]
    # every adjusted p-value is at least its raw p-value
    for raw, adj in zip([0.01, 0.02, 0.03, 0.5], adjusted):
        assert adj >= raw
    # step-down monotonicity: sorted by raw p-value, adjusted never decreases
    pairs = sorted(zip([0.01, 0.02, 0.03, 0.5], adjusted))
    adjusted_in_rank_order = [adj for _, adj in pairs]
    assert adjusted_in_rank_order == sorted(adjusted_in_rank_order)

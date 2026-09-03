"""Tests for conpo.compositor: pairwise family co-violation statistics."""

import random
from pathlib import Path

import pytest

from conpo import compositor
from conpo.compositor import PairStats
from conpo.verifiers import Constraint


def _c(iid: str) -> Constraint:
    return Constraint(iid, {})


# ---------------------------------------------------------------------------
# phi
# ---------------------------------------------------------------------------


def test_phi_hand_computed_positive_correlation() -> None:
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_b"))
    n11, n10, n01, n00 = 8, 1, 1, 8
    vectors = (
        ((False, False),) * n11
        + ((False, True),) * n10
        + ((True, False),) * n01
        + ((True, True),) * n00
    )
    stats.update(constraints, vectors)

    # Independently hand-computed for n11=8, n10=1, n01=1, n00=8:
    # (8*8 - 1*1) / sqrt(9*9*9*9) = 63 / 81 = 7/9.
    expected = 7 / 9
    assert stats.phi("fam_a", "fam_b") == pytest.approx(expected)
    assert stats.phi("fam_b", "fam_a") == pytest.approx(expected)  # unordered key
    assert stats.count("fam_a", "fam_b") == n11 + n10 + n01 + n00


def test_phi_zero_margin_returns_zero() -> None:
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_b"))
    # fam_a fails on every sample: the (n01+n00) margin is 0.
    vectors = ((False, False), (False, True), (False, False), (False, True))
    stats.update(constraints, vectors)
    assert stats.phi("fam_a", "fam_b") == 0.0


def test_phi_unseen_pair_is_zero() -> None:
    stats = PairStats()
    assert stats.phi("nope_a", "nope_b") == 0.0
    assert stats.count("nope_a", "nope_b") == 0


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


def test_update_skips_same_family_index_pairs() -> None:
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_a"))
    stats.update(constraints, ((False, True),))
    assert stats.count("fam_a", "fam_a") == 0


def test_update_accumulates_across_calls() -> None:
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_b"))
    stats.update(constraints, ((True, True),))
    stats.update(constraints, ((False, False),))
    assert stats.count("fam_a", "fam_b") == 2


def test_update_only_scores_distinct_family_pairs_on_a_larger_prompt() -> None:
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_b"), _c("fam_a"))
    # index pairs: (0,1)=a/b, (0,2)=a/a skipped, (1,2)=b/a -> same key as a/b
    stats.update(constraints, ((False, False, True),))
    assert stats.count("fam_a", "fam_b") == 2


def test_update_bins_correctly_when_constraint_order_disagrees_with_family_key_order() -> None:
    # fam_z at index 0, fam_a at index 1: the pair key sorts to ("fam_a", "fam_z"),
    # disagreeing with constraint order, so `first_is_i` is False and update()
    # must take the `else` swap branch (fail_first, fail_second = fail_j, fail_i).
    stats = PairStats()
    constraints = (_c("fam_z"), _c("fam_a"))
    # vector = (pass/fail fam_z, pass/fail fam_a); n10/n01 are asymmetric (3 vs 1)
    # so a swap bug (mixing up which count lands in which bin) is distinguishable.
    vectors = ((False, False),) * 8 + ((True, False),) * 3 + ((False, True),) * 1 + ((True, True),) * 8
    stats.update(constraints, vectors)

    row = stats._counts[("fam_a", "fam_z")]
    # first="fam_a" (index 1), second="fam_z" (index 0):
    # (True, False) -> fam_z passes, fam_a fails -> first fails, second passes -> n10.
    # (False, True) -> fam_z fails, fam_a passes -> first passes, second fails -> n01.
    assert row == {"n11": 8, "n10": 3, "n01": 1, "n00": 8}
    assert stats.count("fam_a", "fam_z") == 20


# ---------------------------------------------------------------------------
# sample_family_set
# ---------------------------------------------------------------------------


def test_sample_family_set_respects_phi_min_and_min_count() -> None:
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_b"), _c("fam_c"))
    strong = ((False, False),) * 15 + ((True, True),) * 15  # phi = 1.0, count = 30
    stats.update((constraints[0], constraints[1]), strong)
    stats.update((constraints[0], constraints[2]), ((False, False), (True, True)))  # phi = 1.0, count = 2

    rng = random.Random(0)
    result = compositor.sample_family_set(constraints, rng, stats, size=2, phi_min=0.5, min_count=20, eta=0.2)
    assert result == (0, 1)


def test_sample_family_set_empty_when_nothing_qualifies() -> None:
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_b"))
    rng = random.Random(0)
    assert compositor.sample_family_set(constraints, rng, stats, size=2) == ()


def test_sample_family_set_empty_below_phi_min() -> None:
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_b"))
    # phi = 0.0 (no correlation): n11=n00=n10=n01=10
    vectors = ((False, False),) * 10 + ((False, True),) * 10 + ((True, False),) * 10 + ((True, True),) * 10
    stats.update(constraints, vectors)
    rng = random.Random(0)
    assert compositor.sample_family_set(constraints, rng, stats, size=2, phi_min=0.1, min_count=1) == ()


def test_sample_family_set_deterministic_under_seeded_rng() -> None:
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_b"), _c("fam_c"), _c("fam_d"))
    strong = ((False, False),) * 12 + ((True, True),) * 12
    stats.update((constraints[0], constraints[1]), strong)
    stats.update((constraints[2], constraints[3]), strong)

    result_1 = compositor.sample_family_set(constraints, random.Random(42), stats, size=2, min_count=20)
    result_2 = compositor.sample_family_set(constraints, random.Random(42), stats, size=2, min_count=20)
    assert result_1 == result_2
    assert result_1 in {(0, 1), (2, 3)}


def test_sample_family_set_does_not_overflow_on_a_small_eta() -> None:
    # phi/eta = 1.0/0.001 = 1000: math.exp(1000) alone overflows; the
    # peak-subtraction guard (mirroring selection.mu) must keep this callable.
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_b"))
    stats.update(constraints, ((False, False),) * 15 + ((True, True),) * 15)  # phi = 1.0, count = 30
    rng = random.Random(0)
    result = compositor.sample_family_set(constraints, rng, stats, size=2, phi_min=0.5, min_count=20, eta=0.001)
    assert result == (0, 1)


def test_sample_family_set_excludes_no_operator_families() -> None:
    stats = PairStats()
    constraints = (_c("language:response_language"), _c("fam_b"), _c("fam_c"))
    strong = ((False, False),) * 15 + ((True, True),) * 15  # phi = 1.0, count = 30 for both pairs
    stats.update((constraints[0], constraints[1]), strong)  # NO_OPERATOR pair: must be excluded
    stats.update((constraints[1], constraints[2]), strong)  # only eligible pair
    rng = random.Random(0)
    result = compositor.sample_family_set(constraints, rng, stats, size=2, phi_min=0.5, min_count=20)
    assert result == (1, 2)


def test_sample_family_set_rejects_sizes_other_than_two() -> None:
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_b"), _c("fam_c"))
    with pytest.raises(NotImplementedError):
        compositor.sample_family_set(constraints, random.Random(0), stats, size=3)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: Path) -> None:
    stats = PairStats()
    constraints = (_c("fam_a"), _c("fam_b"))
    stats.update(constraints, ((False, False), (False, True), (True, True)))
    path = tmp_path / "pair_stats.json"
    stats.save(path)

    loaded = PairStats()
    loaded.load(path)
    assert loaded.count("fam_a", "fam_b") == stats.count("fam_a", "fam_b")
    assert loaded.phi("fam_a", "fam_b") == pytest.approx(stats.phi("fam_a", "fam_b"))


def test_load_replaces_existing_contents(tmp_path: Path) -> None:
    saved = PairStats()
    saved.update((_c("fam_a"), _c("fam_b")), ((False, False),))
    path = tmp_path / "pair_stats.json"
    saved.save(path)

    stats = PairStats()
    stats.update((_c("fam_x"), _c("fam_y")), ((False, False),) * 5)
    stats.load(path)
    assert stats.count("fam_x", "fam_y") == 0
    assert stats.count("fam_a", "fam_b") == 1


# ---------------------------------------------------------------------------
# top_pairs
# ---------------------------------------------------------------------------


def test_top_pairs_orders_by_phi_descending() -> None:
    stats = PairStats()
    a, b, c = _c("fam_a"), _c("fam_b"), _c("fam_c")
    stats.update((a, b), ((False, False),) * 10 + ((True, True),) * 10)  # phi = 1.0
    stats.update((a, c), ((False, True),) * 10 + ((True, False),) * 10)  # phi = -1.0

    top = stats.top_pairs(2)
    assert len(top) == 2
    assert top[0][2] >= top[1][2]
    assert top[0][:2] in {("fam_a", "fam_b"), ("fam_b", "fam_a")}


def test_top_pairs_caps_at_k() -> None:
    stats = PairStats()
    stats.update((_c("fam_a"), _c("fam_b")), ((False, False),))
    stats.update((_c("fam_c"), _c("fam_d")), ((False, False),))
    assert len(stats.top_pairs(1)) == 1

"""Tests for conpo.selection: active-selection statistics, mu(c | x), grad_cosine, probing."""

import random

import pytest
import torch

from conpo import config
from conpo.operators.registry import NO_OPERATOR
from conpo.selection import FamilyStats, grad_cosine, mu, next_probe_family, probe_due, sample_family

FAMILY_A = "family_a"
FAMILY_B = "family_b"
NO_OP_FAMILY = next(iter(NO_OPERATOR))


# --- FamilyStats: EMA arithmetic, including degenerate groups --------------


def test_update_pass_rates_ema_over_two_updates_including_degenerate_groups() -> None:
    stats = FamilyStats()
    decay = config.EMA_DECAY_P_C

    stats.update_pass_rates({FAMILY_A: [True, True, True]})  # degenerate: rate 1.0
    assert stats.p_c(FAMILY_A) == 1.0  # first observation sets p_c directly

    stats.update_pass_rates({FAMILY_A: [False, False]})  # degenerate: rate 0.0
    expected = decay * 1.0 + (1 - decay) * 0.0
    assert stats.p_c(FAMILY_A) == pytest.approx(expected)
    assert stats.p_c(FAMILY_A) != 0.0  # the degenerate group was folded in, not skipped


def test_record_probe_ema_over_two_updates() -> None:
    stats = FamilyStats()
    decay = config.EMA_DECAY_T_C
    assert stats.t_c(FAMILY_A) == 0.0  # T_c starts at 0 until the first probe

    stats.record_probe(FAMILY_A, cosine=0.6, update=10)
    after_first = decay * 0.0 + (1 - decay) * 0.6
    assert stats.t_c(FAMILY_A) == pytest.approx(after_first)

    stats.record_probe(FAMILY_A, cosine=-0.2, update=20)
    after_second = decay * after_first + (1 - decay) * -0.2
    assert stats.t_c(FAMILY_A) == pytest.approx(after_second)
    assert stats.last_probe_update(FAMILY_A) == 20


# --- U(c): maximal at p = 0.5, zero at p in {0, 1} --------------------------


def test_u_is_maximal_at_half_and_zero_at_extremes() -> None:
    stats = FamilyStats()
    stats.update_pass_rates({"half": [True, False]})
    stats.update_pass_rates({"zero": [False, False]})
    stats.update_pass_rates({"one": [True, True]})

    assert stats.U("half") == pytest.approx(1.0)
    assert stats.U("zero") == 0.0
    assert stats.U("one") == 0.0
    assert stats.U("half") > stats.U("zero")


# --- C(c): decreases with n_c ------------------------------------------------


def test_c_decreases_with_admissions() -> None:
    stats = FamilyStats()
    before = stats.C(FAMILY_A)
    for _ in range(5):
        stats.record_admission(FAMILY_A)
    after = stats.C(FAMILY_A)
    assert after < before
    assert stats.n_c(FAMILY_A) == 5
    assert stats.C(FAMILY_A) == pytest.approx(1.0 / (5 + 1) ** 0.5)


# --- mu(c | x): sums to 1, floor respected, excludes NO_OPERATOR -----------


def test_mu_sums_to_one_and_respects_the_exploration_floor() -> None:
    stats = FamilyStats()
    stats.update_pass_rates({FAMILY_A: [True]})
    stats.update_pass_rates({FAMILY_B: [False]})
    families = [FAMILY_A, FAMILY_B, NO_OP_FAMILY]

    probabilities = mu(stats, families)

    assert NO_OP_FAMILY not in probabilities
    assert sum(probabilities.values()) == pytest.approx(1.0)
    floor = config.EPS_A / 2  # two eligible families
    for value in probabilities.values():
        assert value >= floor - 1e-12


def test_mu_dominant_family_wins_at_small_eta(monkeypatch: pytest.MonkeyPatch) -> None:
    stats = FamilyStats()
    stats.update_pass_rates({FAMILY_A: [True, False]})  # p = 0.5 -> U = 1.0, fresh -> C = 1.0
    for _ in range(10_000):
        stats.record_admission(FAMILY_B)  # C ~ 0
    stats.update_pass_rates({FAMILY_B: [True, True]})  # p = 1.0 -> U = 0.0
    monkeypatch.setattr(config, "ETA", 0.01)

    probabilities = mu(stats, [FAMILY_A, FAMILY_B])

    assert probabilities[FAMILY_A] > 0.9
    assert probabilities[FAMILY_B] < 0.1


def test_sample_family_never_draws_the_no_operator_family() -> None:
    stats = FamilyStats()
    rng = random.Random(0)
    drawn = {sample_family(stats, [FAMILY_A, NO_OP_FAMILY], rng) for _ in range(200)}
    assert drawn == {FAMILY_A}


# --- grad_cosine -------------------------------------------------------------


def test_grad_cosine_identical_is_one_orthogonal_is_zero() -> None:
    grad = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    assert grad_cosine(grad, [g.clone() for g in grad]) == pytest.approx(1.0)

    a = [torch.tensor([1.0, 0.0, 0.0])]
    b = [torch.tensor([0.0, 1.0, 0.0])]
    assert grad_cosine(a, b) == pytest.approx(0.0, abs=1e-6)


# --- probe_due / next_probe_family -------------------------------------------


def test_probe_due_never_probed_then_after_period() -> None:
    stats = FamilyStats()
    assert probe_due(stats, FAMILY_A, current_update=0, period=50) is True

    stats.record_probe(FAMILY_A, cosine=0.1, update=0)
    assert probe_due(stats, FAMILY_A, current_update=49, period=50) is False
    assert probe_due(stats, FAMILY_A, current_update=50, period=50) is True


def test_next_probe_family_round_robin_order() -> None:
    stats = FamilyStats()
    families = ["a", "b", "c"]

    first = next_probe_family(stats, families, current_update=0)
    assert first == "a"
    stats.record_probe("a", cosine=0.1, update=0)

    second = next_probe_family(stats, families, current_update=1)
    assert second == "b"
    stats.record_probe("b", cosine=0.1, update=1)

    third = next_probe_family(stats, families, current_update=2)
    assert third == "c"
    stats.record_probe("c", cosine=0.1, update=2)

    fourth = next_probe_family(stats, families, current_update=3)
    assert fourth == "a"  # cycles back to the least recently probed

    assert next_probe_family(stats, [], current_update=4) is None

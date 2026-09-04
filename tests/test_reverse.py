"""Tests for conpo/reverse.py: reverse counterfactual construction (ConPO.pdf section 2.3)."""

import math
import random

import pytest

import conpo.operators  # noqa: F401  (registers all 53 operators)
from conpo import config
from conpo.data import PoolRow
from conpo.edges import FeasibleBuffer
from conpo.judge import StaticJudge
from conpo.operators import registry
from conpo.operators.registry import FIXTURES
from conpo.reverse import ReverseResult, eligible_families, reverse_batch, reverse_edge
from conpo.selection import FamilyStats
from conpo.verifiers import Constraint, check, feasible

NO_COMMA = Constraint("punctuation:no_comma", {})
LOWERCASE = Constraint("change_case:english_lowercase", {})
QUOTATION = Constraint("startend:quotation", {})


def _row(key: str, constraints: tuple[Constraint, ...], english: bool = True) -> PoolRow:
    return PoolRow(
        key=key,
        base_prompt="Write a short passage about the pipeline.",
        prompt="Write a short passage about the pipeline.",
        sentences=(),
        constraints=constraints,
        english=english,
        num_constraints=len(constraints),
    )


def _two_constraint_setup(seed: int = 0) -> tuple[PoolRow, str]:
    """(NO_COMMA, LOWERCASE): a single lowercase, comma-free fixture text
    satisfies both at once (ifeval_b.py's shared filler prose)."""
    constraints = (NO_COMMA, LOWERCASE)
    y_plus = FIXTURES[LOWERCASE.iid](LOWERCASE, random.Random(seed))
    assert feasible(constraints, y_plus)
    return _row("row-2", constraints), y_plus


def _three_constraint_setup(seed: int = 0) -> tuple[PoolRow, str]:
    """(NO_COMMA, LOWERCASE, QUOTATION): wrap the same fixture text in quotes;
    the quote marks are not cased letters, so all three hold jointly."""
    constraints = (NO_COMMA, LOWERCASE, QUOTATION)
    base = FIXTURES[LOWERCASE.iid](LOWERCASE, random.Random(seed))
    y_plus = f'"{base}"'
    assert feasible(constraints, y_plus)
    return _row("row-3", constraints), y_plus


# --- admission ---------------------------------------------------------------


@pytest.mark.parametrize("setup", [_two_constraint_setup, _three_constraint_setup])
def test_reverse_edge_admits_a_single_family_edge(setup) -> None:
    row, y_plus = setup()
    stats = FamilyStats()
    rng = random.Random(1)

    result = reverse_edge(row, y_plus, stats, StaticJudge(keep=True), rng, current_update=3)

    assert result.reason == "admitted"
    assert result.edge is not None
    edge = result.edge
    assert edge.direction == "reverse"
    assert result.family in eligible_families(row)
    assert edge.families == frozenset({result.family})
    assert edge.prompt_id == row.key
    assert edge.y_plus == y_plus
    assert edge.y_minus != y_plus
    assert edge.step == 0
    assert edge.created_update == 3
    assert edge.len_plus == 0 and edge.len_minus == 0
    assert math.isnan(edge.logp_ref_plus) and math.isnan(edge.logp_ref_minus)
    assert result.distance is not None and 0.0 < result.distance <= config.DELTA
    assert stats.n_c(result.family) == 1


# --- not_english skip ---------------------------------------------------------


def test_reverse_edge_skips_non_english_rows() -> None:
    row, y_plus = _two_constraint_setup()
    row = _row(row.key, row.constraints, english=False)

    result = reverse_edge(row, y_plus, FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=0)

    assert result == ReverseResult(edge=None, family=None, reason="not_english", distance=None, veto_kept=None)


# --- no eligible family --------------------------------------------------------


def test_reverse_edge_no_eligible_family_when_only_language_constraint() -> None:
    row = _row("row-lang", (Constraint("language:response_language", {"language": "en"}),))

    result = reverse_edge(row, "some response", FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=0)

    assert result == ReverseResult(edge=None, family=None, reason="no_eligible_family", distance=None, veto_kept=None)


# --- veto rejection ------------------------------------------------------------


def test_reverse_edge_veto_rejects_with_no_edge() -> None:
    row, y_plus = _two_constraint_setup()
    stats = FamilyStats()

    result = reverse_edge(row, y_plus, stats, StaticJudge(keep=False), random.Random(0), current_update=0)

    assert result.edge is None
    assert result.reason == "veto"
    assert result.veto_kept is False
    assert result.family is not None
    assert stats.n_c(result.family) == 0


# --- distance gate ---------------------------------------------------------------


def test_reverse_edge_distance_gate_rejects_under_a_tiny_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    row, y_plus = _two_constraint_setup()
    monkeypatch.setattr(config, "DELTA", 0.0)

    result = reverse_edge(row, y_plus, FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=0)

    assert result.edge is None
    assert result.reason == "distance"
    assert result.distance is not None and result.distance > 0.0


def test_reverse_edge_distance_gate_disabled_admits_past_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    row, y_plus = _two_constraint_setup()
    monkeypatch.setattr(config, "DELTA", 0.0)

    result = reverse_edge(
        row, y_plus, FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=0, distance_gate=False
    )

    assert result.edge is not None
    assert result.reason == "admitted"


# --- selection modes stay within the eligible set -----------------------------


@pytest.mark.parametrize("mode", ["uniform", "half"])
def test_uniform_and_half_selection_only_draw_eligible_families(mode: str) -> None:
    row, y_plus = _three_constraint_setup()
    eligible = set(eligible_families(row))
    stats = FamilyStats()
    rng = random.Random(0)

    drawn = {
        reverse_edge(row, y_plus, stats, StaticJudge(keep=True), rng, current_update=0, selection=mode).family
        for _ in range(200)
    }

    assert drawn
    assert drawn <= eligible


# --- exactness ablation --------------------------------------------------------


def test_reverse_edge_exactness_false_allows_collateral_damage(monkeypatch: pytest.MonkeyPatch) -> None:
    row, y_plus = _two_constraint_setup()  # constraints = (NO_COMMA, LOWERCASE)

    def _break_both(response: str, constraint: Constraint, rng: random.Random):
        # Inserts a comma (breaks NO_COMMA, the target) and an uppercase
        # first letter (breaks LOWERCASE, collateral).
        yield "A," + response[1:]

    monkeypatch.setitem(registry.OPERATORS, NO_COMMA.iid, _break_both)
    monkeypatch.delitem(registry.OPERATORS, LOWERCASE.iid)  # forces the only eligible family to be NO_COMMA

    exact_result = reverse_edge(
        row, y_plus, FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=0, exactness=True
    )
    assert exact_result.edge is None
    assert exact_result.reason.startswith(registry.COLLATERAL)

    loose_result = reverse_edge(
        row, y_plus, FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=0, exactness=False
    )
    assert loose_result.edge is not None
    assert loose_result.family == NO_COMMA.iid
    assert not check(NO_COMMA, loose_result.edge.y_minus)  # target genuinely fails
    assert not check(LOWERCASE, loose_result.edge.y_minus)  # collateral damage went unchecked, as designed


def test_reverse_edge_multi_target_exactness_false_retries_past_a_missing_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target whose operator yields no candidate on one attempt must not
    burn the whole max_candidates budget: the next attempt draws a fresh
    generator and can still succeed (round-2 fix to
    _corrupt_set_without_exactness, which used to `return None` on the
    first miss instead of retrying)."""
    row, y_plus = _two_constraint_setup()  # constraints = (NO_COMMA, LOWERCASE)
    calls = {"no_comma": 0}

    def _flaky_no_comma(response: str, constraint: Constraint, rng: random.Random):
        calls["no_comma"] += 1
        if calls["no_comma"] >= 2:  # first call yields nothing, every later call succeeds
            yield response + ","

    def _break_lowercase(response: str, constraint: Constraint, rng: random.Random):
        yield response[0].upper() + response[1:]

    monkeypatch.setitem(registry.OPERATORS, NO_COMMA.iid, _flaky_no_comma)
    monkeypatch.setitem(registry.OPERATORS, LOWERCASE.iid, _break_lowercase)

    result = reverse_edge(
        row,
        y_plus,
        FamilyStats(),
        StaticJudge(keep=True),
        random.Random(0),
        current_update=0,
        target_indices=(0, 1),
        exactness=False,
    )

    assert calls["no_comma"] >= 2  # the first miss did not stop retrying
    assert result.edge is not None
    assert result.reason == "admitted"
    assert not check(NO_COMMA, result.edge.y_minus)
    assert not check(LOWERCASE, result.edge.y_minus)


# --- reverse_batch -------------------------------------------------------------


def test_reverse_batch_draws_n_results_from_a_populated_feasible_buffer() -> None:
    row, y_plus = _two_constraint_setup()
    buffer = FeasibleBuffer()
    buffer.add(row.key, y_plus)
    stats = FamilyStats()

    results = reverse_batch(
        {row.key: row}, buffer, 5, stats, StaticJudge(keep=True), random.Random(0), current_update=0
    )

    assert len(results) == 5
    assert all(isinstance(r, ReverseResult) for r in results)
    assert all(r.edge is not None for r in results)
    assert stats.n_c(NO_COMMA.iid) + stats.n_c(LOWERCASE.iid) == 5

"""build_repair_messages, monotone_admit, repair_trajectory and repair_failed_group."""

import random

from conpo.data import PoolRow, TokenMeter
from conpo.engine import FakeEngine
from conpo.forward import (
    RepairResult,
    build_repair_messages,
    monotone_admit,
    repair_failed_group,
    repair_trajectory,
)
from conpo.judge import StaticJudge
from conpo.verifiers import Constraint, describe_failure

# ---------------------------------------------------------------------------
# monotone_admit truth table
# ---------------------------------------------------------------------------

_A, _B, _C = Constraint("a", {}), Constraint("b", {}), Constraint("c", {})


def test_monotone_admit_regression_takes_priority() -> None:
    ok, reason = monotone_admit((_A, _B, _C), (True, False, False), (False, False, True), d=0.0)
    assert not ok and reason == "regression:a"


def test_monotone_admit_no_improvement() -> None:
    ok, reason = monotone_admit((_A, _B), (True, False), (True, False), d=0.0)
    assert not ok and reason == "no_improvement"


def test_monotone_admit_distance() -> None:
    ok, reason = monotone_admit((_A, _B), (False, True), (True, True), d=0.5, delta=0.3)
    assert not ok and reason == "distance"


def test_monotone_admit_admitted() -> None:
    ok, reason = monotone_admit((_A, _B), (False, True), (True, True), d=0.1, delta=0.3)
    assert ok and reason == "admitted"


# ---------------------------------------------------------------------------
# build_repair_messages
# ---------------------------------------------------------------------------


def test_build_repair_messages_has_three_turns_and_every_failure() -> None:
    row = PoolRow(
        key="k",
        base_prompt="Do the task.",
        prompt="Do the task.",
        sentences=(),
        constraints=(),
        english=True,
        num_constraints=0,
    )
    response = "The Response, here."
    failed = [
        (Constraint("punctuation:no_comma", {}), describe_failure(Constraint("punctuation:no_comma", {}), response)),
        (
            Constraint("change_case:english_lowercase", {}),
            describe_failure(Constraint("change_case:english_lowercase", {}), response),
        ),
    ]

    messages = build_repair_messages(row, response, failed)

    assert len(messages) == 3
    assert messages[0] == {"role": "user", "content": row.prompt}
    assert messages[1] == {"role": "assistant", "content": response}
    assert messages[2]["role"] == "user"
    for constraint, failure_text in failed:
        assert constraint.iid in messages[2]["content"]
        assert failure_text in messages[2]["content"]


# ---------------------------------------------------------------------------
# repair_trajectory: a two-step trajectory with a rejected (regressing)
# candidate at step 1 and a feasible endpoint at step 2.
# ---------------------------------------------------------------------------

NO_COMMA = Constraint("punctuation:no_comma", {})
LOWERCASE = Constraint("change_case:english_lowercase", {})
NO_BANANA = Constraint("keywords:forbidden_words", {"forbidden_words": ["banana"]})
THREE_CONSTRAINTS = (NO_COMMA, LOWERCASE, NO_BANANA)

Y0 = "The platform, rotates at a constant velocity."  # fails NO_COMMA, LOWERCASE; satisfies NO_BANANA
PROPOSAL_1_FIXES_COMMA = "The platform rotates at a constant velocity."  # fixes NO_COMMA only
PROPOSAL_1_BREAKS_BANANA = "The platform, rotates at a constant banana velocity."  # regresses NO_BANANA
PROPOSAL_2_FIXES_REST = "the platform rotates at a constant velocity."  # fixes LOWERCASE too: feasible


def _row(key: str = "row-1") -> PoolRow:
    return PoolRow(
        key=key,
        base_prompt=Y0,
        prompt=Y0,
        sentences=(),
        constraints=THREE_CONSTRAINTS,
        english=True,
        num_constraints=len(THREE_CONSTRAINTS),
    )


def _scripted_engine() -> FakeEngine:
    # step 1 candidates (order matters: the regressing one is tried first and
    # rejected, then the comma-fixing one is admitted); step 2's first
    # candidate fixes the rest and is admitted immediately.
    return FakeEngine(
        [PROPOSAL_1_BREAKS_BANANA, PROPOSAL_1_FIXES_COMMA, PROPOSAL_2_FIXES_REST, PROPOSAL_2_FIXES_REST]
    )


def test_repair_trajectory_admits_two_edges_and_records_regression() -> None:
    engine = _scripted_engine()
    judge = StaticJudge(keep=True)
    meter = TokenMeter(budget=0)
    row = _row()

    result = repair_trajectory(
        row, Y0, engine, judge, random.Random(0), meter, current_update=7, max_steps=2, max_candidates=2
    )

    assert result.proposals == 4
    assert result.admitted == 2
    assert len(result.edges) == 2

    edge0, edge1 = result.edges
    assert edge0.step == 0
    assert edge0.y_minus == Y0
    assert edge0.y_plus == PROPOSAL_1_FIXES_COMMA
    assert edge0.families == frozenset({"punctuation:no_comma"})
    assert edge0.direction == "forward"
    assert edge0.created_update == 7
    assert edge0.logp_ref_plus != edge0.logp_ref_plus  # nan placeholder
    assert edge0.len_plus == 0 and edge0.len_minus == 0

    assert edge1.step == 1
    assert edge1.y_minus == PROPOSAL_1_FIXES_COMMA
    assert edge1.y_plus == PROPOSAL_2_FIXES_REST
    assert edge1.families == frozenset({"change_case:english_lowercase"})

    assert result.feasible_endpoints == [PROPOSAL_2_FIXES_REST]
    assert result.endpoint_sources == [result.edges]  # the endpoint depends on both trajectory edges
    assert result.reasons["regression:keywords:forbidden_words"] == 1
    assert result.generated_tokens > 0
    assert meter.total == result.generated_tokens


def test_repair_trajectory_static_judge_false_yields_zero_edges() -> None:
    engine = _scripted_engine()
    judge = StaticJudge(keep=False)
    meter = TokenMeter(budget=0)
    row = _row()

    result = repair_trajectory(
        row, Y0, engine, judge, random.Random(0), meter, current_update=0, max_steps=2, max_candidates=2
    )

    assert result.edges == []
    assert result.admitted == 0
    assert result.feasible_endpoints == []
    assert result.endpoint_sources == []
    # the meter is charged for rejected proposals too
    assert meter.total > 0
    assert result.reasons["veto"] >= 1


def test_repair_trajectory_already_feasible_response_needs_no_engine_call() -> None:
    engine = FakeEngine([])  # would raise if ever consumed
    judge = StaticJudge(keep=True)
    meter = TokenMeter(budget=0)
    row = _row()
    feasible_response = "the platform rotates at a constant velocity."

    result = repair_trajectory(
        row, feasible_response, engine, judge, random.Random(0), meter, current_update=0
    )

    # No engine call means no edge was built, so the endpoint has no source
    # edges: nothing to withhold it on (trainer.py's _defer_feasible_endpoint
    # adds it straight to feasible_buffer for exactly this shape).
    assert result == RepairResult(feasible_endpoints=[feasible_response], endpoint_sources=[[]])
    assert engine.calls == []
    assert meter.total == 0


# ---------------------------------------------------------------------------
# repair_failed_group: merges independent trajectories over several responses
# ---------------------------------------------------------------------------


def test_repair_failed_group_merges_edges_across_responses() -> None:
    def fixer(messages, n):
        current = messages[1]["content"]
        return [current.replace(",", "").lower()] * n

    engine = FakeEngine(fixer)
    judge = StaticJudge(keep=True)
    meter = TokenMeter(budget=0)
    row = _row()
    # same shape as Y0 (one leading capital, one comma, 7 tokens) so the
    # single-step fix stays within the same normalised distance bound
    responses = [Y0, "This engine, hums at a constant frequency."]

    result = repair_failed_group(
        row, responses, engine, judge, random.Random(0), meter, current_update=3, max_steps=2, max_candidates=2
    )

    assert result.admitted == 2
    assert len(result.edges) == 2
    assert len(result.feasible_endpoints) == 2
    assert len(result.endpoint_sources) == 2
    assert all(len(sources) == 1 for sources in result.endpoint_sources)  # one edge per single-step trajectory
    assert result.proposals == 2 * 2
    assert meter.total == result.generated_tokens

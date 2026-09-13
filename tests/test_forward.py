"""build_repair_messages, monotone_admit, repair_trajectory, repair_failed_group
and repair_all_groups."""

import random
from collections.abc import Callable, Mapping, Sequence

from conpo import config
from conpo.data import PoolRow, TokenMeter
from conpo.distance import token_distance
from conpo.engine import FakeEngine
from conpo.forward import (
    RepairResult,
    build_repair_messages,
    monotone_admit,
    repair_all_groups,
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


def test_monotone_admit_distance_gate_off_admits_despite_distance() -> None:
    # RUN_NO_DISTANCE (Overrides.no_distance): the same over-bound candidate
    # test_monotone_admit_distance rejects is admitted once distance_gate=False.
    ok, reason = monotone_admit((_A, _B), (False, True), (True, True), d=0.5, delta=0.3, distance_gate=False)
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
# repair_trajectory: distance_gate (RUN_NO_DISTANCE) threading, section 2.2
# ---------------------------------------------------------------------------

FAR_Y0 = "The platform, rotates at a constant velocity."
# Fixes NO_COMMA but is a near-total rewrite: well past DELTA's 0.3 bound.
FAR_PROPOSAL = "A totally different machine spins forever without pause today."


def _far_row(key: str = "row-far") -> PoolRow:
    return PoolRow(
        key=key,
        base_prompt=FAR_Y0,
        prompt=FAR_Y0,
        sentences=(),
        constraints=(NO_COMMA,),
        english=True,
        num_constraints=1,
    )


def test_repair_trajectory_distance_gate_rejects_then_admits_when_disabled() -> None:
    row = _far_row()
    judge = StaticJudge(keep=True)

    d = token_distance(FAR_Y0, FAR_PROPOSAL)
    assert d > config.DELTA  # sanity: the scenario exceeds the bound

    gated = repair_trajectory(
        row,
        FAR_Y0,
        FakeEngine([FAR_PROPOSAL]),
        judge,
        random.Random(0),
        TokenMeter(budget=0),
        current_update=0,
        max_steps=1,
        max_candidates=1,
        distance_gate=True,
    )
    assert gated.edges == []
    assert gated.reasons["distance"] == 1

    ungated = repair_trajectory(
        row,
        FAR_Y0,
        FakeEngine([FAR_PROPOSAL]),
        judge,
        random.Random(0),
        TokenMeter(budget=0),
        current_update=0,
        max_steps=1,
        max_candidates=1,
        distance_gate=False,
    )
    assert len(ungated.edges) == 1
    assert ungated.edges[0].y_plus == FAR_PROPOSAL


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


def test_repair_failed_group_batches_identically_to_sequential_repair() -> None:
    """repair_failed_group batches every still-active response's proposals
    into one engine.generate call per repair step (design choice, Ethan
    devbox perf pass, 2026-09-11) instead of the old one-call-per-response
    walk. `fixer` is a pure function of message content -- ignorant of call
    order and the shared per-step seed -- so it stands in for a real engine
    identically whichever way responses are grouped into calls. This proves
    the batched path admits exactly the same edges, per response, as
    replaying the pre-batching algorithm (one `repair_trajectory` call per
    response, merged in order) would.
    """

    def fixer(messages: list[dict[str, str]], n: int) -> list[str]:
        y_t = messages[1]["content"]
        if "," in y_t:
            candidate = y_t.replace(",", "")
        elif not y_t.islower():
            candidate = y_t.lower()
        else:
            candidate = y_t  # unreachable: an already-feasible response gets no engine call
        return [candidate] * n

    row = _row()
    responses = [
        Y0,  # fails NO_COMMA and LOWERCASE: 2 repair steps
        "The platform rotates at a constant velocity.",  # fails only LOWERCASE: 1 step
        "the platform rotates at a constant velocity.",  # already feasible: 0 engine calls
    ]

    def sequential_reference(responses: list[str]) -> RepairResult:
        """The pre-batching repair_failed_group: repair_trajectory once per
        response, merged in response order."""
        merged = RepairResult()
        for response in responses:
            r = repair_trajectory(
                row,
                response,
                FakeEngine(fixer),
                StaticJudge(keep=True),
                random.Random(0),
                TokenMeter(budget=0),
                current_update=5,
                max_steps=3,
                max_candidates=2,
            )
            merged.edges.extend(r.edges)
            merged.feasible_endpoints.extend(r.feasible_endpoints)
            merged.endpoint_sources.extend(r.endpoint_sources)
            merged.proposals += r.proposals
            merged.admitted += r.admitted
            merged.generated_tokens += r.generated_tokens
            merged.reasons.update(r.reasons)
        return merged

    sequential = sequential_reference(responses)
    batched = repair_failed_group(
        row,
        responses,
        FakeEngine(fixer),
        StaticJudge(keep=True),
        random.Random(0),
        TokenMeter(budget=0),
        current_update=5,
        max_steps=3,
        max_candidates=2,
    )

    def edge_key(edges: list) -> list[tuple[str, str, list[str], int]]:
        return sorted((e.y_minus, e.y_plus, sorted(e.families), e.step) for e in edges)

    # Batching only changes the interleaving of DIFFERENT responses' edges
    # (step-major instead of response-major); every response's own admitted
    # edges, and every aggregate count, must match exactly.
    assert edge_key(sequential.edges) == edge_key(batched.edges)
    assert sequential.admitted == batched.admitted == 3
    assert sequential.proposals == batched.proposals
    assert sorted(sequential.feasible_endpoints) == sorted(batched.feasible_endpoints) == sorted(
        ["the platform rotates at a constant velocity."] * 3
    )
    assert dict(sequential.reasons) == dict(batched.reasons)


# ---------------------------------------------------------------------------
# repair_all_groups: merges repair_failed_group's own lock-step batch one
# level up, across every failing group in a step (REPAIR_BATCH_GENERATION)
# ---------------------------------------------------------------------------


class _CountingEngine(FakeEngine):
    """FakeEngine that also counts engine.generate() INVOCATIONS (not
    prompts): FakeEngine.calls records one entry per prompt regardless of
    how many prompts one generate() call batched together, so it cannot by
    itself show that cross-group merging cut the number of round trips."""

    def __init__(
        self, responses: Callable[[Sequence[Mapping[str, str]], int], Sequence[str]] | Sequence[str]
    ) -> None:
        super().__init__(responses)
        self.generate_call_count = 0

    def generate(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, str]]],
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[list[str]]:
        self.generate_call_count += 1
        return super().generate(messages_batch, n, max_tokens, temperature, top_p, seed)


def test_repair_all_groups_batches_identically_to_per_group_repair_and_merges_calls() -> None:
    """repair_all_groups pools every failing GROUP's active trajectories
    into one engine.generate call per repair round, on top of
    repair_failed_group's own per-response merge within a group (design
    choice, Ethan devbox perf pass, 2026-09-13). `fixer` is a pure function
    of message content -- ignorant of call order and the shared per-step
    seed -- so it stands in for a real engine identically whichever way
    groups are pooled into calls. This proves the cross-group merge admits
    exactly the same edges, per group, as calling repair_failed_group once
    per group (each with its own fresh engine/rng) and keeping the results
    in group order would, while cutting the number of engine.generate
    calls to at most max_steps for the WHOLE pool of groups instead of
    max_steps PER group.
    """

    def fixer(messages: list[dict[str, str]], n: int) -> list[str]:
        y_t = messages[1]["content"]
        if "," in y_t:
            candidate = y_t.replace(",", "")
        elif not y_t.islower():
            candidate = y_t.lower()
        else:
            candidate = y_t  # unreachable: an already-feasible response gets no engine call
        return [candidate] * n

    row_a = _row("group-a")
    row_b = _row("group-b")
    responses_a = [
        Y0,  # fails NO_COMMA and LOWERCASE: 2 repair steps
        "The platform rotates at a constant velocity.",  # fails only LOWERCASE: 1 step
    ]
    responses_b = [
        "the platform rotates at a constant velocity.",  # already feasible: 0 engine calls
        Y0,  # fails NO_COMMA and LOWERCASE: 2 repair steps
    ]
    groups = [(row_a, responses_a), (row_b, responses_b)]

    reference = [
        repair_failed_group(
            row,
            responses,
            FakeEngine(fixer),
            StaticJudge(keep=True),
            random.Random(0),
            TokenMeter(budget=0),
            current_update=7,
            max_steps=3,
            max_candidates=2,
        )
        for row, responses in groups
    ]

    engine = _CountingEngine(fixer)
    batched = repair_all_groups(
        groups,
        engine,
        [StaticJudge(keep=True), StaticJudge(keep=True)],
        random.Random(0),
        TokenMeter(budget=0),
        current_update=7,
        max_steps=3,
        max_candidates=2,
    )

    def edge_key(edges: list) -> list[tuple[str, str, str, list[str], int]]:
        return sorted((e.prompt_id, e.y_minus, e.y_plus, sorted(e.families), e.step) for e in edges)

    assert len(batched) == len(groups) == 2
    for ref_result, batched_result in zip(reference, batched):
        assert edge_key(ref_result.edges) == edge_key(batched_result.edges)
        assert ref_result.admitted == batched_result.admitted
        assert ref_result.proposals == batched_result.proposals
        assert sorted(ref_result.feasible_endpoints) == sorted(batched_result.feasible_endpoints)
        assert dict(ref_result.reasons) == dict(batched_result.reasons)

    # The whole point: both groups need 2 repair rounds each (Y0 needs a
    # comma fix then a case fix in every group it appears in), so running
    # them one at a time costs 2 + 2 = 4 engine.generate calls; pooled here
    # they share the SAME 2 rounds, so only 2 calls cover the whole step.
    assert engine.generate_call_count == 2

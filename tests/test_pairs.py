"""Tests for conpo/pairs.py: DPO baseline pair construction (locked decision
2026-09-02(c)). FakeEngine responses are scripted from real verifier
families -- FIXTURES for passing responses, hand-picked failing text for
corrupted ones -- mirroring tests/test_reverse.py and tests/test_trainer_tiny.py's
established pattern.
"""

import json
import random

import pytest

import conpo.operators  # noqa: F401  registers all corruption operators
from conpo.arms import ARMS, validate
from conpo.data import PoolRow, TokenMeter
from conpo.edges import Edge
from conpo.engine import FakeEngine
from conpo.operators.registry import FIXTURES
from conpo.pairs import Pair, build_edge_pairs, build_relax_pairs, build_vf_pairs, load_pairs, save_pairs
from conpo.verifiers import Constraint, check, feasible

NO_COMMA = Constraint("punctuation:no_comma", {})
LOWERCASE = Constraint("change_case:english_lowercase", {})

# From conpo.operators FIXTURES: passes both NO_COMMA and LOWERCASE.
PASSING = FIXTURES[LOWERCASE.iid](LOWERCASE, random.Random(0))
# Hand-picked failing text: fails NO_COMMA (comma present), passes LOWERCASE.
FAILS_NO_COMMA = "hello, world"
# Hand-picked failing text: fails LOWERCASE (capital H), passes NO_COMMA.
FAILS_LOWERCASE = "Hello world"

assert feasible((NO_COMMA, LOWERCASE), PASSING)
assert not feasible((NO_COMMA, LOWERCASE), FAILS_NO_COMMA)
assert not feasible((NO_COMMA, LOWERCASE), FAILS_LOWERCASE)


def _row(key: str = "row-1") -> PoolRow:
    return PoolRow(
        key=key,
        base_prompt="Write a short passage about the pipeline.",
        prompt="Write a short passage about the pipeline.",
        sentences=(),
        constraints=(NO_COMMA, LOWERCASE),
        english=True,
        num_constraints=2,
    )


# ---------------------------------------------------------------------------
# build_vf_pairs
# ---------------------------------------------------------------------------


def test_build_vf_pairs_admits_a_pair_when_both_outcomes_present() -> None:
    row = _row()
    engine = FakeEngine([PASSING, FAILS_NO_COMMA, PASSING, FAILS_NO_COMMA])
    meter = TokenMeter(budget=0)

    pairs = build_vf_pairs([row], engine, n_samples=4, rng=random.Random(0), meter=meter)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.source == "vf"
    assert pair.prompt_id == row.key
    assert pair.prompt == row.prompt
    assert feasible(row.constraints, pair.chosen)
    assert not feasible(row.constraints, pair.rejected)
    assert meter.total > 0


def test_build_vf_pairs_pairs_per_prompt_draws_that_many_pairs() -> None:
    row = _row()
    engine = FakeEngine([PASSING, FAILS_NO_COMMA, PASSING, FAILS_NO_COMMA])

    pairs = build_vf_pairs([row], engine, n_samples=4, rng=random.Random(0), meter=TokenMeter(budget=0), pairs_per_prompt=3)

    assert len(pairs) == 3
    assert all(p.source == "vf" for p in pairs)


def test_build_vf_pairs_drops_prompt_with_only_passing_responses() -> None:
    row = _row()
    engine = FakeEngine([PASSING, PASSING, PASSING, PASSING])

    pairs = build_vf_pairs([row], engine, n_samples=4, rng=random.Random(0), meter=TokenMeter(budget=0))

    assert pairs == []


def test_build_vf_pairs_drops_prompt_with_only_failing_responses() -> None:
    row = _row()
    engine = FakeEngine([FAILS_NO_COMMA, FAILS_LOWERCASE, FAILS_NO_COMMA, FAILS_LOWERCASE])

    pairs = build_vf_pairs([row], engine, n_samples=4, rng=random.Random(0), meter=TokenMeter(budget=0))

    assert pairs == []


def test_build_vf_pairs_empty_rows_needs_no_engine_call() -> None:
    engine = FakeEngine([])  # would raise if ever consumed

    pairs = build_vf_pairs([], engine, n_samples=4, rng=random.Random(0), meter=TokenMeter(budget=0))

    assert pairs == []
    assert engine.calls == []


# ---------------------------------------------------------------------------
# build_relax_pairs
# ---------------------------------------------------------------------------

SENTENCE_NO_COMMA = "Do not use any commas."
SENTENCE_LOWERCASE = "Write your entire answer in lowercase letters."


def _relax_row(key: str = "row-relax") -> PoolRow:
    base = "Explain how the pipeline works."
    prompt = f"{base} {SENTENCE_NO_COMMA} {SENTENCE_LOWERCASE}"
    return PoolRow(
        key=key,
        base_prompt=base,
        prompt=prompt,
        sentences=(SENTENCE_NO_COMMA, SENTENCE_LOWERCASE),
        constraints=(NO_COMMA, LOWERCASE),
        english=True,
        num_constraints=2,
    )


def _relax_responder(row: PoolRow):
    def responder(messages, n):  # noqa: ANN001
        content = messages[0]["content"]
        if content == row.prompt:
            return [PASSING] * n
        if SENTENCE_NO_COMMA not in content:  # NO_COMMA was the deleted constraint
            return [FAILS_NO_COMMA] * n
        return [FAILS_LOWERCASE] * n  # LOWERCASE was the deleted constraint

    return responder


def test_build_relax_pairs_one_pair_per_constraint_index() -> None:
    row = _relax_row()
    engine = FakeEngine(_relax_responder(row))
    meter = TokenMeter(budget=0)

    pairs = build_relax_pairs([row], engine, n_samples=3, rng=random.Random(0), meter=meter)

    assert len(pairs) == 2
    by_family = {p.meta["deleted_family"]: p for p in pairs}
    assert set(by_family) == {NO_COMMA.iid, LOWERCASE.iid}

    no_comma_pair = by_family[NO_COMMA.iid]
    assert no_comma_pair.source == "relax"
    assert no_comma_pair.meta["deleted_index"] == 0
    assert feasible(row.constraints, no_comma_pair.chosen)
    assert not check(NO_COMMA, no_comma_pair.rejected)

    lowercase_pair = by_family[LOWERCASE.iid]
    assert lowercase_pair.meta["deleted_index"] == 1
    assert feasible(row.constraints, lowercase_pair.chosen)
    assert not check(LOWERCASE, lowercase_pair.rejected)

    assert meter.total > 0


def test_build_relax_pairs_drops_when_no_rejected_fails_the_deleted_constraint() -> None:
    row = _relax_row()

    def always_passing(messages, n):  # noqa: ANN001
        return [PASSING] * n

    engine = FakeEngine(always_passing)  # rejected candidates never fail their deleted constraint

    pairs = build_relax_pairs([row], engine, n_samples=3, rng=random.Random(0), meter=TokenMeter(budget=0))

    assert pairs == []


# ---------------------------------------------------------------------------
# build_edge_pairs
# ---------------------------------------------------------------------------


def _edge(prompt_id: str, direction: str, families: frozenset[str]) -> Edge:
    return Edge(
        prompt_id=prompt_id,
        prompt=f"prompt for {prompt_id}",
        y_plus="the feasible response.",
        y_minus="the corrupted response.",
        families=families,
        direction=direction,
        step=0,
        logp_ref_plus=-1.0,
        logp_ref_minus=-2.0,
        len_plus=5,
        len_minus=5,
        created_update=3,
        meta={"reason": "admitted"},
    )


def test_build_edge_pairs_round_trips_a_two_edge_file(tmp_path) -> None:
    edges = [
        _edge("p1", "reverse", frozenset({NO_COMMA.iid})),
        _edge("p2", "forward", frozenset({LOWERCASE.iid})),
    ]
    path = tmp_path / "edges.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for e in edges:
            f.write(json.dumps(e.to_json()) + "\n")

    pairs = build_edge_pairs(path)

    assert len(pairs) == 2
    for pair, edge in zip(pairs, edges):
        assert pair.source == "edge"
        assert pair.prompt_id == edge.prompt_id
        assert pair.prompt == edge.prompt
        assert pair.chosen == edge.y_plus
        assert pair.rejected == edge.y_minus
        assert pair.meta["direction"] == edge.direction
        assert pair.meta["families"] == sorted(edge.families)


# ---------------------------------------------------------------------------
# Pair jsonl I/O
# ---------------------------------------------------------------------------


def test_save_and_load_pairs_round_trip(tmp_path) -> None:
    pairs = [
        Pair("p1", "prompt one", "chosen one", "rejected one", "vf", {}),
        Pair("p2", "prompt two", "chosen two", "rejected two", "relax", {"deleted_family": NO_COMMA.iid}),
    ]
    path = tmp_path / "pairs.jsonl"

    save_pairs(pairs, path)
    loaded = load_pairs(path)

    assert loaded == pairs


# ---------------------------------------------------------------------------
# conpo/arms.py: vf_dpo / relax_dpo / edge_dpo registration (folded in here
# per the file scope assigned to this task)
# ---------------------------------------------------------------------------


def test_dpo_arms_registered_with_dpo_trainer() -> None:
    for name in ("vf_dpo", "relax_dpo", "edge_dpo"):
        validate(name)  # does not raise
        assert ARMS[name].trainer == "dpo"
    for name in ("full", "ifrlvr", "solve_only", "forward_only", "reverse_only"):
        assert ARMS[name].trainer == "grpo"


def test_validate_still_rejects_unknown_arm() -> None:
    with pytest.raises(ValueError):
        validate("not-a-real-arm")

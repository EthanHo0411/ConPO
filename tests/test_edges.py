"""Tests for conpo.edges."""

import dataclasses
import random
from collections import Counter

import pytest

from conpo.edges import Edge, EdgeBuffer, FeasibleBuffer


def _make_edge(
    direction: str,
    created_update: int,
    prompt_id: str = "p",
    families: frozenset[str] = frozenset({"keywords:existence"}),
) -> Edge:
    return Edge(
        prompt_id=prompt_id,
        prompt="prompt text",
        y_plus="good response",
        y_minus="bad response",
        families=families,
        direction=direction,
        step=0,
        logp_ref_plus=-3.5,
        logp_ref_minus=-4.2,
        len_plus=12,
        len_minus=10,
        created_update=created_update,
    )


def test_edge_json_round_trip() -> None:
    edge = Edge(
        prompt_id="p1",
        prompt="prompt text",
        y_plus="good response",
        y_minus="bad response",
        families=frozenset({"keywords:existence", "punctuation:no_comma"}),
        direction="forward",
        step=1,
        logp_ref_plus=-3.5,
        logp_ref_minus=-4.2,
        len_plus=12,
        len_minus=10,
        created_update=5,
        meta={"note": "test"},
        logp_ref_plus_xminus=-2.1,
        logp_ref_minus_xminus=-6.7,
    )
    restored = Edge.from_json(edge.to_json())
    assert restored == edge
    assert restored.logp_ref_plus_xminus == -2.1
    assert restored.logp_ref_minus_xminus == -6.7


def test_edge_from_json_defaults_missing_xminus_cache_to_none() -> None:
    """A pre-L3 edges.jsonl line has neither key (RUN_CACHE_REF_LOGPROBS,
    L3): from_json must default them to None rather than raise, so an old
    saved buffer still loads and recomputes lazily on next draw."""
    edge = _make_edge(direction="forward", created_update=0)
    pre_l3_json = edge.to_json()
    del pre_l3_json["logp_ref_plus_xminus"]
    del pre_l3_json["logp_ref_minus_xminus"]

    restored = Edge.from_json(pre_l3_json)

    assert restored.logp_ref_plus_xminus is None
    assert restored.logp_ref_minus_xminus is None


def test_edge_rejects_bad_direction() -> None:
    with pytest.raises(ValueError):
        _make_edge(direction="sideways", created_update=0)


def test_fifo_eviction_at_capacity() -> None:
    buf = EdgeBuffer(capacity=5, age_cap=1000)
    for i in range(8):
        buf.push(_make_edge(direction="forward", created_update=i, prompt_id=f"p{i}"))
    assert len(buf) == 5
    remaining = {e.prompt_id for e in buf.sample(10, current_update=8, rng=random.Random(0), balanced=False)}
    assert remaining == {"p3", "p4", "p5", "p6", "p7"}


def test_age_cap_retirement() -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=5)
    buf.push(_make_edge(direction="forward", created_update=0, prompt_id="a"))
    buf.push(_make_edge(direction="forward", created_update=3, prompt_id="b"))
    buf.push(_make_edge(direction="forward", created_update=4, prompt_id="c"))
    assert len(buf) == 3  # nothing exceeds age_cap relative to the updates seen so far

    result = buf.sample(10, current_update=8, rng=random.Random(0), balanced=False)
    # at current_update=8: a age=8 (>5, retired), b age=5 (<=5, kept), c age=4 (kept)
    assert {e.prompt_id for e in result} == {"b", "c"}
    assert len(buf) == 2


# ---------------------------------------------------------------------------
# RUN_EDGE_DEDUPE (audit F6, devbox 2026-09-15): push() refuses an exact
# (prompt, y_plus, y_minus, direction) duplicate of an already-admitted edge.
# _make_edge's default prompt/y_plus/y_minus text is identical across every
# call (only prompt_id/created_update/families vary), so two _make_edge(...)
# calls that share `direction` always share the dedupe key too.
# ---------------------------------------------------------------------------


def test_push_returns_true_and_dedupe_off_by_default() -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    assert buf.dedupe is False
    assert buf.push(_make_edge(direction="forward", created_update=0, prompt_id="a")) is True


def test_push_dedupe_off_allows_exact_duplicate_key() -> None:
    """Byte-identical to before RUN_EDGE_DEDUPE existed: two edges sharing
    (prompt, y_plus, y_minus, direction) both get admitted when dedupe is off."""
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    assert buf.push(_make_edge(direction="forward", created_update=0, prompt_id="a")) is True
    assert buf.push(_make_edge(direction="forward", created_update=1, prompt_id="b")) is True
    assert len(buf) == 2


def test_push_dedupe_on_refuses_exact_duplicate_key() -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=1000, dedupe=True)
    assert buf.push(_make_edge(direction="forward", created_update=0, prompt_id="a")) is True
    assert buf.push(_make_edge(direction="forward", created_update=1, prompt_id="b")) is False
    assert len(buf) == 1


def test_push_dedupe_on_allows_different_direction() -> None:
    """The dedupe key includes direction, so a forward and a reverse edge
    over the identical (prompt, y_plus, y_minus) text are not duplicates."""
    buf = EdgeBuffer(capacity=1000, age_cap=1000, dedupe=True)
    assert buf.push(_make_edge(direction="forward", created_update=0)) is True
    assert buf.push(_make_edge(direction="reverse", created_update=0)) is True
    assert len(buf) == 2


def test_push_dedupe_key_survives_fifo_eviction() -> None:
    """"Already admitted" is a run-lifetime fact, not "still in the buffer":
    a key stays refused even after the original edge that admitted it has
    aged out of `_edges`."""
    buf = EdgeBuffer(capacity=1000, age_cap=1, dedupe=True)
    assert buf.push(_make_edge(direction="forward", created_update=0, prompt_id="a")) is True

    # A later, distinct-key edge ages the first one out (age_cap=1, age 5 > 1).
    assert buf.push(_make_edge(direction="reverse", created_update=5, prompt_id="filler")) is True
    assert len(buf) == 1  # the forward edge retired; only "filler" remains

    duplicate = _make_edge(direction="forward", created_update=5, prompt_id="b")
    assert buf.push(duplicate) is False  # same key still refused though the original is gone
    assert len(buf) == 1


def test_balanced_sampling_counts() -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    for i in range(30):
        buf.push(_make_edge(direction="forward", created_update=0, prompt_id=f"f{i}"))
    for i in range(10):
        buf.push(_make_edge(direction="reverse", created_update=0, prompt_id=f"r{i}"))
    rng = random.Random(0)

    sample20 = buf.sample(20, current_update=0, rng=rng)
    assert Counter(e.direction for e in sample20) == {"forward": 10, "reverse": 10}

    sample40 = buf.sample(40, current_update=0, rng=rng)
    assert Counter(e.direction for e in sample40) == {"forward": 30, "reverse": 10}


def test_balanced_sampling_falls_back_when_one_side_missing() -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    for i in range(4):
        buf.push(_make_edge(direction="forward", created_update=0, prompt_id=f"f{i}"))
    sampled = buf.sample(10, current_update=0, rng=random.Random(0))
    assert len(sampled) == 4
    assert all(e.direction == "forward" for e in sampled)


def test_counts_by_direction_and_family() -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    buf.push(
        _make_edge(direction="forward", created_update=0, prompt_id="f0", families=frozenset({"keywords:existence"}))
    )
    buf.push(
        _make_edge(
            direction="reverse",
            created_update=0,
            prompt_id="r0",
            families=frozenset({"keywords:existence", "punctuation:no_comma"}),
        )
    )
    assert buf.counts_by_direction() == {"forward": 1, "reverse": 1}
    assert buf.counts_by_family() == {"keywords:existence": 2, "punctuation:no_comma": 1}


def test_save_load_round_trip(tmp_path) -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    buf.push(_make_edge(direction="forward", created_update=0, prompt_id="f0"))
    buf.push(_make_edge(direction="reverse", created_update=1, prompt_id="r0"))
    path = tmp_path / "edges.jsonl"
    buf.save(path)

    loaded = EdgeBuffer(capacity=1000, age_cap=1000)
    loaded.load(path)

    assert len(loaded) == len(buf)
    assert loaded.counts_by_direction() == buf.counts_by_direction()
    original_ids = {e.prompt_id for e in buf.sample(10, current_update=1, rng=random.Random(0), balanced=False)}
    loaded_ids = {e.prompt_id for e in loaded.sample(10, current_update=1, rng=random.Random(0), balanced=False)}
    assert original_ids == loaded_ids


def test_save_load_round_trip_preserves_xminus_cache(tmp_path) -> None:
    """RUN_CACHE_REF_LOGPROBS (L3): a filled x^-A cache survives save/load
    (RUN_RESUME), so a resumed run does not needlessly recompute it."""
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    cached = dataclasses.replace(
        _make_edge(direction="forward", created_update=0, prompt_id="f0"),
        logp_ref_plus_xminus=-1.23,
        logp_ref_minus_xminus=-4.56,
    )
    buf.push(cached)
    buf.push(_make_edge(direction="reverse", created_update=1, prompt_id="r0"))
    path = tmp_path / "edges.jsonl"
    buf.save(path)

    loaded = EdgeBuffer(capacity=1000, age_cap=1000)
    loaded.load(path)

    by_id = {e.prompt_id: e for e in loaded.sample(10, current_update=1, rng=random.Random(0), balanced=False)}
    assert by_id["f0"].logp_ref_plus_xminus == -1.23
    assert by_id["f0"].logp_ref_minus_xminus == -4.56
    assert by_id["r0"].logp_ref_plus_xminus is None


def test_discard_removes_given_edges_and_reports_count() -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    edges = [_make_edge(direction="forward", created_update=0, prompt_id=f"f{i}") for i in range(5)]
    for e in edges:
        buf.push(e)

    removed = buf.discard([edges[1], edges[3]])

    assert removed == 2
    assert len(buf) == 3
    remaining = {e.prompt_id for e in buf.sample(10, current_update=0, rng=random.Random(0), balanced=False)}
    assert remaining == {"f0", "f2", "f4"}


def test_discard_edge_not_in_buffer_counts_as_not_found() -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    buf.push(_make_edge(direction="forward", created_update=0, prompt_id="kept"))
    stray = _make_edge(direction="forward", created_update=0, prompt_id="stray")  # never pushed

    assert buf.discard([stray]) == 0
    assert len(buf) == 1


def test_discard_empty_list_is_a_no_op() -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    buf.push(_make_edge(direction="forward", created_update=0, prompt_id="a"))

    assert buf.discard([]) == 0
    assert len(buf) == 1


def test_discarded_edge_does_not_return_after_save_load_round_trip(tmp_path) -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    edges = [_make_edge(direction="forward", created_update=0, prompt_id=f"f{i}") for i in range(3)]
    for e in edges:
        buf.push(e)
    buf.discard([edges[1]])

    path = tmp_path / "edges.jsonl"
    buf.save(path)
    loaded = EdgeBuffer(capacity=1000, age_cap=1000)
    loaded.load(path)

    assert len(loaded) == 2
    loaded_ids = {e.prompt_id for e in loaded.sample(10, current_update=0, rng=random.Random(0), balanced=False)}
    assert loaded_ids == {"f0", "f2"}


def test_sample_where_filters_by_predicate_and_retires_stale() -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=5)
    buf.push(_make_edge(direction="forward", created_update=0, prompt_id="stale", families=frozenset({"a"})))
    buf.push(_make_edge(direction="forward", created_update=3, prompt_id="b", families=frozenset({"a"})))
    buf.push(_make_edge(direction="reverse", created_update=4, prompt_id="c", families=frozenset({"other"})))
    buf.push(_make_edge(direction="reverse", created_update=4, prompt_id="d", families=frozenset({"a"})))

    result = buf.sample_where(lambda e: "a" in e.families, n=10, current_update=8, rng=random.Random(0))

    # "stale" (age=8 > age_cap=5) is retired before the predicate ever sees
    # it; "c" survives retirement but doesn't match the predicate.
    assert {e.prompt_id for e in result} == {"b", "d"}
    assert len(buf) == 3


def test_sample_where_caps_at_n_with_no_duplicates() -> None:
    buf = EdgeBuffer(capacity=1000, age_cap=1000)
    for i in range(20):
        buf.push(_make_edge(direction="forward", created_update=0, prompt_id=f"f{i}"))

    result = buf.sample_where(lambda e: True, n=5, current_update=0, rng=random.Random(0))

    assert len(result) == 5
    assert len({e.prompt_id for e in result}) == 5


def test_feasible_buffer_caps_per_prompt() -> None:
    buf = FeasibleBuffer(per_prompt_cap=3)
    for i in range(5):
        buf.add("p1", f"response {i}")
    assert len(buf) == 3

    prompt_id, response = buf.sample_prompt(random.Random(0))
    assert prompt_id == "p1"
    assert response in {"response 2", "response 3", "response 4"}  # oldest two evicted

    buf.drop("p1")
    assert len(buf) == 0

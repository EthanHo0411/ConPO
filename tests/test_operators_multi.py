"""Tests for the multi-target registry additions: exact_set and corrupt_set
(H6 part A, design 2026-09-03). Fixtures below are two- and three-constraint
prompts hand-built to satisfy every constraint on them, spanning both IFEval
(groups A/B) and IFBench-Train (groups C/D) families, mirroring the
cross-group fixture style of tests/test_operators_all.py.
"""

from __future__ import annotations

import random
import re

import pytest

import conpo.operators  # noqa: F401  (registers every family)
from conpo.operators import registry
from conpo.verifiers import Constraint, check, feasible

# ---------------------------------------------------------------------------
# multi-constraint fixtures: (name, (constraint_0, constraint_1), text),
# both constraints are the corrupt_set targets (target_indices = (0, 1)).
# ---------------------------------------------------------------------------

MULTI_CASES: list[tuple[str, tuple[Constraint, Constraint], str]] = [
    (
        "no_comma[B]+word_once[C]",
        (
            Constraint("punctuation:no_comma", {}),
            Constraint("keywords:word_once", {"keyword": "unique"}),
        ),
        "the report highlights one unique finding from this quarter without any repeated commas in this response",
    ),
    (
        "english_lowercase[B]+last_word_answer[C]",
        (
            Constraint("change_case:english_lowercase", {}),
            Constraint("last_word:last_word_answer", {"last_word": "today"}),
        ),
        "the meeting finished early and everyone left the building quietly before sunset today",
    ),
    (
        "title[B]+punctuation_exclamation[D]",
        (
            Constraint("detectable_format:title", {}),
            Constraint("punctuation:punctuation_exclamation", {}),
        ),
        "<<field notes>> gentle rivers wander through quiet valleys near old stone bridges each morning",
    ),
    (
        "forbidden_words[A]+count_unique[D]",
        (
            Constraint("keywords:forbidden_words", {"forbidden_words": ["banana", "grape"]}),
            Constraint("count:count_unique", {}),
        ),
        "the orchard grows apples and pears throughout this year without any banned produce mentioned here",
    ),
    (
        "end_checker[B]+no_adjacent_consecutive[C]",
        (
            Constraint("startend:end_checker", {"end_phrase": "that is all"}),
            Constraint("keywords:no_adjacent_consecutive", {}),
        ),
        "the schedule review finished early and every question got answered well before the "
        "closing remarks and that is all",
    ),
    (
        "number_words[A]+first_word_answer[C]",
        (
            Constraint("length_constraints:number_words", {"num_words": 10, "relation": "at least"}),
            Constraint("first_word:first_word_answer", {"first_word": "however"}),
        ),
        "however the storm passed quickly and the sky cleared before noon today",
    ),
]


# ---------------------------------------------------------------------------
# (a) fixtures are jointly feasible to start with
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,pair,text", MULTI_CASES)
def test_fixture_satisfies_both_targets(name: str, pair: tuple[Constraint, Constraint], text: str) -> None:
    assert feasible(pair, text) is True, f"{name}: fixture does not satisfy both constraints: {text!r}"


# ---------------------------------------------------------------------------
# (b) corrupt_set flips exactly the two targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,pair,text", MULTI_CASES)
def test_corrupt_set_fails_exactly_both_targets(name: str, pair: tuple[Constraint, Constraint], text: str) -> None:
    corrupted = registry.corrupt_set(pair, (0, 1), text, random.Random(0))
    assert corrupted is not None, f"{name}: corrupt_set produced no composed candidate"
    assert check(pair[0], corrupted) is False, f"{name}: target 0 ({pair[0].iid}) still passes"
    assert check(pair[1], corrupted) is False, f"{name}: target 1 ({pair[1].iid}) still passes"
    ok, reason = registry.exact_set(pair, (0, 1), text, corrupted)
    assert ok is True, f"{name}: exact_set rejected corrupt_set's own candidate ({reason})"
    assert reason == registry.EXACT


# ---------------------------------------------------------------------------
# (c) locality: a double-space marker unrelated to either target survives
# ---------------------------------------------------------------------------


def _inject_double_space(text: str) -> tuple[str, str] | None:
    """Double `text`'s 3rd/4th-word gap; return (variant, marker substring)."""
    words = list(re.finditer(r"\S+", text))
    if len(words) < 4:
        return None
    third, fourth = words[2], words[3]
    marker = text[third.start() : third.end()] + "  " + text[fourth.start() : fourth.end()]
    pos = third.end()
    return text[:pos] + " " + text[pos:], marker


def _inject_divider(text: str) -> tuple[str, str] | None:
    """Insert a blank-line paragraph divider at the 6th/7th-word gap (distinct
    from the 3rd/4th-word gap `_inject_double_space` uses, since none of
    MULTI_CASES' single-sentence fixtures carry a ". Uppercase" sentence
    boundary for tests/test_operators_locality.py's `_divider_edit` heuristic
    to anchor on); return (variant, marker substring). None when `text` has
    fewer than 7 words."""
    words = list(re.finditer(r"\S+", text))
    if len(words) < 7:
        return None
    sixth, seventh = words[5], words[6]
    marker = text[sixth.start() : sixth.end()] + "\n\n" + text[seventh.start() : seventh.end()]
    return text[: sixth.end()] + "\n\n" + text[seventh.start() :], marker


@pytest.mark.parametrize("name,pair,text", MULTI_CASES)
def test_corrupt_set_preserves_an_unrelated_whitespace_marker(
    name: str, pair: tuple[Constraint, Constraint], text: str
) -> None:
    injected = _inject_double_space(text)
    if injected is None:
        pytest.skip(f"{name}: fixture too short to inject a locality marker")
    variant, marker = injected
    assert feasible(pair, variant) is True, f"{name}: whitespace-injected variant broke a target"

    corrupted = registry.corrupt_set(pair, (0, 1), variant, random.Random(0))
    assert corrupted is not None, f"{name}: corrupt_set produced no composed candidate on the variant"
    assert marker in corrupted, f"{name}: locality marker {marker!r} destroyed by corrupt_set"


@pytest.mark.parametrize("name,pair,text", MULTI_CASES)
def test_corrupt_set_preserves_an_unrelated_divider_marker(
    name: str, pair: tuple[Constraint, Constraint], text: str
) -> None:
    injected = _inject_divider(text)
    if injected is None:
        pytest.skip(f"{name}: fixture too short to inject a divider marker")
    variant, marker = injected
    assert feasible(pair, variant) is True, f"{name}: divider-injected variant broke a target"

    corrupted = registry.corrupt_set(pair, (0, 1), variant, random.Random(0))
    assert corrupted is not None, f"{name}: corrupt_set produced no composed candidate on the variant"
    assert marker in corrupted, f"{name}: divider marker {marker!r} destroyed by corrupt_set"


# ---------------------------------------------------------------------------
# (d) "passes the rest": a genuine third, untargeted constraint survives
# ---------------------------------------------------------------------------


def test_corrupt_set_leaves_a_bystander_constraint_passing() -> None:
    triple = (
        Constraint("detectable_format:title", {}),
        Constraint("punctuation:punctuation_exclamation", {}),
        Constraint("length_constraints:number_words", {"num_words": 5, "relation": "at least"}),
    )
    text = "<<field notes>> gentle rivers wander through quiet valleys near old stone bridges each morning"
    assert feasible(triple, text) is True

    corrupted = registry.corrupt_set(triple, (0, 1), text, random.Random(0))
    assert corrupted is not None

    verdicts = tuple(check(c, corrupted) for c in triple)
    assert verdicts[0] is False  # target 0 (title) fails
    assert verdicts[1] is False  # target 1 (punctuation_exclamation) fails
    assert verdicts[2] is True  # bystander (number_words) still passes

    ok, reason = registry.exact_set(triple, (0, 1), text, corrupted)
    assert ok is True, f"exact_set rejected a candidate whose bystander verdict is True ({reason})"


# ---------------------------------------------------------------------------
# (e) a NO_OPERATOR target makes corrupt_set return None
# ---------------------------------------------------------------------------


def test_corrupt_set_returns_none_when_a_target_has_no_operator() -> None:
    pair = (
        Constraint("language:response_language", {"language": "en"}),
        Constraint("keywords:existence", {"keywords": ["apple"]}),
    )
    text = "the apple orchard grows quietly beside the river every single spring season"
    assert "language:response_language" in registry.NO_OPERATOR

    corrupted = registry.corrupt_set(pair, (0, 1), text, random.Random(0))
    assert corrupted is None


# ---------------------------------------------------------------------------
# (f) exact_set rejects a candidate that also breaks a non-target constraint
# ---------------------------------------------------------------------------


def test_exact_set_rejects_collateral_damage_on_a_non_target() -> None:
    triple = (
        Constraint("keywords:word_once", {"keyword": "unique"}),
        Constraint("punctuation:no_comma", {}),
        Constraint("length_constraints:number_words", {"num_words": 5, "relation": "at least"}),
    )
    original = "a longer unrelated placeholder sentence used only as the unchanged-check baseline"
    corrupted = "hi,"

    # sanity: corrupted breaks all three, including the non-target
    assert check(triple[0], corrupted) is False  # "unique" absent -> word_once fails (target)
    assert check(triple[1], corrupted) is False  # comma present -> no_comma fails (target)
    assert check(triple[2], corrupted) is False  # 1 word -> number_words>=5 fails (bystander, collateral)

    ok, reason = registry.exact_set(triple, (0, 1), original, corrupted)
    assert ok is False
    assert reason == f"{registry.COLLATERAL}:length_constraints:number_words"


def test_exact_set_unchanged_when_corrupted_equals_original() -> None:
    pair = (
        Constraint("punctuation:no_comma", {}),
        Constraint("keywords:word_once", {"keyword": "unique"}),
    )
    text = "the report highlights one unique finding without any repeated commas here"
    ok, reason = registry.exact_set(pair, (0, 1), text, text)
    assert ok is False
    assert reason == registry.UNCHANGED


def test_exact_set_target_still_passes() -> None:
    pair = (
        Constraint("punctuation:no_comma", {}),
        Constraint("keywords:word_once", {"keyword": "unique"}),
    )
    original = "placeholder baseline text that differs from the corrupted string below"
    # both targets still pass in `corrupted` (no comma, "unique" once)
    corrupted = "the report highlights one unique finding without any repeated commas here"
    ok, reason = registry.exact_set(pair, (0, 1), original, corrupted)
    assert ok is False
    assert reason == registry.TARGET_STILL_PASSES


# ---------------------------------------------------------------------------
# (g) empty target_indices is rejected explicitly, not silently mishandled.
# sample_family_set's "nothing qualifies" sentinel is this same empty tuple;
# a caller that forwards it without an `if target_indices:` guard must get a
# named contract violation, not an opaque exception from a private helper.
# ---------------------------------------------------------------------------


def test_corrupt_set_raises_on_empty_target_indices() -> None:
    pair = (
        Constraint("punctuation:no_comma", {}),
        Constraint("keywords:word_once", {"keyword": "unique"}),
    )
    text = "the report highlights one unique finding without any repeated commas here"
    with pytest.raises(ValueError, match="target_indices must be non-empty"):
        registry.corrupt_set(pair, (), text, random.Random(0))


def test_exact_set_raises_on_empty_target_indices() -> None:
    pair = (
        Constraint("punctuation:no_comma", {}),
        Constraint("keywords:word_once", {"keyword": "unique"}),
    )
    original = "the report highlights one unique finding without any repeated commas here"
    corrupted = "hi,"
    with pytest.raises(ValueError, match="target_indices must be non-empty"):
        registry.exact_set(pair, (), original, corrupted)

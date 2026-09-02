"""Tests for conpo/operators/ifeval_b.py (12 IFEval constraint families).

(a) each family's fixture passes its own check() for two rng seeds.
(b) each family's operator produces an EXACT single-target corruption whose
    corrupted text differs from the fixture in at most 30% of whitespace
    tokens (a quick token-level diff computed here, not conpo.distance).
(c) four two-constraint prompts built from families in this group, both
    directions of registry.corrupt() returning EXACT.
(d) every family id is registered in registry.OPERATORS and registry.FIXTURES.
"""

import difflib
import json
import random
from pathlib import Path

import pytest

import conpo.operators.ifeval_b  # noqa: F401  (registers the 12 families)
from conpo.operators import registry
from conpo.verifiers import Constraint, check, feasible

FAMILIES: tuple[str, ...] = (
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections",
    "detectable_format:json_format",
    "detectable_format:title",
    "combination:two_responses",
    "combination:repeat_prompt",
    "startend:end_checker",
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "punctuation:no_comma",
    "startend:quotation",
)

FIXTURE_KWARGS: dict[str, dict[str, object]] = json.loads(
    (Path(__file__).parent / "fixtures" / "family_kwargs.json").read_text()
)


def _token_diff_ratio(a: str, b: str) -> float:
    """Fraction of whitespace tokens that differ between `a` and `b`."""
    ta, tb = a.split(), b.split()
    return 1 - difflib.SequenceMatcher(None, ta, tb).ratio()


# (a) fixtures pass their own check(), for two seeds -------------------------


@pytest.mark.parametrize("iid", FAMILIES)
def test_fixture_passes_for_two_seeds(iid: str) -> None:
    c = Constraint.from_ground_truth(iid, FIXTURE_KWARGS[iid])
    fixture = registry.FIXTURES[iid]
    for seed in (0, 1):
        text = fixture(c, random.Random(seed))
        assert check(c, text) is True, f"{iid} seed={seed}: {text!r}"


# (b) operators produce an exact, minimal single-target corruption -----------


@pytest.mark.parametrize("iid", FAMILIES)
def test_operator_corrupts_target_with_minimal_edit(iid: str) -> None:
    c = Constraint.from_ground_truth(iid, FIXTURE_KWARGS[iid])
    fixture = registry.FIXTURES[iid]
    text = fixture(c, random.Random(0))

    corrupted, reason = registry.corrupt((c,), 0, text, random.Random(0))

    assert corrupted is not None, f"{iid}: {reason}"
    assert reason == registry.EXACT
    assert check(c, corrupted) is False
    assert _token_diff_ratio(text, corrupted) <= 0.30


def test_capital_word_frequency_less_than_relation() -> None:
    """A second setting for capital_word_frequency: "less than" exercises the
    operator's other direction (raise the capital count instead of lowering it)."""
    c = Constraint.from_ground_truth(
        "change_case:capital_word_frequency", {"capital_frequency": 3, "capital_relation": "less than"}
    )
    fixture = registry.FIXTURES[c.iid]
    for seed in (0, 1):
        text = fixture(c, random.Random(seed))
        assert check(c, text) is True

    text = fixture(c, random.Random(0))
    corrupted, reason = registry.corrupt((c,), 0, text, random.Random(0))
    assert corrupted is not None and reason == registry.EXACT
    assert check(c, corrupted) is False


# (c) two-constraint prompts, both directions EXACT ---------------------------


def _assert_pair_exact(c0: Constraint, c1: Constraint, text: str) -> None:
    both = (c0, c1)
    assert feasible(both, text) is True
    corrupted0, reason0 = registry.corrupt(both, 0, text, random.Random(3))
    assert corrupted0 is not None and reason0 == registry.EXACT
    corrupted1, reason1 = registry.corrupt(both, 1, text, random.Random(3))
    assert corrupted1 is not None and reason1 == registry.EXACT


def test_pair_no_comma_and_quotation() -> None:
    c0 = Constraint.from_ground_truth("punctuation:no_comma", {})
    c1 = Constraint.from_ground_truth("startend:quotation", {})
    prose = registry.FIXTURES["punctuation:no_comma"](c0, random.Random(5))
    text = f'"{prose}"'
    _assert_pair_exact(c0, c1, text)


def test_pair_title_and_two_responses() -> None:
    c0 = Constraint.from_ground_truth("detectable_format:title", {})
    c1 = Constraint.from_ground_truth("combination:two_responses", {})
    base = registry.FIXTURES["combination:two_responses"](c1, random.Random(11))
    text = base.replace("\n******\n", "\n<<my title>>\n******\n", 1)
    _assert_pair_exact(c0, c1, text)


def test_pair_english_lowercase_and_no_comma() -> None:
    c0 = Constraint.from_ground_truth("change_case:english_lowercase", {})
    c1 = Constraint.from_ground_truth("punctuation:no_comma", {})
    text = registry.FIXTURES["change_case:english_lowercase"](c0, random.Random(2))
    _assert_pair_exact(c0, c1, text)


def test_pair_highlighted_sections_and_capital_word_frequency() -> None:
    c0 = Constraint.from_ground_truth("detectable_format:number_highlighted_sections", {"num_highlights": 2})
    c1 = Constraint.from_ground_truth(
        "change_case:capital_word_frequency", {"capital_frequency": 2, "capital_relation": "at least"}
    )
    text = "This report covers three items. *point one* and *point two* stand out. WORD0 WORD1 remain unresolved."
    _assert_pair_exact(c0, c1, text)


# (d) every family id is registered -------------------------------------------


def test_every_family_registered() -> None:
    for iid in FAMILIES:
        assert iid in registry.OPERATORS
        assert iid in registry.FIXTURES

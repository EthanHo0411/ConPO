"""Tests for conpo/operators/ifeval_a.py: fixtures, corruption, exactness."""

import difflib
import json
import random
from pathlib import Path

import pytest

import conpo.operators.ifeval_a  # noqa: F401  (registers the 12 families on import)
from conpo.operators import registry
from conpo.verifiers import Constraint, check, feasible

FAMILY_IDS: tuple[str, ...] = (
    "keywords:existence",
    "keywords:frequency",
    "keywords:forbidden_words",
    "keywords:letter_frequency",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:number_words",
    "length_constraints:nth_paragraph_first_word",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:number_bullet_lists",
    "detectable_format:constrained_response",
)

KWARGS = json.loads((Path(__file__).parent / "fixtures" / "family_kwargs.json").read_text())


def _token_diff_ratio(a: str, b: str) -> float:
    """Fraction of whitespace tokens changed between `a` and `b`."""
    ta, tb = a.split(), b.split()
    matcher = difflib.SequenceMatcher(a=ta, b=tb)
    changed = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")
    return changed / max(len(ta), len(tb), 1)


# ---------------------------------------------------------------------------
# (d) registration
# ---------------------------------------------------------------------------


def test_every_family_registered() -> None:
    for iid in FAMILY_IDS:
        assert iid in registry.OPERATORS
        assert iid in registry.FIXTURES


# ---------------------------------------------------------------------------
# (a) fixtures pass for two different seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iid", FAMILY_IDS)
def test_fixture_passes_for_two_seeds(iid: str) -> None:
    c = Constraint.from_ground_truth(iid, KWARGS[iid])
    for seed in (0, 1):
        text = registry.FIXTURES[iid](c, random.Random(seed))
        assert check(c, text) is True, f"{iid} fixture failed for seed {seed}: {text!r}"


# ---------------------------------------------------------------------------
# (b) corruption fails the target and stays a small edit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iid", FAMILY_IDS)
def test_corrupt_fails_target_and_stays_minimal(iid: str) -> None:
    c = Constraint.from_ground_truth(iid, KWARGS[iid])
    rng = random.Random(0)
    fixture_text = registry.FIXTURES[iid](c, rng)
    assert check(c, fixture_text) is True

    corrupted, reason = registry.corrupt((c,), 0, fixture_text, random.Random(1))
    assert corrupted is not None, f"{iid} produced no corruption ({reason})"
    assert reason == registry.EXACT
    assert check(c, corrupted) is False
    ratio = _token_diff_ratio(fixture_text, corrupted)
    assert ratio <= 0.30, f"{iid} corruption changed {ratio:.0%} of tokens: {corrupted!r}"


# ---------------------------------------------------------------------------
# (c) two-constraint prompts: exactness targets each constraint in turn
# ---------------------------------------------------------------------------

TWO_CONSTRAINT_CASES = [
    (
        Constraint("keywords:existence", {"keywords": ["apple", "river"]}),
        Constraint("length_constraints:number_words", {"num_words": 20, "relation": "at least"}),
        "The apple orchard sits beside the quiet river valley where visitors often walk "
        "during the cool autumn afternoon each year.",
    ),
    (
        Constraint("keywords:forbidden_words", {"forbidden_words": ["banana", "grape"]}),
        Constraint("detectable_format:constrained_response", {}),
        "My answer is yes. The committee reviewed the proposal and found no major "
        "concerns worth reporting today.",
    ),
    (
        Constraint("detectable_content:number_placeholders", {"num_placeholders": 2}),
        Constraint("detectable_content:postscript", {"postscript_marker": "P.S."}),
        "Please provide [name] and [date] so we can finalize the schedule.\n\n"
        "P.S. Thanks for confirming everything soon.",
    ),
    (
        Constraint("detectable_format:number_bullet_lists", {"num_bullets": 2}),
        Constraint("length_constraints:number_paragraphs", {"num_paragraphs": 2}),
        "Here is the plan for tomorrow.\n***\n"
        "* Confirm the venue booking.\n* Send out the invitations today.",
    ),
]


@pytest.mark.parametrize("first,second,text", TWO_CONSTRAINT_CASES)
def test_two_constraint_prompts_corrupt_exactly(first: Constraint, second: Constraint, text: str) -> None:
    both = (first, second)
    assert feasible(both, text) is True, f"hand-built fixture does not satisfy both constraints: {text!r}"

    corrupted0, reason0 = registry.corrupt(both, 0, text, random.Random(0))
    assert reason0 == registry.EXACT, f"target 0 ({first.iid}) corruption reason: {reason0}"
    assert corrupted0 is not None

    corrupted1, reason1 = registry.corrupt(both, 1, text, random.Random(0))
    assert reason1 == registry.EXACT, f"target 1 ({second.iid}) corruption reason: {reason1}"
    assert corrupted1 is not None


def test_at_least_four_pairs_covered() -> None:
    assert len(TWO_CONSTRAINT_CASES) >= 4

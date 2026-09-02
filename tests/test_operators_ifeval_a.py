"""Tests for conpo/operators/ifeval_a.py: fixtures, corruption, exactness."""

import difflib
import json
import random
import re
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


# ---------------------------------------------------------------------------
# regression tests for previously-open review findings
# ---------------------------------------------------------------------------


def test_keyword_frequency_less_than_the_terminates() -> None:
    """keyword='the' with relation='less than' used to hang forever: the fixture's
    hardcoded 'the topic' replacement reintroduces 'the' on every substitution, so
    the occurrence count never dropped below the threshold."""
    c = Constraint("keywords:frequency", {"keyword": "the", "frequency": 1, "relation": "less than"})
    text = registry.FIXTURES["keywords:frequency"](c, random.Random(0))
    assert check(c, text) is True, f"fixture violates its own constraint: {text!r}"

    corrupted, reason = registry.corrupt((c,), 0, text, random.Random(0))
    assert corrupted is not None, f"produced no corruption ({reason})"
    assert reason == registry.EXACT
    assert check(c, corrupted) is False


def test_forbidden_words_fixture_avoids_filler_collisions() -> None:
    """The fixture used to ignore the Constraint entirely and return raw filler
    text, which contains everyday words (e.g. 'garden', 'morning') that can
    themselves be the forbidden words."""
    c = Constraint("keywords:forbidden_words", {"forbidden_words": ["garden", "morning"]})
    for seed in (0, 1):
        text = registry.FIXTURES["keywords:forbidden_words"](c, random.Random(seed))
        assert check(c, text) is True, f"fixture contains a forbidden word: {text!r}"


def test_postscript_minimal_candidate_first() -> None:
    """The marker-rename edit ('P.S' -> 'Note', keeping the marker's own
    trailing period in place) is smaller than stripping the whole postscript
    line, so it must be yielded first: registry.corrupt returns the first
    candidate that passes exact()."""
    c = Constraint("detectable_content:postscript", {"postscript_marker": "P.S."})
    text = registry.FIXTURES["detectable_content:postscript"](c, random.Random(0))
    assert check(c, text) is True

    first = next(registry.candidates(text, c, random.Random(0)))
    expected = text.replace("P.S", "Note", 1)
    assert first == expected, f"expected the minimal marker-rename candidate first, got: {first!r}"
    assert check(c, first) is False

    corrupted, reason = registry.corrupt((c,), 0, text, random.Random(0))
    assert reason == registry.EXACT
    assert corrupted == expected


# ---------------------------------------------------------------------------
# locality regressions: an operator may touch only the span its own family
# needs, never unrelated whitespace elsewhere in the response. Each test below
# injects a double space and a blank-line divider that sit outside the
# operator's own target span and asserts both survive verbatim.
# ---------------------------------------------------------------------------


def _inject_locality_markers(text: str) -> tuple[str, str, str]:
    """Insert a double space after the 3rd word and a blank-line divider
    before the word following the LAST '. Uppercase' boundary. Returns
    (variant, double-space marker, divider marker); both markers are expected
    to survive a locality-preserving corruption untouched."""
    words = list(re.finditer(r"\S+", text))
    ds_pos = words[2].end()
    ds_marker = text[words[2].start() : words[2].end()] + "  " + text[words[3].start() : words[3].end()]

    div_matches = list(re.finditer(r"\. ([A-Z]\w*)", text))
    dm = div_matches[-1]
    div_start = dm.start() + 1
    div_marker = ".\n\n" + dm.group(1)

    edits = sorted([(ds_pos, ds_pos, " "), (div_start, div_start + 1, "\n\n")], key=lambda e: e[0], reverse=True)
    variant = text
    for start, end, repl in edits:
        variant = variant[:start] + repl + variant[end:]
    return variant, ds_marker, div_marker


def test_number_sentences_preserves_irregular_whitespace() -> None:
    """Regression: the 'at least' branch used to rebuild the whole response
    with ' '.join(merged sentences), normalising every sentence boundary's
    whitespace, not just the one boundary it merged."""
    iid = "length_constraints:number_sentences"
    c = Constraint.from_ground_truth(iid, KWARGS[iid])
    text = registry.FIXTURES[iid](c, random.Random(0))
    assert check(c, text) is True

    variant, ds_marker, div_marker = _inject_locality_markers(text)
    assert check(c, variant) is True

    corrupted, reason = registry.corrupt((c,), 0, variant, random.Random(1))
    assert corrupted is not None, f"produced no corruption ({reason})"
    assert reason == registry.EXACT
    assert check(c, corrupted) is False
    assert ds_marker in corrupted, f"double space destroyed: {corrupted!r}"
    assert div_marker in corrupted, f"paragraph divider destroyed: {corrupted!r}"


def test_postscript_preserves_irregular_whitespace() -> None:
    """Regression: renaming the whole 'P.S.' marker, trailing period
    included, used to eat a paragraph divider that happened to sit right
    after it."""
    iid = "detectable_content:postscript"
    c = Constraint.from_ground_truth(iid, KWARGS[iid])
    text = registry.FIXTURES[iid](c, random.Random(0))
    assert check(c, text) is True

    variant, ds_marker, div_marker = _inject_locality_markers(text)
    assert check(c, variant) is True

    corrupted, reason = registry.corrupt((c,), 0, variant, random.Random(0))
    assert corrupted is not None, f"produced no corruption ({reason})"
    assert reason == registry.EXACT
    assert check(c, corrupted) is False
    assert ds_marker in corrupted, f"double space destroyed: {corrupted!r}"
    assert div_marker in corrupted, f"paragraph divider destroyed: {corrupted!r}"


def test_letter_frequency_preserves_irregular_whitespace() -> None:
    """Regression: trimming the target letter used to leave a double space
    behind wherever a whole one-letter word (the standalone 'a') was
    deleted, colliding with an unrelated marker elsewhere in the text."""
    iid = "keywords:letter_frequency"
    c = Constraint.from_ground_truth(iid, KWARGS[iid])
    text = registry.FIXTURES[iid](c, random.Random(0))
    assert check(c, text) is True

    variant, ds_marker, div_marker = _inject_locality_markers(text)
    assert check(c, variant) is True

    corrupted, reason = registry.corrupt((c,), 0, variant, random.Random(1))
    assert corrupted is not None, f"produced no corruption ({reason})"
    assert reason == registry.EXACT
    assert check(c, corrupted) is False
    assert ds_marker in corrupted, f"double space destroyed: {corrupted!r}"
    assert div_marker in corrupted, f"paragraph divider destroyed: {corrupted!r}"

"""Tests for the group-D IFBench-Train operators and fixtures (conpo/operators/iftrain_d.py)."""

import difflib
import json
import random
from pathlib import Path

import pytest

from conpo.operators import iftrain_d  # noqa: F401  (registers on import)
from conpo.operators import registry
from conpo.verifiers import Constraint, check, feasible

FAMILIES: tuple[str, ...] = (
    "detectable_format:bigram_wrapping",
    "copy:copying_simple",
    "copy:copying_multiple",
    "punctuation:punctuation_dot",
    "punctuation:punctuation_exclamation",
    "count:lowercase_counting",
    "letters:letter_counting",
    "letters:letter_counting2",
    "count:counting_composition",
    "count:count_unique",
    "count:count_increment_word",
    "keywords:palindrome",
    "keywords:keyword_specific_position",
    "keywords:start_end",
)

_KWARGS = json.loads((Path(__file__).parent / "fixtures" / "family_kwargs.json").read_text())

# A second kwargs setting for the families where it matters most: exact
# boundary counters (letter_counting both directions), the highest-collateral
# composition family, and the substring-count family.
_SECOND_KWARGS: dict[str, dict] = {
    "letters:letter_counting": {"N": 150, "relation": "less than"},
    "letters:letter_counting2": {"letter": "e", "let_frequency": 2, "let_relation": "less than"},
    "count:counting_composition": {"n_sent": 1, "n_words": 4},
    "count:count_increment_word": {"keyword1": "lantern", "keyword2": "compass"},
    "count:lowercase_counting": {"N": 0},
    "copy:copying_multiple": {"prompt_to_repeat": "Say this back exactly.", "N": 3},
    "keywords:keyword_specific_position": {"keyword": "ocean", "n": 2, "m": 3},
}


def _constraint(iid: str, second: bool = False) -> Constraint:
    kwargs = _SECOND_KWARGS.get(iid, _KWARGS[iid]) if second else _KWARGS[iid]
    return Constraint(iid, kwargs)


def _token_diff_fraction(a: str, b: str) -> float:
    """Fraction of whitespace tokens that differ between two strings (0 = identical)."""
    ta, tb = a.split(), b.split()
    matcher = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    changed = max(len(ta), len(tb)) - matched
    return changed / max(len(ta), len(tb), 1)


# ---------------------------------------------------------------------------
# (a) every family's fixture passes its own verifier, for two different seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iid", FAMILIES)
def test_fixture_passes_for_two_seeds(iid: str) -> None:
    c = _constraint(iid)
    fixture = registry.FIXTURES[iid]
    assert check(c, fixture(c, random.Random(1)))
    assert check(c, fixture(c, random.Random(2)))


@pytest.mark.parametrize("iid", sorted(_SECOND_KWARGS))
def test_fixture_passes_with_second_kwargs(iid: str) -> None:
    c = _constraint(iid, second=True)
    fixture = registry.FIXTURES[iid]
    assert check(c, fixture(c, random.Random(1)))
    assert check(c, fixture(c, random.Random(2)))


# ---------------------------------------------------------------------------
# (b) every family's operator produces an exact, span-local corruption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iid", FAMILIES)
def test_corrupt_flips_only_target(iid: str) -> None:
    c = _constraint(iid)
    fixture_text = registry.FIXTURES[iid](c, random.Random(1))
    assert check(c, fixture_text)

    corrupted, reason = registry.corrupt((c,), 0, fixture_text, random.Random(3))

    assert corrupted is not None, f"{iid}: no exact corruption found ({reason})"
    assert reason == registry.EXACT
    assert not check(c, corrupted)
    assert _token_diff_fraction(fixture_text, corrupted) <= 0.30


@pytest.mark.parametrize("iid", sorted(_SECOND_KWARGS))
def test_corrupt_flips_only_target_with_second_kwargs(iid: str) -> None:
    c = _constraint(iid, second=True)
    fixture_text = registry.FIXTURES[iid](c, random.Random(1))
    assert check(c, fixture_text)

    corrupted, reason = registry.corrupt((c,), 0, fixture_text, random.Random(3))

    assert corrupted is not None, f"{iid}: no exact corruption found ({reason})"
    assert reason == registry.EXACT
    assert not check(c, corrupted)
    assert _token_diff_fraction(fixture_text, corrupted) <= 0.30


# ---------------------------------------------------------------------------
# (c) two-constraint prompts: exactness holds targeting each family in turn
# ---------------------------------------------------------------------------


def _pair_dot_exclamation() -> tuple[tuple[Constraint, Constraint], str]:
    constraints = (
        Constraint("punctuation:punctuation_dot", {}),
        Constraint("punctuation:punctuation_exclamation", {}),
    )
    text = registry.FIXTURES["punctuation:punctuation_dot"](constraints[0], random.Random(1))
    return constraints, text


def _pair_palindrome_start_end() -> tuple[tuple[Constraint, Constraint], str]:
    constraints = (
        Constraint("keywords:palindrome", {}),
        Constraint("keywords:start_end", {}),
    )
    text = (
        "Morning arrived quietly near the wooden dock while the tide stayed level with the worn "
        "stone steps leading up toward the village until visitors returning home this calm autumn morning"
    )
    return constraints, text


def _pair_unique_start_end() -> tuple[tuple[Constraint, Constraint], str]:
    constraints = (
        Constraint("count:count_unique", {}),
        Constraint("keywords:start_end", {}),
    )
    text = (
        "Morning brings wandering travelers who discover hidden paths winding through quiet forests "
        "beyond ancient stone bridges where gentle streams reflect golden autumn light before evening "
        "arrives across distant hills this calm and pleasant morning"
    )
    return constraints, text


def _pair_bigram_lowercase() -> tuple[tuple[Constraint, Constraint], str]:
    constraints = (
        Constraint("detectable_format:bigram_wrapping", {}),
        Constraint("count:lowercase_counting", {"N": 3}),
    )
    words = [
        "This", "Report", "Covers", "Recent", "Weather", "Patterns", "Across", "The",
        "Northern", "Region", "During", "Early", "spring", "months", "while", "Farmers",
    ]
    pairs = [f"<<{words[i]} {words[i + 1]}>>" for i in range(0, len(words), 2)]
    text = " ".join(pairs)
    return constraints, text


TWO_CONSTRAINT_PAIRS = (
    _pair_dot_exclamation(),
    _pair_palindrome_start_end(),
    _pair_unique_start_end(),
    _pair_bigram_lowercase(),
)


@pytest.mark.parametrize("constraints,text", TWO_CONSTRAINT_PAIRS)
def test_two_constraint_pair_is_feasible(constraints: tuple[Constraint, Constraint], text: str) -> None:
    assert feasible(constraints, text)


@pytest.mark.parametrize("constraints,text", TWO_CONSTRAINT_PAIRS)
def test_two_constraint_pair_corrupts_each_target_exactly(
    constraints: tuple[Constraint, Constraint], text: str
) -> None:
    for target_index in (0, 1):
        corrupted, reason = registry.corrupt(constraints, target_index, text, random.Random(target_index))
        assert reason == registry.EXACT, (
            f"target={constraints[target_index].iid} in ({constraints[0].iid}, {constraints[1].iid}): {reason}"
        )
        assert corrupted is not None


# ---------------------------------------------------------------------------
# (d) every family in this group is registered
# ---------------------------------------------------------------------------


def test_all_families_registered() -> None:
    for iid in FAMILIES:
        assert iid in registry.OPERATORS, f"{iid} missing from registry.OPERATORS"
        assert iid in registry.FIXTURES, f"{iid} missing from registry.FIXTURES"


# ---------------------------------------------------------------------------
# regression: two review-flagged bugs, fixed 2026-09-02
# ---------------------------------------------------------------------------


def test_keyword_specific_position_fixture_at_m_equals_one() -> None:
    """m=1 puts the keyword itself first; it must stay in its given case, not
    get capitalized (KeywordSpecificPositionChecker compares case-sensitively)."""
    c = Constraint("keywords:keyword_specific_position", {"keyword": "river", "n": 1, "m": 1})
    fixture = registry.FIXTURES["keywords:keyword_specific_position"]
    assert check(c, fixture(c, random.Random(1)))
    assert check(c, fixture(c, random.Random(2)))

    fixture_text = fixture(c, random.Random(1))
    corrupted, reason = registry.corrupt((c,), 0, fixture_text, random.Random(3))
    assert reason == registry.EXACT
    assert corrupted is not None


def test_letter_counting_second_candidate_survives_multi_delete() -> None:
    """The forward-scan candidate used to delete-while-iterating a list, which
    shifts indices out from under itself once more than one deletion is needed."""
    c = Constraint("letters:letter_counting", {"N": 5, "relation": "at least"})
    response = " ".join(["alpha"] * 6)  # 30 letters, well above N=5
    op = registry.OPERATORS["letters:letter_counting"]
    candidates = list(op(response, c, random.Random(0)))
    assert len(candidates) == 2
    for cand in candidates:
        assert not check(c, cand)


def test_letter_counting2_second_candidate_survives_multi_delete() -> None:
    c = Constraint("letters:letter_counting2", {"letter": "a", "let_frequency": 3, "let_relation": "at least"})
    response = " ".join(["banana"] * 5)  # 15 a's, well above let_frequency=3
    op = registry.OPERATORS["letters:letter_counting2"]
    candidates = list(op(response, c, random.Random(0)))
    assert len(candidates) == 2
    for cand in candidates:
        assert not check(c, cand)


# ---------------------------------------------------------------------------
# regression: two more review-flagged bugs, fixed 2026-09-02
# ---------------------------------------------------------------------------


def test_counting_composition_fixture_at_n_words_equals_one() -> None:
    """n_words=1 must yield a single-token sentence ('.'), not a 2-token one;
    the old max(n_words - 1, 1) floor forced an extra content word in."""
    c = Constraint("count:counting_composition", {"n_sent": 1, "n_words": 1})
    fixture = registry.FIXTURES["count:counting_composition"]
    assert check(c, fixture(c, random.Random(1)))
    assert check(c, fixture(c, random.Random(2)))

    fixture_text = fixture(c, random.Random(1))
    corrupted, reason = registry.corrupt((c,), 0, fixture_text, random.Random(3))
    assert reason == registry.EXACT
    assert corrupted is not None
    assert not check(c, corrupted)


def test_keyword_specific_position_targets_correct_occurrence() -> None:
    """The keyword must also appear earlier in the response, before its
    required sentence/position; the operator must edit the checker-read
    occurrence (sentence n-1, word m-1), not the first substring match."""
    c = Constraint("keywords:keyword_specific_position", {"keyword": "river", "n": 2, "m": 3})
    response = "I once visited a beautiful river near my home. Gentle wind river moved slowly today."
    assert check(c, response)

    corrupted, reason = registry.corrupt((c,), 0, response, random.Random(3))
    assert corrupted is not None, f"no exact corruption found ({reason})"
    assert reason == registry.EXACT
    assert not check(c, corrupted)

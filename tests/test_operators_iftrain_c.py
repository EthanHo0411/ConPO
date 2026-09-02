"""Tests for the group-C IFBench-Train operators and fixtures (conpo/operators/iftrain_c.py)."""

import difflib
import json
import random
from pathlib import Path

import pytest

from conpo.operators import iftrain_c  # noqa: F401  (registers on import)
from conpo.operators import registry
from conpo.verifiers import Constraint, check, feasible

FAMILIES: tuple[str, ...] = (
    "copy:repeat_phrase",
    "copy:copy",
    "new:copy_span_idx",
    "detectable_format:sentence_hyphens",
    "keywords:no_adjacent_consecutive",
    "detectable_format:square_brackets",
    "keywords:word_once",
    "keywords:word_count_different_numbers",
    "keywords:exclude_word_harder",
    "paragraphs:paragraphs",
    "paragraphs:paragraphs2",
    "first_word:first_word_sent",
    "first_word:first_word_answer",
    "last_word:last_word_sent",
    "last_word:last_word_answer",
)

_KWARGS = json.loads((Path(__file__).parent / "fixtures" / "family_kwargs.json").read_text())

# new:copy_span_idx's shared JSON kwargs extract a 2-word span; any single-token
# corruption of a 2-word fixture moves 30-50% of its tokens by construction, so
# the corruption-distance check (b) below uses a longer span for that one family.
_SPAN_PROMPT = (
    "The morning market opened early and vendors arranged colorful stalls along the narrow "
    "cobblestone street before sunrise."
)
_SPAN_START = _SPAN_PROMPT.index("market")
_SPAN_END = _SPAN_PROMPT.index("narrow") + len("narrow")
_CORRUPT_KWARGS = {
    "new:copy_span_idx": {"prompt_to_repeat": _SPAN_PROMPT, "n_start": _SPAN_START, "n_end": _SPAN_END},
}


def _constraint(iid: str, for_corruption: bool = False) -> Constraint:
    kwargs = _CORRUPT_KWARGS.get(iid, _KWARGS[iid]) if for_corruption else _KWARGS[iid]
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


# ---------------------------------------------------------------------------
# (b) every family's operator produces an exact, span-local corruption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iid", FAMILIES)
def test_corrupt_flips_only_target(iid: str) -> None:
    c = _constraint(iid, for_corruption=True)
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


def _pair_a() -> tuple[tuple[Constraint, Constraint], str]:
    constraints = (
        Constraint("first_word:first_word_answer", {"first_word": "however"}),
        Constraint("last_word:last_word_answer", {"last_word": "today"}),
    )
    text = (
        "However the committee reviewed every proposal carefully before reaching a final decision "
        "that satisfied nearly everyone involved today"
    )
    return constraints, text


def _pair_b() -> tuple[tuple[Constraint, Constraint], str]:
    constraints = (
        Constraint("keywords:word_once", {"keyword": "orchid"}),
        Constraint("keywords:exclude_word_harder", {"keyword": "cactus"}),
    )
    text = (
        "The botanical garden featured a rare orchid blooming beside the greenhouse, drawing "
        "photographers from across the city who wanted to capture its delicate petals in the soft "
        "afternoon light before the exhibit closed for the season next week."
    )
    return constraints, text


def _pair_c() -> tuple[tuple[Constraint, Constraint], str]:
    constraints = (
        Constraint("paragraphs:paragraphs2", {}),
        Constraint("last_word:last_word_answer", {"last_word": "today"}),
    )
    text = (
        "First paragraph text describing the morning routine of a small coastal village where "
        "fishermen prepare their boats before dawn breaks over the harbor.\n\nSecond paragraph text "
        "continuing the story as the market opens and traders arrange their stalls along the busy "
        "waterfront today"
    )
    return constraints, text


def _pair_d() -> tuple[tuple[Constraint, Constraint], str]:
    constraints = (
        Constraint("first_word:first_word_sent", {"first_word": "clearly"}),
        Constraint("last_word:last_word_sent", {"last_word": "daily"}),
    )
    text = (
        "Clearly the young athletes trained together every single morning daily. Clearly the "
        "coaching staff reviewed detailed performance footage every afternoon daily. Clearly the "
        "entire team's fitness results kept improving steadily each week daily. Clearly the season "
        "finished with everyone feeling proud and accomplished daily."
    )
    return constraints, text


TWO_CONSTRAINT_PAIRS = (_pair_a(), _pair_b(), _pair_c(), _pair_d())


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

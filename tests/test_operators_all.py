"""Tests for the assembled operator registry: coverage, per-family fixtures and
corruption, and cross-group exactness on hand-built two-constraint prompts.
"""

import json
import random
from pathlib import Path

import pytest

import conpo.operators  # noqa: F401  (imports all four group modules, registers every family)
from conpo.operators import registry
from conpo.verifiers import Constraint, FAMILIES, check, feasible

KWARGS: dict[str, dict[str, object]] = json.loads(
    (Path(__file__).parent / "fixtures" / "family_kwargs.json").read_text()
)

REGISTERED_FAMILIES: tuple[str, ...] = tuple(sorted(registry.OPERATORS))

# ---------------------------------------------------------------------------
# (a) registry coverage
# ---------------------------------------------------------------------------


def test_registry_covers_every_family_except_language() -> None:
    assert len(registry.OPERATORS) == 53
    assert set(registry.OPERATORS) == set(FAMILIES) - registry.NO_OPERATOR
    assert set(registry.FIXTURES) == set(FAMILIES) - registry.NO_OPERATOR


def test_coverage_reports_no_missing_families() -> None:
    report = conpo.operators.coverage()
    assert report["registered_operators"] == 53
    assert report["registered_fixtures"] == 53
    assert report["missing"] == []


# ---------------------------------------------------------------------------
# (b) every registered family: fixture passes check(), corrupt yields a candidate,
# for two seeds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iid", REGISTERED_FAMILIES)
def test_fixture_and_corrupt_for_two_seeds(iid: str) -> None:
    c = Constraint.from_ground_truth(iid, KWARGS[iid])
    for seed in (0, 1):
        text = registry.FIXTURES[iid](c, random.Random(seed))
        assert check(c, text) is True, f"{iid} fixture failed for seed {seed}: {text!r}"
        corrupted, reason = registry.corrupt((c,), 0, text, random.Random(seed))
        assert corrupted is not None, f"{iid} produced no corruption for seed {seed} ({reason})"


# ---------------------------------------------------------------------------
# (c) cross-group exactness: two-constraint prompts pairing families from
# different operator groups, each fixture built by hand to satisfy both.
# ---------------------------------------------------------------------------

CROSS_GROUP_CASES: list[tuple[str, Constraint, Constraint, str]] = [
    (
        "no_comma[B]+word_once[C]",
        Constraint("punctuation:no_comma", {}),
        Constraint("keywords:word_once", {"keyword": "unique"}),
        "the report highlights one unique finding from this quarter without any repeated commas in this response",
    ),
    (
        "english_lowercase[B]+last_word_answer[C]",
        Constraint("change_case:english_lowercase", {}),
        Constraint("last_word:last_word_answer", {"last_word": "today"}),
        "the meeting finished early and everyone left the building quietly before sunset today",
    ),
    (
        "title[B]+punctuation_exclamation[D]",
        Constraint("detectable_format:title", {}),
        Constraint("punctuation:punctuation_exclamation", {}),
        "<<field notes>> gentle rivers wander through quiet valleys near old stone bridges each morning",
    ),
    (
        "forbidden_words[A]+count_unique[D]",
        Constraint("keywords:forbidden_words", {"forbidden_words": ["banana", "grape"]}),
        Constraint("count:count_unique", {}),
        "the orchard grows apples and pears throughout this year without any banned produce mentioned here",
    ),
    (
        "end_checker[B]+no_adjacent_consecutive[C]",
        Constraint("startend:end_checker", {"end_phrase": "that is all"}),
        Constraint("keywords:no_adjacent_consecutive", {}),
        "the schedule review finished early and every question got answered well before the "
        "closing remarks and that is all",
    ),
    (
        "existence[A]+quotation[B]",
        Constraint("keywords:existence", {"keywords": ["apple", "river"]}),
        Constraint("startend:quotation", {}),
        '"the apple orchard sits beside the quiet river valley near the old wooden bridge"',
    ),
    (
        "number_words[A]+first_word_answer[C]",
        Constraint("length_constraints:number_words", {"num_words": 10, "relation": "at least"}),
        Constraint("first_word:first_word_answer", {"first_word": "however"}),
        "however the storm passed quickly and the sky cleared before noon today",
    ),
    (
        "number_placeholders[A]+punctuation_exclamation[D]",
        Constraint("detectable_content:number_placeholders", {"num_placeholders": 2}),
        Constraint("punctuation:punctuation_exclamation", {}),
        "please fill in [name] and [date] before the deadline arrives",
    ),
    (
        "english_capital[B]+word_count_different_numbers[C]",
        Constraint("change_case:english_capital", {}),
        Constraint(
            "keywords:word_count_different_numbers", {"keyword": "apple", "frequency": 2, "relation": "at least"}
        ),
        "THE ORCHARD SELLS APPLE PIES AND APPLE JUICE EVERY SINGLE WEEKEND MORNING",
    ),
    (
        "repeat_prompt[B]+palindrome[D]",
        Constraint("combination:repeat_prompt", {"prompt_to_repeat": "Describe the weather today."}),
        Constraint("keywords:palindrome", {}),
        "Describe the weather today. the level water stayed calm beneath the quiet bridge all afternoon long",
    ),
    (
        "exclude_word_harder[C]+letter_counting[D]",
        Constraint("keywords:exclude_word_harder", {"keyword": "banana"}),
        Constraint("letters:letter_counting", {"N": 20, "relation": "at least"}),
        "the busy market sold fresh apples and pears every single weekend morning",
    ),
    (
        "number_bullet_lists[A]+start_end[D]",
        Constraint("detectable_format:number_bullet_lists", {"num_bullets": 2}),
        Constraint("keywords:start_end", {}),
        "morning here is the checklist for tomorrow\n* confirm the venue booking\n* send out the invitations morning",
    ),
]


@pytest.mark.parametrize("name,first,second,text", CROSS_GROUP_CASES)
def test_cross_group_pair_corrupts_exactly(name: str, first: Constraint, second: Constraint, text: str) -> None:
    both = (first, second)
    assert feasible(both, text) is True, f"{name}: joint fixture does not satisfy both constraints: {text!r}"
    for target in (0, 1):
        corrupted, reason = registry.corrupt(both, target, text, random.Random(0))
        assert reason == registry.EXACT, f"{name} target {target} ({both[target].iid}): {reason}"
        assert corrupted is not None


def test_at_least_ten_cross_group_pairs() -> None:
    assert len(CROSS_GROUP_CASES) >= 10


# ---------------------------------------------------------------------------
# (d) summary: per-family reason printout, overall exact rate over every
# (pair, target) case in CROSS_GROUP_CASES
# ---------------------------------------------------------------------------


def test_summary_prints_reasons_and_checks_exact_rate() -> None:
    print()
    for iid in REGISTERED_FAMILIES:
        c = Constraint.from_ground_truth(iid, KWARGS[iid])
        text = registry.FIXTURES[iid](c, random.Random(0))
        _, reason = registry.corrupt((c,), 0, text, random.Random(0))
        print(f"{iid}: {reason}")

    total = 0
    exact = 0
    for name, first, second, text in CROSS_GROUP_CASES:
        both = (first, second)
        for target in (0, 1):
            _, reason = registry.corrupt(both, target, text, random.Random(0))
            total += 1
            exact += reason == registry.EXACT
            print(f"{name} target={target} ({both[target].iid}): {reason}")

    rate = exact / total
    print(f"overall exact rate: {rate:.3f} ({exact}/{total})")
    assert rate >= 0.9

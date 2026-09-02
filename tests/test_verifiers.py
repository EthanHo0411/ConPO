import json
from pathlib import Path

import pytest

from conpo.verifiers import (
    CHECK_ERRORS,
    FAMILIES,
    HELDOUT_TEST_FAMILIES,
    IFEVAL_FAMILIES,
    IFTRAIN_FAMILIES,
    Constraint,
    build,
    check,
    description,
    describe_failure,
    feasible,
    violation_set,
)
import conpo.vendor.ifevalg.instructions as ifevalg_instructions

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "family_kwargs.json").read_text())

GENERIC = (
    "This is a generic response used for a smoke test. It has multiple sentences here. "
    "Here is a third sentence for good measure.\n\n"
    "This is a second paragraph with more content. It continues for a while. "
    "Testing is now complete."
)


# (a) registry shape -----------------------------------------------------


def test_family_counts_and_split() -> None:
    assert len(FAMILIES) == 54
    assert len(IFEVAL_FAMILIES) == 25
    assert len(IFTRAIN_FAMILIES) == 29
    assert set(IFEVAL_FAMILIES) | set(IFTRAIN_FAMILIES) == set(FAMILIES)
    assert set(IFEVAL_FAMILIES).isdisjoint(IFTRAIN_FAMILIES)


def test_heldout_disjoint_from_families() -> None:
    assert len(HELDOUT_TEST_FAMILIES) == 58
    assert set(HELDOUT_TEST_FAMILIES).isdisjoint(FAMILIES)


def test_fixture_covers_every_family() -> None:
    assert set(FIXTURES) == set(FAMILIES)


# (b) every family builds, checks, describes ------------------------------


@pytest.mark.parametrize("iid", FAMILIES)
def test_every_family_builds_checks_and_describes(iid: str) -> None:
    c = Constraint.from_ground_truth(iid, FIXTURES[iid])
    inst = build(c)
    assert inst.id == iid

    result = check(c, GENERIC)
    assert isinstance(result, bool)

    desc = description(c)
    assert isinstance(desc, str) and desc.strip()

    failure = describe_failure(c, GENERIC)
    assert isinstance(failure, str) and failure.strip()


# (c) hand-written passing/failing responses -------------------------------

PASS_FAIL_CASES = {
    "length_constraints:number_words": (
        {"num_words": 5, "relation": "at least"},
        "one two three four five six seven",
        "one two",
    ),
    "punctuation:no_comma": (
        {},
        "This has no commas at all.",
        "This, has a comma.",
    ),
    "startend:quotation": (
        {},
        '"This is quoted."',
        "This is not quoted.",
    ),
    "keywords:existence": (
        {"keywords": ["apple", "river"]},
        "I saw an apple by the river.",
        "I saw a fruit by the water.",
    ),
    "length_constraints:number_sentences": (
        {"num_sentences": 3, "relation": "at least"},
        "One. Two. Three. Four.",
        "Only one sentence.",
    ),
    "detectable_format:number_bullet_lists": (
        {"num_bullets": 2},
        "* point one\n* point two",
        "no bullets here",
    ),
    "detectable_content:number_placeholders": (
        {"num_placeholders": 2},
        "Please fill in [name] and [date].",
        "No placeholders here.",
    ),
    "copy:copying_simple": (
        {"prompt_to_repeat": "Repeat this request exactly as written."},
        "Repeat this request exactly as written.",
        "I will not repeat it.",
    ),
    "keywords:word_once": (
        {"keyword": "unique"},
        "This contains the word unique only once.",
        "This has no matching word.",
    ),
    "count:count_unique": (
        {},
        "apple banana cherry date fig grape honeydew",
        "apple apple banana banana",
    ),
    "keywords:palindrome": (
        {},
        "The dog saw racecar today.",
        "The dog ran across the field quickly today.",
    ),
    "detectable_format:title": (
        {},
        "Here is my answer. <<My Title>>",
        "Here is my answer with no title.",
    ),
}


@pytest.mark.parametrize("iid", sorted(PASS_FAIL_CASES))
def test_hand_written_pass_fail(iid: str) -> None:
    assert len(PASS_FAIL_CASES) >= 10
    kwargs, passing, failing = PASS_FAIL_CASES[iid]
    c = Constraint.from_ground_truth(iid, kwargs)

    assert check(c, passing) is True
    assert check(c, failing) is False
    assert "observed" in describe_failure(c, failing).lower()


# (d) violation_set / feasible on a two-constraint prompt -------------------


def test_violation_set_and_feasible_two_constraints() -> None:
    constraints = (
        Constraint.from_ground_truth("length_constraints:number_words", {"num_words": 5, "relation": "at least"}),
        Constraint.from_ground_truth("punctuation:no_comma", {}),
    )

    both_ok = "one two three four five six seven words here"
    assert feasible(constraints, both_ok) is True
    assert violation_set(constraints, both_ok) == frozenset()

    one_fails = "one two three, four five six seven"  # has a comma
    assert feasible(constraints, one_fails) is False
    assert violation_set(constraints, one_fails) == frozenset({"punctuation:no_comma"})

    both_fail = "one, two"  # too few words and a comma
    assert feasible(constraints, both_fail) is False
    assert violation_set(constraints, both_fail) == frozenset(
        {"length_constraints:number_words", "punctuation:no_comma"}
    )


# (e) exception counter ------------------------------------------------------


def test_check_error_counter_increments(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: object, value: str) -> bool:
        raise RuntimeError("simulated checker failure")

    monkeypatch.setattr(ifevalg_instructions.NumberOfWords, "check_following", boom)

    c = Constraint.from_ground_truth("length_constraints:number_words", {"num_words": 5, "relation": "at least"})
    before = CHECK_ERRORS["length_constraints:number_words"]

    assert check(c, "any response text") is False

    assert CHECK_ERRORS["length_constraints:number_words"] == before + 1

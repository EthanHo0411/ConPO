"""Tests for conpo.metrics on 3 hand-built PoolRow / response pairs.

Row A (1 constraint) passes all; row B (2 constraints) passes some; row C
(3 constraints) passes none. Verifier behaviour for each (constraint,
response) pair below is asserted directly in test_verifier_behaviour_matches_hand_computation,
so the hand-computed PAcc/IAcc/macro/worst-decile values that the rest of
this file checks are traceable to that assertion rather than assumed.
"""

from conpo import metrics
from conpo.data import PoolRow
from conpo.verifiers import Constraint, check

NO_COMMA = Constraint("punctuation:no_comma", {})
QUOTATION = Constraint("startend:quotation", {})
LOWERCASE = Constraint("change_case:english_lowercase", {})


def _row(key: str, constraints: tuple[Constraint, ...]) -> PoolRow:
    return PoolRow(
        key=key,
        base_prompt="do the task",
        prompt="do the task with constraints",
        sentences=tuple(f"c{i}" for i in range(len(constraints))),
        constraints=constraints,
        english=True,
        num_constraints=len(constraints),
    )


ROW_A = _row("a", (NO_COMMA,))
ROW_B = _row("b", (NO_COMMA, QUOTATION))
ROW_C = _row("c", (NO_COMMA, QUOTATION, LOWERCASE))

RESPONSE_A = "no commas here"  # passes NO_COMMA -> all of row A's constraints pass
RESPONSE_B = "no comma but not quoted"  # passes NO_COMMA, fails QUOTATION -> some pass
RESPONSE_C = "This, has commas and Caps"  # fails all three of row C's constraints

ROWS = [ROW_A, ROW_B, ROW_C]
RESPONSES = [RESPONSE_A, RESPONSE_B, RESPONSE_C]


def test_verifier_behaviour_matches_hand_computation() -> None:
    assert check(NO_COMMA, RESPONSE_A) is True
    assert check(NO_COMMA, RESPONSE_B) is True
    assert check(QUOTATION, RESPONSE_B) is False
    assert check(NO_COMMA, RESPONSE_C) is False
    assert check(QUOTATION, RESPONSE_C) is False
    assert check(LOWERCASE, RESPONSE_C) is False


def test_pacc() -> None:
    # row A: 1/1 pass -> 1.0; row B: 1/2 pass -> 0.0; row C: 0/3 pass -> 0.0
    assert metrics.pacc(ROWS, RESPONSES) == (1.0 + 0.0 + 0.0) / 3


def test_iacc() -> None:
    # 6 constraint instances total, 2 pass (row A's NO_COMMA, row B's NO_COMMA)
    assert metrics.iacc(ROWS, RESPONSES) == 2 / 6


def test_family_accuracies() -> None:
    accuracies = metrics.family_accuracies(ROWS, RESPONSES)
    assert accuracies == {
        "punctuation:no_comma": 2 / 3,  # A pass, B pass, C fail
        "startend:quotation": 0.0,  # B fail, C fail
        "change_case:english_lowercase": 0.0,  # C fail
    }


def test_macro_family_accuracy() -> None:
    expected = (2 / 3 + 0.0 + 0.0) / 3
    assert metrics.macro_family_accuracy(ROWS, RESPONSES) == expected


def test_worst_decile_family_accuracy() -> None:
    # 3 families -> bottom max(1, round(0.3)) = 1 family; sorted ascending the
    # worst is 0.0 (both startend:quotation and change_case:english_lowercase
    # tie at 0.0, so either sort order gives the same bottom-1 value).
    assert metrics.worst_decile_family_accuracy(ROWS, RESPONSES) == 0.0


def test_stratified_by_count() -> None:
    strat = metrics.stratified_by_count(ROWS, RESPONSES)
    assert strat == {1: 1.0, 2: 0.0, 3: 0.0}


def test_repetition_rate_on_repeated_sentence() -> None:
    # "a b c d" repeated 3 times: 12 words -> 9 overlapping 4-grams, but only
    # 4 distinct ones (abcd, bcda, cdab, dabc) since the window slides by 1
    # through a period-4 cycle; 5 of the 9 duplicate an earlier window.
    text = "a b c d a b c d a b c d"
    assert metrics.repetition_rate(text, n=4) == 5 / 9


def test_repetition_rate_no_repeats() -> None:
    assert metrics.repetition_rate("a b c d e f g h", n=4) == 0.0


def test_repetition_rate_shorter_than_n() -> None:
    assert metrics.repetition_rate("a b", n=4) == 0.0


def test_mean_length_whitespace() -> None:
    assert metrics.mean_length(["one two three", "four five"]) == (3 + 2) / 2


def test_mean_length_empty() -> None:
    assert metrics.mean_length([]) == 0.0


def test_summarise_has_every_metric_key() -> None:
    summary = metrics.summarise(ROWS, RESPONSES, "unit_test_split")
    assert summary["split"] == "unit_test_split"
    assert summary["num_prompts"] == 3
    assert summary["pacc"] == metrics.pacc(ROWS, RESPONSES)
    assert summary["iacc"] == metrics.iacc(ROWS, RESPONSES)
    assert summary["family_accuracies"] == metrics.family_accuracies(ROWS, RESPONSES)
    assert summary["macro_family_accuracy"] == metrics.macro_family_accuracy(ROWS, RESPONSES)
    assert summary["worst_decile_family_accuracy"] == metrics.worst_decile_family_accuracy(ROWS, RESPONSES)
    assert summary["stratified_by_count"] == metrics.stratified_by_count(ROWS, RESPONSES)
    assert summary["mean_length"] == metrics.mean_length(RESPONSES)
    assert summary["repetition_rate"] == sum(metrics.repetition_rate(r) for r in RESPONSES) / len(RESPONSES)
    assert summary["task_relevance"] is None

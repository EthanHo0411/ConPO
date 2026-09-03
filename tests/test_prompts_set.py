import pytest

from conpo.data import PoolRow, load_pool
from conpo.prompts import counterfactual_prompts, counterfactual_prompts_set

_SAMPLE_LIMIT = 500


@pytest.fixture(scope="module")
def three_sentence_row() -> PoolRow:
    rows, _ = load_pool(limit=_SAMPLE_LIMIT)
    return next(r for r in rows if len(r.sentences) >= 3)


def test_counterfactual_prompts_set_removes_two_targets(three_sentence_row: PoolRow) -> None:
    row = three_sentence_row

    x_plus, x_minus = counterfactual_prompts_set(row, [0, 1])
    assert x_plus == row.prompt
    assert row.sentences[0] not in x_minus
    assert row.sentences[1] not in x_minus
    for sentence in row.sentences[2:]:
        assert sentence in x_minus


def test_counterfactual_prompts_set_single_index_matches_counterfactual_prompts(
    three_sentence_row: PoolRow,
) -> None:
    row = three_sentence_row
    for index in range(len(row.sentences)):
        expected = counterfactual_prompts(row, index)
        actual = counterfactual_prompts_set(row, [index])
        assert actual == expected


def test_counterfactual_prompts_set_single_index_matches_across_pool() -> None:
    # Sweeps every (row, index) pair in the first _SAMPLE_LIMIT pool rows,
    # rather than one fixture: some pool prompts carry irregular (double or
    # triple) whitespace between constraint sentences, which a single lucky
    # fixture can miss entirely while the general claim still fails on ~9% of pairs.
    rows, _ = load_pool(limit=_SAMPLE_LIMIT)
    for row in rows:
        for index in range(len(row.sentences)):
            assert counterfactual_prompts_set(row, [index]) == counterfactual_prompts(row, index)


def test_counterfactual_prompts_set_raises_when_a_target_sentence_missing() -> None:
    row = PoolRow(
        key="k",
        base_prompt="base",
        prompt="base tail here",
        sentences=("tail here", "not present verbatim"),
        constraints=(),
        english=True,
        num_constraints=2,
    )
    with pytest.raises(ValueError):
        counterfactual_prompts_set(row, [0, 1])


def test_counterfactual_prompts_set_order_of_indices_does_not_matter() -> None:
    dup_a, dup_b, dup_c = "First requirement here.", "Second requirement here.", "Third requirement here."
    sentences = (dup_a, dup_b, dup_c)
    prompt = f"Do the task. {dup_a} {dup_b} {dup_c}"
    row = PoolRow(
        key="k",
        base_prompt="Do the task.",
        prompt=prompt,
        sentences=sentences,
        constraints=(),
        english=True,
        num_constraints=3,
    )
    _, forward = counterfactual_prompts_set(row, [0, 2])
    _, reversed_order = counterfactual_prompts_set(row, [2, 0])
    assert forward == reversed_order == f"{row.base_prompt} {dup_b}"


def test_counterfactual_prompts_set_empty_indices_keeps_every_sentence() -> None:
    dup_a, dup_b = "First requirement here.", "Second requirement here."
    sentences = (dup_a, dup_b)
    prompt = f"Do the task. {dup_a} {dup_b}"
    row = PoolRow(
        key="k",
        base_prompt="Do the task.",
        prompt=prompt,
        sentences=sentences,
        constraints=(),
        english=True,
        num_constraints=2,
    )
    x_plus, x_minus = counterfactual_prompts_set(row, [])
    assert x_plus == row.prompt
    assert x_minus == row.prompt

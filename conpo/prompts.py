"""Counterfactual prompt construction (ConPO.pdf section 2.5): x^+ keeps every
constraint sentence, x^- drops exactly the target sentence."""

from collections.abc import Sequence

from conpo.data import PoolRow


def _require_verbatim(prompt: str, sentence: str) -> None:
    if sentence not in prompt:
        raise ValueError(f"sentence not found verbatim in prompt: {sentence!r}")


def _sentence_span(prompt: str, sentences: Sequence[str], target_index: int) -> tuple[int, int]:
    """Locate `sentences[target_index]`'s occurrence in `prompt` by walking
    every sentence in order (each search starts after the previous match).
    This picks the physically correct occurrence even when two sentences in
    `sentences` are textually identical, instead of always matching the
    leftmost occurrence of the text."""
    pos = 0
    start = end = -1
    for i, sentence in enumerate(sentences):
        idx = prompt.index(sentence, pos)
        if i == target_index:
            start, end = idx, idx + len(sentence)
            break
        pos = idx + len(sentence)
    if start < 0:
        raise ValueError(f"sentence not found verbatim in prompt: {sentences[target_index]!r}")
    return start, end


def _remove_sentence(prompt: str, sentences: Sequence[str], target_index: int) -> str:
    start, end = _sentence_span(prompt, sentences, target_index)
    before = prompt[:start].rstrip(" ")
    after = prompt[end:].lstrip(" ")
    return f"{before} {after}" if after else before


def counterfactual_prompts(row: PoolRow, target_index: int) -> tuple[str, str]:
    """(x_plus, x_minus): x_plus is `row.prompt` unchanged, x_minus drops the
    sentence at `target_index` (exact string removal, single-space repair,
    every other sentence byte-identical). Raises if the sentence is not
    found verbatim in `row.prompt`."""
    x_plus = row.prompt
    x_minus = _remove_sentence(row.prompt, row.sentences, target_index)
    return x_plus, x_minus


def prompt_with_only(row: PoolRow, indices: Sequence[int]) -> str:
    """`row.base_prompt` plus only the constraint sentences at `indices`, in
    their original order. Raises if any selected sentence is not found
    verbatim in `row.prompt`."""
    selected = [row.sentences[i] for i in indices]
    for sentence in selected:
        _require_verbatim(row.prompt, sentence)
    if not selected:
        return row.base_prompt
    return f"{row.base_prompt} {' '.join(selected)}"

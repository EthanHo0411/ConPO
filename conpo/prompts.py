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


def _remove_sentences(prompt: str, sentences: Sequence[str], target_indices: Sequence[int]) -> str:
    """Remove every sentence at an index in `target_indices` from `prompt`.

    Locates each target's span with `_sentence_span` (against the original,
    unmodified `prompt`, so every span offset is valid up front), then splices
    spans out highest-offset first so an unprocessed span's offset is never
    shifted by an earlier splice, collapsing each deletion seam to a single
    space exactly as single-sentence removal does. Every character kept is
    byte-identical to `prompt`; empty `target_indices` returns `prompt`
    unchanged."""
    spans = sorted((_sentence_span(prompt, sentences, i) for i in set(target_indices)), reverse=True)
    result = prompt
    for start, end in spans:
        before = result[:start].rstrip(" ")
        after = result[end:].lstrip(" ")
        result = f"{before} {after}" if after else before
    return result


def _remove_sentence(prompt: str, sentences: Sequence[str], target_index: int) -> str:
    return _remove_sentences(prompt, sentences, [target_index])


def counterfactual_prompts(row: PoolRow, target_index: int) -> tuple[str, str]:
    """(x_plus, x_minus): x_plus is `row.prompt` unchanged, x_minus drops the
    sentence at `target_index` (exact string removal, single-space repair,
    every other sentence byte-identical). Raises if the sentence is not
    found verbatim in `row.prompt`."""
    x_plus = row.prompt
    x_minus = _remove_sentence(row.prompt, row.sentences, target_index)
    return x_plus, x_minus


def counterfactual_prompts_set(row: PoolRow, target_indices: Sequence[int]) -> tuple[str, str]:
    """(x_plus, x_minus): x_plus is `row.prompt` unchanged, x_minus drops every
    sentence at an index in `target_indices` (byte-preserving span removal,
    same seam handling as `counterfactual_prompts`, every other sentence and
    surrounding whitespace byte-identical to `row.prompt`). Raises if any
    sentence at a `target_indices` index is not found verbatim in
    `row.prompt`. Single-index behaviour is identical to
    `counterfactual_prompts`.

    x_minus is not interchangeable with `prompt_with_only(row, kept)` for the
    complementary kept indices: x_minus splices spans out of `row.prompt` and
    so preserves the original inter-sentence whitespace byte-for-byte, while
    `prompt_with_only` rebuilds from `row.base_prompt` and joins kept
    sentences with a single space, collapsing any irregular whitespace
    between them. The two disagree whenever `row.prompt` has double/triple
    spacing between two kept sentences (verified empirically: 8.2% of rows
    with >= 3 sentences in a 2000-row pool sample)."""
    x_plus = row.prompt
    x_minus = _remove_sentences(row.prompt, row.sentences, target_indices)
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

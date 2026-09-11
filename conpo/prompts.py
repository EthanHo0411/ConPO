"""Counterfactual prompt construction (ConPO.pdf section 2.5): x^+ keeps every
constraint sentence, x^- drops exactly the target sentence."""

import dataclasses
from collections.abc import Sequence
from typing import Any

from conpo import config
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


# ---------------------------------------------------------------------------
# Prompt-length cap (TASK G, design choice, builder, 2026-09-11): TRL 1.9.2's
# GRPOConfig has no `max_prompt_length` field at all (verified: absent from
# the installed trl/trainer/grpo_config.py; GRPOTrainer._tokenize_prompts
# renders every prompt through apply_chat_template(tokenize=True) with no
# truncation, and _generate_single_turn hands the colocated vLLM engine those
# raw, unbounded token ids as-is), so nothing upstream of this pipeline's own
# data layer ever caps a prompt -- three crashes confirmed it (VLLMValidationError,
# prompt longer than MAX_MODEL_LEN, at training steps and in eval). These two
# functions are the cap, applied once at row-load time (scripts/train.py) so
# every downstream consumer of `row.prompt` (the TRL dataset, conpo/forward.py's
# repair prompts, conpo/reverse.py's corruption and this module's own
# counterfactual_prompts_set, conpo/judge.py's VetoRequest.prompt) sees an
# already-capped prompt without any of those modules needing to know about it.
# ---------------------------------------------------------------------------


def cap_prompt_messages(
    messages: list[dict[str, str]],
    tokenizer: Any,
    max_tokens: int,
) -> tuple[list[dict[str, str]], bool]:
    """Left-truncate `messages[0]`'s ("user") content, keeping its tail, so
    the rendered prompt (policy chat template, add_generation_prompt=True,
    thinking off -- config.POLICY_CHAT_TEMPLATE_KWARGS) tokenizes to at most
    `max_tokens` tokens.

    Tail kept, head dropped: IF_multi_constraints_upto5's constraint
    sentences are appended at the end of the user turn (verified against the
    cached corpus), so a tail truncation keeps every constraint sentence
    intact and only drops free task text from the front. Mirrors the same
    left-truncation convention as scripts/eval_external.py's
    left_truncate_prompt (eval side; that script has no shared package
    import, so it keeps its own copy).

    `overhead` (the rendered length of an empty user turn) accounts for the
    chat template's own tokens (role tags, the generation prompt, the
    thinking-disabled block) so the trimmed content is sized correctly on
    the first attempt; the loop after it only runs the rare extra time or
    two a token can re-merge differently right at the cut boundary.

    Returns (messages, was_truncated); every messages list this pipeline
    builds is a single user turn, so only messages[0] is ever touched.
    """

    def rendered_len(msgs: list[dict[str, str]]) -> int:
        text = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, **config.POLICY_CHAT_TEMPLATE_KWARGS
        )
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    if rendered_len(messages) <= max_tokens:
        return messages, False

    content = messages[0]["content"]
    overhead = rendered_len([{"role": "user", "content": ""}])
    budget = max(0, max_tokens - overhead)
    ids = tokenizer(content, add_special_tokens=False)["input_ids"]
    tail = ids[-budget:] if budget else []
    while True:
        trimmed = tokenizer.decode(tail, skip_special_tokens=True) if tail else ""
        new_messages = [{"role": "user", "content": trimmed}, *messages[1:]]
        if rendered_len(new_messages) <= max_tokens or not tail:
            return new_messages, True
        tail = tail[1:]


def cap_pool_rows(
    rows: Sequence[PoolRow],
    tokenizer: Any,
    max_tokens: int,
) -> tuple[list[PoolRow], int]:
    """Cap every row's `.prompt` with `cap_prompt_messages`, once, so no
    downstream consumer ever hands an engine a prompt over `max_tokens`
    (module docstring above lists them).

    Only `.prompt` changes -- `.base_prompt` and `.sentences` stay at their
    original, pre-truncation values (`.sentences` still occur verbatim in
    the possibly-shorter `.prompt`, since truncation only ever removes
    characters from the head and the sentences sit at the tail); a reader of
    `.base_prompt` alone (e.g. scripts/check_overlap.py's train/eval overlap
    check) wants the original untruncated task text regardless.

    Returns (new_rows, truncated_count).
    """
    new_rows: list[PoolRow] = []
    truncated_count = 0
    for row in rows:
        messages, was_truncated = cap_prompt_messages(
            [{"role": "user", "content": row.prompt}], tokenizer, max_tokens
        )
        if was_truncated:
            truncated_count += 1
            row = dataclasses.replace(row, prompt=messages[0]["content"])
        new_rows.append(row)
    return new_rows, truncated_count


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

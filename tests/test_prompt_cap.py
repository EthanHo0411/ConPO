"""TASK G: conpo.prompts.cap_prompt_messages/cap_pool_rows (the ConPO data-layer
prompt cap TRL itself never applies -- see conpo/prompts.py's module docstring
for that module's own docstring above cap_prompt_messages), and the judge's
raised max_model_len (conpo/config.py's JUDGE_MAX_MODEL_LEN).

CPU only: a real Qwen3-4B tokenizer (cached, no GPU) plus conpo.forward's
real repair_trajectory with a FakeEngine, no model load.
"""

from __future__ import annotations

import random

from transformers import AutoTokenizer

from conpo import config
from conpo.data import PoolRow, TokenMeter
from conpo.engine import FakeEngine
from conpo.forward import repair_trajectory
from conpo.judge import StaticJudge, VetoRequest, build_veto_prompt
from conpo.prompts import cap_pool_rows, cap_prompt_messages
from conpo.verifiers import Constraint

TOKENIZER = AutoTokenizer.from_pretrained(config.POLICY_MODEL)

NO_COMMA = Constraint("punctuation:no_comma", {})
LOWERCASE = Constraint("change_case:english_lowercase", {})


def _n_token_text(n: int) -> str:
    """`n`-token text under TOKENIZER, built by over-generating then slicing
    ids to an exact count (same technique used to size the worst-case veto
    prompt below)."""
    ids = TOKENIZER("constraint " * (n * 2 + 10), add_special_tokens=False)["input_ids"][:n]
    return TOKENIZER.decode(ids, skip_special_tokens=True)


def _rendered_len(messages: list[dict[str, str]]) -> int:
    text = TOKENIZER.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, **config.POLICY_CHAT_TEMPLATE_KWARGS
    )
    return len(TOKENIZER(text, add_special_tokens=False)["input_ids"])


# ---------------------------------------------------------------------------
# cap_prompt_messages
# ---------------------------------------------------------------------------


def test_cap_prompt_messages_under_cap_is_unchanged() -> None:
    messages = [{"role": "user", "content": "Write a short poem."}]
    capped, was_truncated = cap_prompt_messages(messages, TOKENIZER, max_tokens=2048)
    assert was_truncated is False
    assert capped == messages


def test_cap_prompt_messages_truncates_6000_token_turn_and_keeps_tail() -> None:
    marker = "FINAL CONSTRAINT: do not use the word banana."
    content = f"{_n_token_text(6000)} {marker}"
    messages = [{"role": "user", "content": content}]

    capped, was_truncated = cap_prompt_messages(messages, TOKENIZER, max_tokens=config.MAX_PROMPT_TOKENS)

    assert was_truncated is True
    assert _rendered_len(capped) <= config.MAX_PROMPT_TOKENS
    assert capped[0]["content"].endswith(marker)
    # Actually shorter, not padded/unchanged.
    assert len(capped[0]["content"]) < len(content)


# ---------------------------------------------------------------------------
# cap_pool_rows
# ---------------------------------------------------------------------------


def _make_row(key: str, base: str) -> PoolRow:
    sentence = "Do not use any commas."
    return PoolRow(
        key=key,
        base_prompt=base,
        prompt=f"{base} {sentence}",
        sentences=(sentence,),
        constraints=(NO_COMMA,),
        english=True,
        num_constraints=1,
    )


def test_cap_pool_rows_only_truncates_over_cap_rows_and_preserves_base_prompt() -> None:
    short_row = _make_row("short", "Explain how gravity works.")
    long_sentence = "Do not use any commas."
    long_base = _n_token_text(6000)
    long_row = PoolRow(
        key="long",
        base_prompt=long_base,
        prompt=f"{long_base} {long_sentence}",
        sentences=(long_sentence,),
        constraints=(NO_COMMA,),
        english=True,
        num_constraints=1,
    )

    capped_rows, truncated_count = cap_pool_rows([short_row, long_row], TOKENIZER, config.MAX_PROMPT_TOKENS)

    assert truncated_count == 1
    capped_short, capped_long = capped_rows
    assert capped_short == short_row  # untouched, same object contents
    assert capped_long.prompt != long_row.prompt
    assert capped_long.prompt.endswith(long_sentence)  # constraint sentence survives, verbatim
    # base_prompt/sentences are the pre-truncation, original values (module
    # docstring): a downstream reader of base_prompt alone still sees the
    # full original task text.
    assert capped_long.base_prompt == long_base
    assert capped_long.sentences == (long_sentence,)
    assert _rendered_len([{"role": "user", "content": capped_long.prompt}]) <= config.MAX_PROMPT_TOKENS


# ---------------------------------------------------------------------------
# The tiny trainer path (test_trainer_tiny.py/test_batched_veto_trainer.py
# style, FakeEngine): forward repair never hands the engine an over-cap
# prompt, once the row it operates on was capped at load time like
# scripts/train.py now does.
# ---------------------------------------------------------------------------


def test_forward_repair_never_hands_engine_a_prompt_over_the_cap() -> None:
    raw_prompt = f"{_n_token_text(6000)} Do not use any commas."
    capped_messages, was_truncated = cap_prompt_messages(
        [{"role": "user", "content": raw_prompt}], TOKENIZER, config.MAX_PROMPT_TOKENS
    )
    assert was_truncated is True
    row = PoolRow(
        key="huge-row",
        base_prompt=raw_prompt,
        prompt=capped_messages[0]["content"],
        sentences=("Do not use any commas.",),
        constraints=(NO_COMMA,),
        english=True,
        num_constraints=1,
    )

    # Every proposal still has a comma, so monotone_admit never admits one
    # (repair_trajectory stops after the first step with nothing admitted) --
    # what matters here is the one engine.generate call that does happen.
    engine = FakeEngine(lambda messages, n: ["Still, has a comma."] * n)
    meter = TokenMeter(budget=0)

    result = repair_trajectory(
        row,
        "Initial, response, with commas.",
        engine,
        StaticJudge(keep=True),
        random.Random(0),
        meter,
        current_update=0,
        max_steps=3,
        max_candidates=2,
        veto=False,
        distance_gate=False,
    )

    assert engine.calls  # repair actually ran, not a vacuous pass
    assert result.admitted == 0
    for messages, _n in engine.calls:
        prompt_turn = messages[0]["content"]
        assert prompt_turn == row.prompt  # repair never re-derives it, always the already-capped row.prompt
        assert _rendered_len([{"role": "user", "content": prompt_turn}]) <= config.MAX_PROMPT_TOKENS


# ---------------------------------------------------------------------------
# Judge worst case (TASK G part 2): build_veto_prompt embeds the task prompt
# PLUS both full responses; conpo/config.py's JUDGE_MAX_MODEL_LEN must cover
# the worst case with room left for the judge's own reply.
# ---------------------------------------------------------------------------


def test_veto_prompt_worst_case_fits_judge_max_model_len() -> None:
    # 5 constraints is the corpus maximum (a row with a 6th would land in
    # eval_iii, never train -- conpo/data.py's make_splits); each description
    # here is sized like the longest real ones measured against the
    # installed verifiers (~40-50 tokens each).
    descriptions = tuple(
        f"The response must satisfy requirement number {i} of this multi-part instruction in full, "
        "without omitting or paraphrasing any clause of it."
        for i in range(5)
    )
    req = VetoRequest(
        prompt=_n_token_text(config.MAX_PROMPT_TOKENS),
        y_plus=_n_token_text(config.MAX_COMPLETION_TOKENS),
        y_minus=_n_token_text(config.MAX_COMPLETION_TOKENS),
        families=tuple(f"family_{i}" for i in range(5)),
        descriptions=descriptions,
    )

    veto_prompt = build_veto_prompt(req)
    rendered = TOKENIZER.apply_chat_template(
        [{"role": "user", "content": veto_prompt}], add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    input_tokens = len(TOKENIZER(rendered, add_special_tokens=False)["input_ids"])
    # 256: conpo/judge.py's _vllm_burst SamplingParams(max_tokens=256), the
    # judge's own reply budget -- vLLM's colocate/offline path stops
    # generation at max_model_len rather than erroring mid-generation, but a
    # correctly-sized budget should never need to rely on that.
    total_worst_case = input_tokens + 256

    assert total_worst_case <= config.JUDGE_MAX_MODEL_LEN
    # Sanity: this really is the scenario the old hardcoded 4096 could not
    # fit (JUDGE_MAX_MODEL_LEN's own comment in conpo/config.py has the
    # measured numbers) -- otherwise this test would pass by accident.
    assert total_worst_case > 4096

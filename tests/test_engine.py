"""HFEngine on a tiny random Qwen3 model with the real Qwen3 tokenizer; FakeEngine scripting."""

import torch
import pytest
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from conpo import config
from conpo.engine import FakeEngine, HFEngine

MESSAGES_A = [{"role": "user", "content": "hi"}]
MESSAGES_B = [{"role": "user", "content": "say two sentences about cats"}]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(config.POLICY_MODEL)


@pytest.fixture(scope="module")
def hf_engine(tokenizer) -> HFEngine:
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=151936,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
    )
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return HFEngine(model, tokenizer, device="cpu")


def test_rendered_prompt_ends_generation_prompt_with_no_think_block(tokenizer) -> None:
    rendered = tokenizer.apply_chat_template(
        MESSAGES_A, add_generation_prompt=True, tokenize=False, **config.POLICY_CHAT_TEMPLATE_KWARGS
    )
    assert rendered.endswith("<think>\n\n</think>\n\n")


def test_generate_returns_n_completions_per_prompt(hf_engine: HFEngine) -> None:
    out = hf_engine.generate(
        [MESSAGES_A, MESSAGES_B], n=3, max_tokens=8, temperature=0.8, top_p=0.95, seed=0
    )
    assert len(out) == 2
    for completions in out:
        assert len(completions) == 3
        for text in completions:
            assert isinstance(text, str)


def test_generate_greedy_when_temperature_zero(hf_engine: HFEngine) -> None:
    out = hf_engine.generate([MESSAGES_A], n=2, max_tokens=6, temperature=0.0, top_p=1.0, seed=0)
    assert len(out[0]) == 2
    # greedy: both members of a num_return_sequences=2 batch are identical
    assert out[0][0] == out[0][1]


def test_generate_empty_batch_returns_empty_list(hf_engine: HFEngine) -> None:
    assert hf_engine.generate([], n=2, max_tokens=8, temperature=0.8, top_p=0.95, seed=0) == []


def test_count_tokens_is_positive_and_monotone(hf_engine: HFEngine) -> None:
    assert hf_engine.count_tokens([]) == 0
    one = hf_engine.count_tokens(["hello world"])
    two = hf_engine.count_tokens(["hello world", "hello world"])
    assert one > 0
    assert two == 2 * one


def test_fake_engine_list_mode_consumes_n_at_a_time() -> None:
    engine = FakeEngine(["a", "b", "c", "d"])
    first = engine.generate([MESSAGES_A], n=2, max_tokens=8, temperature=0.8, top_p=0.95, seed=0)
    second = engine.generate([MESSAGES_A], n=2, max_tokens=8, temperature=0.8, top_p=0.95, seed=0)
    assert first == [["a", "b"]]
    assert second == [["c", "d"]]
    assert len(engine.calls) == 2


def test_fake_engine_callable_mode_receives_messages_and_n() -> None:
    def script(messages, n):
        return [f"{messages[0]['content']}-{i}" for i in range(n)]

    engine = FakeEngine(script)
    out = engine.generate([MESSAGES_A, MESSAGES_B], n=2, max_tokens=8, temperature=0.8, top_p=0.95, seed=0)
    assert out == [["hi-0", "hi-1"], ["say two sentences about cats-0", "say two sentences about cats-1"]]


def test_fake_engine_count_tokens() -> None:
    engine = FakeEngine(["irrelevant"])
    assert engine.count_tokens(["one two three", "four"]) == 4

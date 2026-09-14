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


# --- VLLMColocateEngine: prompt contract (2026-09-08) --------------------------


class _RecordingGeneration:
    """Stands in for trl's VLLMGeneration: records the prompts it was handed."""

    def __init__(self) -> None:
        self.temperature = 0.5
        self.top_p = 0.9
        self.max_completion_length = 16
        self.generation_kwargs: dict[str, object] = {}
        self.prompts: list[object] = []
        self.synced = 0
        # Snapshot of generation_kwargs as generate() saw it, taken before
        # VLLMColocateEngine's `finally` restores the caller's prior value.
        self.generation_kwargs_seen: dict[str, object] | None = None

    def sync_weights(self) -> None:
        self.synced += 1

    def generate(self, prompts, images, num_generations):  # type: ignore[no-untyped-def]
        self.prompts = list(prompts)
        self.generation_kwargs_seen = dict(self.generation_kwargs)
        return None, [[151644, 872] for _ in prompts], None, None


def test_vllm_colocate_engine_hands_vllm_plain_token_id_lists(tokenizer) -> None:
    """transformers 5 returns a BatchEncoding from apply_chat_template(tokenize=True);
    vLLM's input validation does max(prompt_ids), which on a dict sees the key
    'input_ids' (a str) and raises. The engine must pass plain int lists."""
    from types import SimpleNamespace

    from conpo.engine import VLLMColocateEngine

    gen = _RecordingGeneration()
    trainer = SimpleNamespace(processing_class=tokenizer, vllm_generation=gen, state=SimpleNamespace(global_step=3))
    engine = VLLMColocateEngine(trainer)
    out = engine.generate([[{"role": "user", "content": "hi"}]], n=2, max_tokens=8, temperature=0.1, top_p=1.0, seed=7)

    assert len(gen.prompts) == 2  # n copies of the single prompt
    for ids in gen.prompts:
        assert isinstance(ids, list) and ids and all(isinstance(t, int) for t in ids)
    assert gen.synced == 1
    assert (gen.temperature, gen.top_p, gen.max_completion_length, gen.generation_kwargs) == (0.5, 0.9, 16, {})
    assert len(out) == 1 and len(out[0]) == 2


def test_vllm_colocate_engine_never_forwards_a_seed_into_generation_kwargs(tokenizer) -> None:
    """B1 (audit 2026-09-14): TRL's colocate branch builds ONE SamplingParams
    for the whole duplicated-prompt list, so a per-call seed there would give
    vLLM's request-level torch.Generator the same seed for every duplicate of
    a prompt and every duplicate would sample identically. generate() must
    never write "seed" into gen.generation_kwargs, and must restore whatever
    generation_kwargs the trainer had configured, untouched, once it returns."""
    from types import SimpleNamespace

    from conpo.engine import VLLMColocateEngine

    gen = _RecordingGeneration()
    gen.generation_kwargs = {"top_k": 20}  # a trainer-configured kwarg, unrelated to seed
    trainer = SimpleNamespace(processing_class=tokenizer, vllm_generation=gen, state=SimpleNamespace(global_step=3))
    engine = VLLMColocateEngine(trainer)

    engine.generate([[{"role": "user", "content": "hi"}]], n=4, max_tokens=8, temperature=0.1, top_p=1.0, seed=99)

    assert gen.generation_kwargs_seen is not None
    assert "seed" not in gen.generation_kwargs_seen
    assert gen.generation_kwargs_seen == {"top_k": 20}
    assert gen.generation_kwargs == {"top_k": 20}  # restored to the prior value, untouched


def test_vllm_colocate_engine_skips_second_sync_when_trainer_already_synced_this_step(tokenizer) -> None:
    """B5 (ii, audit 2026-09-14): under RUN_VLLM_PUSH_ONCE_PER_STEP, the
    trainer's own reload guard already pushes+syncs weights once per step
    (trainer._vllm_synced_this_step); _sync_if_stale must not sync_weights()
    again for a same-step generate() call (e.g. forward-repair, which runs
    after TRL's own rollout in the same step) -- that would push unchanged
    weights a second time for nothing."""
    from types import SimpleNamespace

    from conpo.engine import VLLMColocateEngine

    gen = _RecordingGeneration()
    trainer = SimpleNamespace(
        processing_class=tokenizer,
        vllm_generation=gen,
        state=SimpleNamespace(global_step=3),
        _vllm_push_once=True,
        _vllm_synced_this_step=True,
    )
    engine = VLLMColocateEngine(trainer)

    engine.generate([[{"role": "user", "content": "hi"}]], n=2, max_tokens=8, temperature=0.1, top_p=1.0, seed=7)

    assert gen.synced == 0


def test_vllm_colocate_engine_still_syncs_when_push_once_off_or_trainer_not_yet_synced(tokenizer) -> None:
    """The skip in _sync_if_stale must be scoped to the push-once-and-already-
    synced case only: push_once off (per-call reload, old behaviour) and
    push_once on but not yet synced this step both still sync_weights()."""
    from types import SimpleNamespace

    from conpo.engine import VLLMColocateEngine

    for push_once, synced_this_step in [(False, True), (True, False), (False, False)]:
        gen = _RecordingGeneration()
        trainer = SimpleNamespace(
            processing_class=tokenizer,
            vllm_generation=gen,
            state=SimpleNamespace(global_step=3),
            _vllm_push_once=push_once,
            _vllm_synced_this_step=synced_this_step,
        )
        engine = VLLMColocateEngine(trainer)

        engine.generate([[{"role": "user", "content": "hi"}]], n=2, max_tokens=8, temperature=0.1, top_p=1.0, seed=7)

        assert gen.synced == 1, (push_once, synced_this_step)

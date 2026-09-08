"""Sampling engines: the one thing forward.py and reverse.py generate through.

`Engine` is the Protocol every producer codes against. `HFEngine` runs a
loaded transformers model directly (used for tests and any HF-generate path).
`VLLMColocateEngine` wraps a live TRL `GRPOTrainer`'s colocated vLLM
generation for the real training loop. `FakeEngine` is a scripted stand-in
for tests.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol

import torch

from conpo import config
from conpo.judge import strip_think

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class Engine(Protocol):
    """Sampling + tokenisation contract every producer codes against."""

    def generate(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, str]]],
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[list[str]]:
        """`n` decoded completions per messages list, in `messages_batch` order.

        Completions carry no special tokens and no leading think block.
        """
        ...

    def count_tokens(self, texts: Sequence[str]) -> int:
        """Total token count across `texts`, via this engine's own tokeniser."""
        ...


class HFEngine:
    """Engine backed by a loaded transformers model, run directly on `device`.

    Batched, left-padded `model.generate`; Qwen3's chat template (with
    `config.POLICY_CHAT_TEMPLATE_KWARGS`) ends the generation prompt with
    "<think>\\n\\n</think>\\n\\n", so completions carry no think block, but
    `strip_think` is still applied defensively.
    """

    def __init__(self, model: "PreTrainedModel", tokenizer: "PreTrainedTokenizerBase", device: str = "cpu") -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, str]]],
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[list[str]]:
        if not messages_batch:
            return []
        torch.manual_seed(seed)
        prompts = [
            self.tokenizer.apply_chat_template(
                list(messages),
                add_generation_prompt=True,
                tokenize=False,
                **config.POLICY_CHAT_TEMPLATE_KWARGS,
            )
            for messages in messages_batch
        ]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        do_sample = temperature > 0
        # Greedy decoding is deterministic: transformers rejects
        # num_return_sequences > 1 without do_sample, and generating n
        # identical sequences would be wasted compute anyway, so ask for 1
        # and replicate it n times below.
        request_n = n if do_sample else 1
        generate_kwargs: dict[str, object] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "num_return_sequences": request_n,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        with torch.no_grad():
            output = self.model.generate(**encoded, **generate_kwargs)

        prompt_len = encoded["input_ids"].shape[1]
        texts = [strip_think(t) for t in self.tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)]
        if do_sample:
            return [texts[i * n : (i + 1) * n] for i in range(len(messages_batch))]
        return [[texts[i]] * n for i in range(len(messages_batch))]

    def count_tokens(self, texts: Sequence[str]) -> int:
        if not texts:
            return 0
        encoded = self.tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return sum(len(ids) for ids in encoded)


class VLLMColocateEngine:
    """Engine backed by a live TRL `GRPOTrainer`'s colocated vLLM generation.

    UNTESTED: no GPU is available in this environment. Verified only by
    reading the installed trl==1.9.2 source: `trl/generation/vllm_generation.py`
    (`VLLMGeneration.generate`, `sync_weights`) and
    `trl/trainer/grpo_trainer.py:1784-1805` (`_generate_single_turn`, which
    shows the real `n=1`-per-call colocate contract: the caller must already
    hand `generate()` one token-id list per completion wanted, not one per
    prompt, and per-call sampling params other than `n` are read off the
    `VLLMGeneration` instance's own attributes rather than passed as
    arguments) and the `_last_loaded_step` gating pattern at line 1791.

    `trainer` is duck-typed (a `GRPOTrainer`) so importing trl is not forced
    on every caller of this module; only `trainer.state.global_step`,
    `trainer.vllm_generation` and `trainer.processing_class` are used.
    """

    def __init__(self, trainer: object) -> None:
        self.trainer = trainer
        self._last_synced_step = -1

    def _sync_if_stale(self) -> None:
        step = self.trainer.state.global_step  # type: ignore[attr-defined]
        if step != self._last_synced_step:
            self.trainer.vllm_generation.sync_weights()  # type: ignore[attr-defined]
            self._last_synced_step = step

    def generate(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, str]]],
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[list[str]]:
        if not messages_batch:
            return []
        self._sync_if_stale()
        tokenizer = self.trainer.processing_class  # type: ignore[attr-defined]
        gen = self.trainer.vllm_generation  # type: ignore[attr-defined]

        prompt_ids = [
            tokenizer.apply_chat_template(
                list(messages),
                add_generation_prompt=True,
                tokenize=True,
                # transformers 5 returns a BatchEncoding dict by default; vLLM
                # needs the plain id list (tasks/lessons.md 2026-09-08).
                return_dict=False,
                **config.POLICY_CHAT_TEMPLATE_KWARGS,
            )
            for messages in messages_batch
        ]
        # colocate mode generates exactly 1 completion per prompt entry
        # (VLLMGeneration.generate's `n` is hard-coded to 1 there), so the
        # trainer's own num_generations duplication has to happen here.
        duplicated = [ids for ids in prompt_ids for _ in range(n)]

        prior = (gen.temperature, gen.top_p, gen.max_completion_length, gen.generation_kwargs)
        gen.temperature = temperature
        gen.top_p = top_p
        gen.max_completion_length = max_tokens
        gen.generation_kwargs = {**prior[3], "seed": seed}
        try:
            _, completion_ids, _, _ = gen.generate(prompts=duplicated, images=None, num_generations=n)
        finally:
            gen.temperature, gen.top_p, gen.max_completion_length, gen.generation_kwargs = prior

        texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return [[strip_think(t) for t in texts[i * n : (i + 1) * n]] for i in range(len(messages_batch))]

    def count_tokens(self, texts: Sequence[str]) -> int:
        if not texts:
            return 0
        tokenizer = self.trainer.processing_class  # type: ignore[attr-defined]
        encoded = tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return sum(len(ids) for ids in encoded)


class FakeEngine:
    """Scripted Engine for tests.

    `responses` is either:
      - a flat sequence of strings, consumed `n` at a time, in call order,
        shared across every prompt in every `generate()` call; or
      - a callable `(messages, n) -> list[str]`, invoked once per prompt in
        the batch and expected to return exactly `n` completions.
    """

    def __init__(
        self,
        responses: Callable[[Sequence[Mapping[str, str]], int], Sequence[str]] | Sequence[str],
    ) -> None:
        self._callable = responses if callable(responses) else None
        self._queue: list[str] = [] if self._callable is not None else list(responses)
        self.calls: list[tuple[list[Mapping[str, str]], int]] = []

    def generate(
        self,
        messages_batch: Sequence[Sequence[Mapping[str, str]]],
        n: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> list[list[str]]:
        result: list[list[str]] = []
        for messages in messages_batch:
            self.calls.append((list(messages), n))
            if self._callable is not None:
                batch = list(self._callable(messages, n))
            else:
                batch, self._queue = self._queue[:n], self._queue[n:]
            if len(batch) != n:
                raise ValueError(f"FakeEngine needed {n} responses, got {len(batch)}")
            result.append(batch)
        return result

    def count_tokens(self, texts: Sequence[str]) -> int:
        return sum(len(t.split()) for t in texts)

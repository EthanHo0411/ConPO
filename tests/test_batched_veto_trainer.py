"""Trainer-side tests for the batched veto (USER DECISIONS 2026-09-02):
ConPOTrainer.pending_queue / flush_pending_veto and the forward.py/reverse.py
wiring in conpo/trainer.py:_generate_and_score_completions.

Judge-side pieces (PendingPair, PendingQueue, VLLMBatchedJudge, HFBatchedJudge,
build_judge, all in conpo/judge.py) have their own coverage in
tests/test_batched_veto.py; this file only exercises what conpo/trainer.py
does with them.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl import GRPOConfig, GRPOTrainer

import conpo.operators  # noqa: F401  registers all corruption operators
from conpo import config, forward
from conpo.arms import resolve_arm
from conpo.data import PoolRow, TokenMeter
from conpo.edges import Edge, EdgeBuffer, FeasibleBuffer
from conpo.engine import FakeEngine, HFEngine
from conpo.judge import Verdict, VetoRequest
from conpo.operators.registry import FIXTURES
from conpo.selection import FamilyStats
from conpo.trainer import ConPOTrainer, PendingPair, _QueueingVetoCollector
from conpo.verifiers import Constraint

# ---------------------------------------------------------------------------
# Shared fixtures: same tiny-model / row shape as tests/test_trainer_tiny.py
# ---------------------------------------------------------------------------

NO_COMMA = Constraint("punctuation:no_comma", {})
LOWERCASE = Constraint("change_case:english_lowercase", {})


def _tiny_model_and_tokenizer() -> tuple[Qwen3ForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(config.POLICY_MODEL)
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
    return Qwen3ForCausalLM(cfg), tokenizer


def _lora_config() -> LoraConfig:
    return LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=list(config.LORA_TARGET_MODULES), task_type="CAUSAL_LM"
    )


def _make_row(key: str, base: str) -> PoolRow:
    sentence_no_comma = "Do not use any commas."
    sentence_lowercase = "Write your entire answer in lowercase letters."
    prompt = f"{base} {sentence_no_comma} {sentence_lowercase}"
    return PoolRow(
        key=key,
        base_prompt=base,
        prompt=prompt,
        sentences=(sentence_no_comma, sentence_lowercase),
        constraints=(NO_COMMA, LOWERCASE),
        english=True,
        num_constraints=2,
    )


ROW_A = _make_row("batched-veto-row-a", "Explain how gravity works.")


def _engine_factory(trainer: ConPOTrainer) -> HFEngine:
    return HFEngine(trainer.accelerator.unwrap_model(trainer.model), trainer.processing_class, device="cpu")


def _grpo_config(tmp_path: Path, max_steps: int, per_device: int = 2, num_generations: int = 2) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=per_device,
        num_generations=num_generations,
        max_completion_length=16,
        max_steps=max_steps,
        use_vllm=False,
        beta=config.KL_BETA,
        loss_type="grpo",
        learning_rate=1e-3,
        report_to=[],
        logging_steps=1,
        gradient_accumulation_steps=1,
        chat_template_kwargs=config.POLICY_CHAT_TEMPLATE_KWARGS,
        seed=0,
        use_cpu=True,
        bf16=False,
        fp16=False,
        # HF Trainer forces a checkpoint save at the end of training even
        # under the default save_steps=500 with max_steps<500; that would be
        # its own (entirely valid) flush trigger and would confound tests
        # that isolate the periodic/queue-max/explicit flush conditions.
        save_strategy="no",
    )


class _RecordingJudge:
    """Judge that answers every request the same way and records every
    veto_many call, in order, so a test can assert exactly what the batched
    veto sent it."""

    def __init__(self, keep: bool = True) -> None:
        self.keep = keep
        self.calls: list[list[VetoRequest]] = []

    def veto(self, req: VetoRequest) -> Verdict:
        return self.veto_many([req])[0]

    def veto_many(self, reqs: Sequence[VetoRequest]) -> list[Verdict]:
        self.calls.append(list(reqs))
        return [Verdict(keep=self.keep, reason="test", raw="", parse_ok=True) for _ in reqs]


class _SelectiveJudge:
    """Keeps every request except those whose y_plus is in `reject_y_plus`,
    for tests that need a mixed-verdict burst rather than _RecordingJudge's
    uniform answer."""

    def __init__(self, reject_y_plus: set[str]) -> None:
        self.reject_y_plus = reject_y_plus
        self.calls: list[list[VetoRequest]] = []

    def veto(self, req: VetoRequest) -> Verdict:
        return self.veto_many([req])[0]

    def veto_many(self, reqs: Sequence[VetoRequest]) -> list[Verdict]:
        self.calls.append(list(reqs))
        return [
            Verdict(keep=r.y_plus not in self.reject_y_plus, reason="test", raw="", parse_ok=True) for r in reqs
        ]


class _MixedJudge:
    """FakeJudge with three verdict shapes keyed off request.y_plus: a name
    in `parse_fail_y_plus` gets a fail-closed (parse_ok=False) verdict as a
    real judge would return on an unparseable reply from a small model; a
    name in `reject_y_plus` gets a real (parse_ok=True) rejection; anything
    else is kept. Lets a test tell "the judge's reply never parsed" apart
    from "the judge said this pair is degenerate" in one burst.
    """

    def __init__(self, parse_fail_y_plus: set[str] = frozenset(), reject_y_plus: set[str] = frozenset()) -> None:
        self.parse_fail_y_plus = parse_fail_y_plus
        self.reject_y_plus = reject_y_plus
        self.calls: list[list[VetoRequest]] = []

    def veto(self, req: VetoRequest) -> Verdict:
        return self.veto_many([req])[0]

    def veto_many(self, reqs: Sequence[VetoRequest]) -> list[Verdict]:
        self.calls.append(list(reqs))
        verdicts: list[Verdict] = []
        for r in reqs:
            if r.y_plus in self.parse_fail_y_plus:
                verdicts.append(
                    Verdict(
                        keep=False,
                        reason="judge reply did not parse (fail-closed)",
                        raw="garbled reply text",
                        parse_ok=False,
                    )
                )
            elif r.y_plus in self.reject_y_plus:
                verdicts.append(Verdict(keep=False, reason="test-reject", raw="", parse_ok=True))
            else:
                verdicts.append(Verdict(keep=True, reason="test-keep", raw="", parse_ok=True))
        return verdicts


class _FakeVLLMEngine:
    """Records .sleep()/.wake_up() calls, standing in for the real
    vllm.LLM handle _sleep_engine_if_enabled/_wake_engine reach through
    trainer.engine.trainer.vllm_generation.llm."""

    def __init__(self) -> None:
        self.sleep_calls: list[int] = []
        self.wake_calls: int = 0

    def sleep(self, level: int = 1) -> None:
        self.sleep_calls.append(level)

    def wake_up(self, *args: object, **kwargs: object) -> None:
        self.wake_calls += 1


class _FakeColocateHandle:
    """Fakes the `trainer.trainer.vllm_generation.llm` attribute chain
    _sleep_engine_if_enabled/_wake_engine read off `self.engine`."""

    def __init__(self, llm: _FakeVLLMEngine) -> None:
        self.trainer = type("_FakeGRPOTrainer", (), {"vllm_generation": type("_FakeGen", (), {"llm": llm})()})()


def _minimal_trainer(tmp_path: Path, judge: object, **trainer_kwargs: object) -> ConPOTrainer:
    """A ConPOTrainer on the "solve_only" arm (no forward/reverse construction),
    for tests that drive conpo.pending_queue / flush_pending_veto directly
    rather than through real construction."""
    model, tokenizer = _tiny_model_and_tokenizer()
    rows_by_key = {ROW_A.key: ROW_A}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": ROW_A.prompt}], "key": ROW_A.key}])
    arm, overrides = resolve_arm("solve_only")
    trainer = ConPOTrainer(
        model=model,
        args=_grpo_config(tmp_path, max_steps=1),
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=EdgeBuffer(),
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=judge,
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=overrides,
        peft_config=_lora_config(),
        **trainer_kwargs,
    )
    return trainer


def _hand_built_edge(created_update: int) -> tuple[Edge, VetoRequest]:
    edge = Edge(
        prompt_id=ROW_A.key,
        prompt=ROW_A.prompt,
        y_plus="a clean response",
        y_minus="A dirty, response",
        families=frozenset({NO_COMMA.iid}),
        direction="reverse",
        step=0,
        logp_ref_plus=float("nan"),
        logp_ref_minus=float("nan"),
        len_plus=0,
        len_minus=0,
        created_update=created_update,
        meta={"distance": 0.1},
    )
    req = VetoRequest(
        prompt=ROW_A.prompt,
        y_plus=edge.y_plus,
        y_minus=edge.y_minus,
        families=(NO_COMMA.iid,),
        descriptions=("no comma",),
    )
    return edge, req


# ---------------------------------------------------------------------------
# flush_pending_veto mechanics, driven directly (no real construction)
# ---------------------------------------------------------------------------


def test_flush_pending_veto_scores_every_pair_and_preserves_construction_step_age(tmp_path: Path) -> None:
    judge = _RecordingJudge(keep=True)
    trainer = _minimal_trainer(tmp_path, judge)
    edge, req = _hand_built_edge(created_update=3)
    trainer.pending_queue.push(PendingPair(edge=edge, request=req, direction="reverse"))

    trainer.state.global_step = 50
    trainer.flush_pending_veto()

    assert judge.calls == [[req]]
    assert len(trainer.pending_queue) == 0
    assert len(trainer.edge_buffer) == 1

    [pushed] = trainer.edge_buffer.sample(1, current_update=50, rng=random.Random(0), balanced=False)
    assert pushed.created_update == 3  # construction step, not the flush step
    assert trainer.edge_buffer.age(pushed, 50) == 47
    assert not math.isnan(pushed.logp_ref_plus)
    assert not math.isnan(pushed.logp_ref_minus)

    lines = [json.loads(line) for line in (tmp_path / "mechanism.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    record = lines[0]
    assert record["event"] == "veto_flush"
    assert record["pending_size_before"] == 1
    assert record["burst_kept_reverse"] == 1
    assert record["burst_kept_forward"] == 0
    assert record["burst_rejected_forward"] == 0
    assert record["burst_rejected_reverse"] == 0
    assert record["veto_rate_reverse"] == 0.0
    assert record["veto_rate_forward"] == 0.0
    assert record["burst_seconds"] >= 0.0
    assert record["burst_mean_edit_distance"] == pytest.approx(0.1)  # _hand_built_edge's meta["distance"]


def test_flush_pending_veto_reject_all_leaves_edge_buffer_empty(tmp_path: Path) -> None:
    judge = _RecordingJudge(keep=False)
    trainer = _minimal_trainer(tmp_path, judge)
    edge, req = _hand_built_edge(created_update=0)
    trainer.pending_queue.push(PendingPair(edge=edge, request=req, direction="forward"))

    trainer.flush_pending_veto()

    assert judge.calls == [[req]]  # the judge did see the pair
    assert len(trainer.edge_buffer) == 0  # but nothing was admitted
    record = json.loads((tmp_path / "mechanism.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["burst_kept_forward"] == 0
    assert record["burst_rejected_forward"] == 1
    assert record["veto_rate_forward"] == 1.0
    # Item 4 fix: burst_mean_edit_distance averages every pair judged this
    # burst, instead of only the ones kept -- a rejected pair still has a
    # distance (_hand_built_edge's meta["distance"] is 0.1), so this must
    # not read 0.0 even though nothing was kept.
    assert record["burst_mean_edit_distance"] == pytest.approx(0.1)


def test_flush_pending_veto_is_a_noop_on_an_empty_queue(tmp_path: Path) -> None:
    judge = _RecordingJudge(keep=True)
    trainer = _minimal_trainer(tmp_path, judge)

    trainer.flush_pending_veto()

    assert judge.calls == []
    assert not (tmp_path / "mechanism.jsonl").exists()


# ---------------------------------------------------------------------------
# Batched-veto diagnostics (design 2026-09-04): Verdict.parse_ok separating
# parse failures from real rejections, burst_mean_edit_distance over every
# judged pair rather than kept ones alone, and the judge_replies.jsonl dump.
# ---------------------------------------------------------------------------


def test_flush_pending_veto_counts_parse_failures_separately_from_real_rejections(tmp_path: Path) -> None:
    edge_a, req_a = _hand_built_edge(created_update=0)  # kept
    edge_b = dataclasses.replace(edge_a, y_plus="rejected response")
    req_b = dataclasses.replace(req_a, y_plus=edge_b.y_plus)  # real reject
    edge_c = dataclasses.replace(edge_a, y_plus="unparseable response")
    req_c = dataclasses.replace(req_a, y_plus=edge_c.y_plus)  # parse failure
    judge = _MixedJudge(parse_fail_y_plus={edge_c.y_plus}, reject_y_plus={edge_b.y_plus})
    trainer = _minimal_trainer(tmp_path, judge)
    trainer.pending_queue.push(PendingPair(edge=edge_a, request=req_a, direction="reverse"))
    trainer.pending_queue.push(PendingPair(edge=edge_b, request=req_b, direction="reverse"))
    trainer.pending_queue.push(PendingPair(edge=edge_c, request=req_c, direction="forward"))

    trainer.flush_pending_veto()

    record = json.loads((tmp_path / "mechanism.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["burst_kept_reverse"] == 1
    assert record["burst_rejected_reverse"] == 1
    # A parse failure still counts as a rejection (fail-closed), so old
    # readers of burst_rejected_* keep seeing the same total.
    assert record["burst_rejected_forward"] == 1
    assert record["burst_parse_failures_forward"] == 1
    assert record["burst_parse_failures_reverse"] == 0
    assert len(trainer.edge_buffer) == 1  # only edge_a was admitted


def test_flush_pending_veto_burst_mean_edit_distance_averages_all_judged_pairs(tmp_path: Path) -> None:
    """Item 4 fix: previously burst_mean_edit_distance only averaged kept
    pairs' distances, reading 0.0 whenever a burst rejected everything even
    though every judged pair had a nonzero distance. It must now average
    every pair the judge saw this burst."""
    edge_a, req_a = _hand_built_edge(created_update=0)  # meta["distance"] == 0.1, kept
    edge_b = dataclasses.replace(edge_a, y_plus="second response", meta={"distance": 0.3})
    req_b = dataclasses.replace(req_a, y_plus=edge_b.y_plus)  # rejected
    judge = _MixedJudge(reject_y_plus={edge_b.y_plus})
    trainer = _minimal_trainer(tmp_path, judge)
    trainer.pending_queue.push(PendingPair(edge=edge_a, request=req_a, direction="reverse"))
    trainer.pending_queue.push(PendingPair(edge=edge_b, request=req_b, direction="reverse"))

    trainer.flush_pending_veto()

    record = json.loads((tmp_path / "mechanism.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["burst_mean_edit_distance"] == pytest.approx((0.1 + 0.3) / 2)


def test_flush_pending_veto_dumps_up_to_8_judge_replies_with_expected_keys(tmp_path: Path) -> None:
    judge = _RecordingJudge(keep=True)
    trainer = _minimal_trainer(tmp_path, judge, out_dir=tmp_path)
    base_edge, base_req = _hand_built_edge(created_update=0)
    for i in range(10):
        edge = dataclasses.replace(base_edge, y_plus=f"response {i}")
        req = dataclasses.replace(base_req, y_plus=edge.y_plus)
        trainer.pending_queue.push(PendingPair(edge=edge, request=req, direction="reverse"))

    trainer.flush_pending_veto()

    lines = [json.loads(line) for line in (tmp_path / "judge_replies.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 8  # at most JUDGE_REPLY_DUMP_PER_BURST, not all 10
    expected_keys = {
        "step",
        "direction",
        "families",
        "keep",
        "parse_ok",
        "raw",
        "prompt_chars",
        "y_plus_chars",
        "y_minus_chars",
    }
    for record in lines:
        assert set(record.keys()) == expected_keys
        assert record["direction"] == "reverse"
        assert record["keep"] is True
        assert record["parse_ok"] is True


def test_flush_pending_veto_dump_does_not_pad_below_the_pair_count(tmp_path: Path) -> None:
    judge = _RecordingJudge(keep=True)
    trainer = _minimal_trainer(tmp_path, judge, out_dir=tmp_path)
    edge, req = _hand_built_edge(created_update=0)
    trainer.pending_queue.push(PendingPair(edge=edge, request=req, direction="reverse"))

    trainer.flush_pending_veto()

    lines = (tmp_path / "judge_replies.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1


def test_flush_pending_veto_dump_is_a_noop_without_out_dir(tmp_path: Path) -> None:
    judge = _RecordingJudge(keep=True)
    trainer = _minimal_trainer(tmp_path, judge)  # no out_dir given
    edge, req = _hand_built_edge(created_update=0)
    trainer.pending_queue.push(PendingPair(edge=edge, request=req, direction="reverse"))

    trainer.flush_pending_veto()

    assert not (tmp_path / "judge_replies.jsonl").exists()


# ---------------------------------------------------------------------------
# Round2 finding 1: a forward trajectory's feasible endpoint must wait for
# every one of its source edges' real veto verdict before feasible_buffer
# sees it, not the collector's placeholder keep=True -- otherwise
# reverse.reverse_batch could sample a response the real judge never
# approved, in the same step it was built.
# ---------------------------------------------------------------------------


def test_deferred_feasible_endpoint_reaches_feasible_buffer_only_after_all_edges_kept(tmp_path: Path) -> None:
    judge = _RecordingJudge(keep=True)
    trainer = _minimal_trainer(tmp_path, judge)
    edge_a, req_a = _hand_built_edge(created_update=0)
    edge_b, req_b = _hand_built_edge(created_update=0)
    response = "the fully feasible trajectory endpoint"

    trainer._defer_feasible_endpoint(ROW_A.key, response, [edge_a, edge_b])
    trainer.pending_queue.push(PendingPair(edge=edge_a, request=req_a, direction="forward"))

    trainer.flush_pending_veto()
    assert len(trainer.feasible_buffer) == 0  # edge_a alone kept, edge_b's verdict is still outstanding

    trainer.pending_queue.push(PendingPair(edge=edge_b, request=req_b, direction="forward"))
    trainer.flush_pending_veto()

    assert trainer.feasible_buffer._by_prompt == {ROW_A.key: [response]}
    assert len(trainer.edge_buffer) == 2  # both source edges were admitted too


def test_deferred_feasible_endpoint_dropped_when_any_source_edge_is_rejected(tmp_path: Path) -> None:
    edge_a, req_a = _hand_built_edge(created_update=0)
    edge_b = dataclasses.replace(edge_a, y_plus="a different clean response")
    req_b = dataclasses.replace(req_a, y_plus=edge_b.y_plus)
    judge = _SelectiveJudge(reject_y_plus={edge_b.y_plus})
    trainer = _minimal_trainer(tmp_path, judge)
    response = "endpoint built on a trajectory with one rejected transition"

    trainer._defer_feasible_endpoint(ROW_A.key, response, [edge_a, edge_b])
    trainer.pending_queue.push(PendingPair(edge=edge_a, request=req_a, direction="forward"))
    trainer.pending_queue.push(PendingPair(edge=edge_b, request=req_b, direction="forward"))

    trainer.flush_pending_veto()

    assert len(trainer.feasible_buffer) == 0  # one rejected sibling kills the whole endpoint
    assert len(trainer.edge_buffer) == 1  # edge_a was still admitted on its own merits


def test_save_checkpoint_flushes_pending_queue_before_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = _RecordingJudge(keep=True)
    trainer = _minimal_trainer(tmp_path, judge)
    order: list[str] = []
    monkeypatch.setattr(ConPOTrainer, "flush_pending_veto", lambda self: order.append("flush"))
    monkeypatch.setattr(GRPOTrainer, "_save_checkpoint", lambda self, model, trial: order.append("save"))

    trainer._save_checkpoint(trainer.model, None)

    assert order == ["flush", "save"]


# ---------------------------------------------------------------------------
# End to end through _generate_and_score_completions: real forward/reverse
# construction pushes into pending_queue, RUN_VETO_PERIOD (veto_period_k)
# drives the periodic flush.
# ---------------------------------------------------------------------------


def _reverse_only_trainer(
    tmp_path: Path, judge: object, monkeypatch: pytest.MonkeyPatch, veto_period_k: int
) -> tuple[ConPOTrainer, EdgeBuffer]:
    monkeypatch.setattr(config, "REVERSE_DRAWS_PER_STEP", 2)
    model, tokenizer = _tiny_model_and_tokenizer()
    rows_by_key = {ROW_A.key: ROW_A}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": ROW_A.prompt}], "key": ROW_A.key}])
    feasible_buffer = FeasibleBuffer()
    feasible_buffer.add(ROW_A.key, FIXTURES[LOWERCASE.iid](LOWERCASE, random.Random(0)))
    arm, overrides = resolve_arm("reverse_only")
    edge_buffer = EdgeBuffer()

    trainer = ConPOTrainer(
        model=model,
        args=_grpo_config(tmp_path, max_steps=2),
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=feasible_buffer,
        stats=FamilyStats(),
        judge=judge,
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=overrides,
        veto_period_k=veto_period_k,
        peft_config=_lora_config(),
    )
    return trainer, edge_buffer


def test_batched_veto_end_to_end_keep_all_queues_and_admits_real_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = _RecordingJudge(keep=True)
    trainer, edge_buffer = _reverse_only_trainer(tmp_path, judge, monkeypatch, veto_period_k=1)

    trainer.train()

    assert trainer.state.global_step == 2
    total_requests = sum(len(c) for c in judge.calls)
    assert total_requests > 0  # real reverse construction produced pending pairs
    assert len(edge_buffer) == total_requests  # keep-all: every scored pair was admitted
    assert len(trainer.pending_queue) == 0  # RUN_VETO_PERIOD=1 flushed every step

    records = [json.loads(line) for line in (tmp_path / "mechanism.jsonl").read_text(encoding="utf-8").splitlines()]
    flush_records = [r for r in records if r.get("event") == "veto_flush"]
    assert flush_records
    expected = {
        "pending_size_before",
        "burst_seconds",
        "burst_kept_forward",
        "burst_kept_reverse",
        "burst_rejected_forward",
        "burst_rejected_reverse",
        "burst_parse_failures_forward",
        "burst_parse_failures_reverse",
        "veto_rate_forward",
        "veto_rate_reverse",
        "burst_mean_edit_distance",
    }
    for record in flush_records:
        assert expected <= record.keys()


def test_batched_veto_end_to_end_reject_all_leaves_edge_buffer_empty_after_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = _RecordingJudge(keep=False)
    trainer, edge_buffer = _reverse_only_trainer(tmp_path, judge, monkeypatch, veto_period_k=1)

    trainer.train()

    total_requests = sum(len(c) for c in judge.calls)
    assert total_requests > 0  # construction still ran and queued pairs
    assert len(edge_buffer) == 0  # every one of them was rejected


def test_batched_veto_high_period_leaves_pairs_queued_until_explicit_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = _RecordingJudge(keep=True)
    trainer, edge_buffer = _reverse_only_trainer(tmp_path, judge, monkeypatch, veto_period_k=1000)

    trainer.train()

    assert judge.calls == []  # no periodic flush fired within 2 steps
    assert len(edge_buffer) == 0  # invariant: no edge enters a minibatch before its verdict
    assert len(trainer.pending_queue) > 0  # but real construction did queue pairs

    trainer.flush_pending_veto()

    assert judge.calls  # the explicit end-of-training-style flush scored them
    assert len(edge_buffer) == sum(len(c) for c in judge.calls)
    assert len(trainer.pending_queue) == 0


# ---------------------------------------------------------------------------
# _sleep_engine_if_enabled / _wake_engine (review fix: TRL's own generate()
# already sleeps the colocated vLLM engine unconditionally at the end of
# every call under vllm_enable_sleep_mode, so a burst must never call
# .sleep() itself -- only confirm the invariant and wake the engine after).
# ---------------------------------------------------------------------------


def test_flush_pending_veto_never_sleeps_the_engine_and_wakes_it_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUN_USE_VLLM", "1")
    judge = _RecordingJudge(keep=True)
    trainer = _minimal_trainer(tmp_path, judge)
    trainer.args.vllm_enable_sleep_mode = True
    fake_llm = _FakeVLLMEngine()
    trainer.engine = _FakeColocateHandle(fake_llm)
    edge, req = _hand_built_edge(created_update=0)
    trainer.pending_queue.push(PendingPair(edge=edge, request=req, direction="forward"))

    trainer.flush_pending_veto()

    assert fake_llm.sleep_calls == []  # never a second sleep with no intervening wake
    assert fake_llm.wake_calls == 1


def test_flush_pending_veto_skips_sleep_wake_when_sleep_mode_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUN_USE_VLLM", "1")
    judge = _RecordingJudge(keep=True)
    trainer = _minimal_trainer(tmp_path, judge)
    trainer.args.vllm_enable_sleep_mode = False
    fake_llm = _FakeVLLMEngine()
    trainer.engine = _FakeColocateHandle(fake_llm)
    edge, req = _hand_built_edge(created_update=0)
    trainer.pending_queue.push(PendingPair(edge=edge, request=req, direction="forward"))

    trainer.flush_pending_veto()

    assert fake_llm.sleep_calls == []
    assert fake_llm.wake_calls == 0


# ---------------------------------------------------------------------------
# _QueueingVetoCollector: documents (and locks) that under the batched veto,
# forward.py's per-step candidate search never retries past the first
# monotone-admitted candidate -- see the collector's own docstring.
# ---------------------------------------------------------------------------


def test_queueing_veto_collector_never_retries_past_first_monotone_admitted_candidate() -> None:
    y0 = "The platform, rotates at a constant velocity."
    fixes_comma_only = "The platform rotates at a constant velocity."  # still capitalised: LOWERCASE still fails
    fixes_both = "the platform rotates at a constant velocity."  # would also be monotone-admitted, never tried
    row = PoolRow(
        key="qvc-row",
        base_prompt=y0,
        prompt=y0,
        sentences=(),
        constraints=(NO_COMMA, LOWERCASE),
        english=True,
        num_constraints=2,
    )
    engine = FakeEngine([fixes_comma_only, fixes_both])
    collector = _QueueingVetoCollector()
    meter = TokenMeter(budget=0)

    result = forward.repair_trajectory(
        row, y0, engine, collector, random.Random(0), meter, current_update=0, max_steps=1, max_candidates=2
    )

    assert len(collector.captured) == 1  # the second candidate was never even offered to the judge
    assert len(result.edges) == 1
    assert result.edges[0].y_plus == fixes_comma_only  # first monotone-admitted candidate, not the "better" one

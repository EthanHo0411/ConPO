"""ConPOTrainer on a tiny random Qwen3 model (CPU, must finish in well under 3 minutes).

Hand-built PoolRows (not the real dataset), mirroring tests/test_forward.py and
tests/test_reverse.py's established pattern: constraints drawn from real
verifier families so `feasible`/`registry.corrupt` behave exactly as they do
on real data, while every row and edge stays fully under this file's control.
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

import conpo.operators  # noqa: F401  registers all 53 corruption operators
from conpo import config
from conpo.arms import ArmSpec, Overrides, resolve_arm
from conpo.data import PoolRow, TokenMeter
from conpo.distance import token_distance
from conpo.edges import Edge, EdgeBuffer, FeasibleBuffer
from conpo.engine import HFEngine
from conpo.forward import RepairResult
from conpo.judge import StaticJudge
from conpo.operators import registry
from conpo.operators.registry import FIXTURES
from conpo.reverse import ReverseResult
from conpo.selection import FamilyStats
from conpo.trainer import _SAMPLED_MECHANISM_FIELDS, ConPOTrainer, _pairs_this_microbatch, score_response
from conpo.verifiers import Constraint, feasible

# ---------------------------------------------------------------------------
# Fixed model/rows shared by both tests
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


ROW_A = _make_row("edge-row-a", "Explain how gravity works.")
ROW_B = _make_row("edge-row-b", "Describe how photosynthesis works.")
ROW_C = _make_row("edge-row-c", "Summarize the water cycle.")
ROW_D = _make_row("edge-row-d", "Outline the rock cycle.")


def _hand_built_edges() -> list[Edge]:
    """2 reverse + 2 forward edges: a FIXTURES-feasible response and its
    registry.corrupt corruption, one per (row, target family) -- exactly
    what reverse.py admits (y_plus=feasible response, y_minus=corruption),
    reused here to also stand in for a forward repair transition
    (y_minus=pre-repair, y_plus=post-repair)."""
    specs = [(ROW_A, 0, "reverse"), (ROW_A, 1, "reverse"), (ROW_B, 0, "forward"), (ROW_B, 1, "forward")]
    edges = []
    for i, (row, target_index, direction) in enumerate(specs):
        y_plus = FIXTURES[LOWERCASE.iid](LOWERCASE, random.Random(i))
        assert feasible(row.constraints, y_plus)
        corrupted, reason = registry.corrupt(row.constraints, target_index, y_plus, random.Random(i + 100))
        assert corrupted is not None, reason
        family = row.constraints[target_index].iid
        edges.append(
            Edge(
                prompt_id=row.key,
                prompt=row.prompt,
                y_plus=y_plus,
                y_minus=corrupted,
                families=frozenset({family}),
                direction=direction,
                step=0,
                logp_ref_plus=float("nan"),
                logp_ref_minus=float("nan"),
                len_plus=0,
                len_minus=0,
                created_update=0,
                meta={"distance": token_distance(y_plus, corrupted)},
            )
        )
    return edges


def _multi_family_edge(row: PoolRow, seed: int = 0) -> Edge:
    """A reverse edge whose families carry both of `row`'s constraints
    (H6-style joint corruption via registry.corrupt_set), for gap 4's
    L_resp-ungated-on-multi regression test."""
    y_plus = FIXTURES[LOWERCASE.iid](LOWERCASE, random.Random(seed))
    assert feasible(row.constraints, y_plus)
    corrupted = registry.corrupt_set(row.constraints, (0, 1), y_plus, random.Random(seed + 100))
    assert corrupted is not None
    return Edge(
        prompt_id=row.key,
        prompt=row.prompt,
        y_plus=y_plus,
        y_minus=corrupted,
        families=frozenset({NO_COMMA.iid, LOWERCASE.iid}),
        direction="reverse",
        step=0,
        logp_ref_plus=float("nan"),
        logp_ref_minus=float("nan"),
        len_plus=0,
        len_minus=0,
        created_update=0,
        meta={"distance": token_distance(y_plus, corrupted)},
    )


def _canned_edge(direction: str) -> Edge:
    """A minimal placeholder edge for gap 5's arm-direction-gating test,
    where the real construction path is monkeypatched away and only the
    trainer's own routing (which direction's result lands in edge_buffer)
    is under test."""
    return Edge(
        prompt_id=ROW_A.key,
        prompt=ROW_A.prompt,
        y_plus="a response with no comma",
        y_minus="a response, with comma",
        families=frozenset({NO_COMMA.iid}),
        direction=direction,
        step=0,
        logp_ref_plus=float("nan"),
        logp_ref_minus=float("nan"),
        len_plus=0,
        len_minus=0,
        created_update=0,
        meta={"distance": 0.1},
    )


class _CaptureCallback(TrainerCallback):
    """Snapshots trainer.last_loss_stats at the end of every optimizer step
    (before the next compute_loss call overwrites it)."""

    def __init__(self) -> None:
        self.trainer: ConPOTrainer | None = None
        self.per_step: list[dict[str, float]] = []

    def on_step_end(self, args, state, control, **kwargs) -> None:  # noqa: ANN001
        if self.trainer is not None:
            self.per_step.append(dict(self.trainer.last_loss_stats))


def _engine_factory(trainer: ConPOTrainer) -> HFEngine:
    return HFEngine(trainer.accelerator.unwrap_model(trainer.model), trainer.processing_class, device="cpu")


# ---------------------------------------------------------------------------
# arm "full": two optimizer steps, L_resp/L_prompt exercised via hand-built edges
# ---------------------------------------------------------------------------


def test_full_arm_two_optimizer_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # forward.py reads these two constants fresh from `config` on every call
    # (not bound as function defaults), so shrinking them here keeps the
    # repair-trajectory generations this test triggers small and fast.
    monkeypatch.setattr(config, "MAX_COMPLETION_TOKENS", 16)
    monkeypatch.setattr(config, "REVERSE_DRAWS_PER_STEP", 2)

    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B, ROW_C, ROW_D]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=2,
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
    )
    arm, overrides = resolve_arm("full")
    edge_buffer = EdgeBuffer()
    meter = TokenMeter(budget=0)
    mechanism_log_path = tmp_path / "mechanism.jsonl"
    capture = _CaptureCallback()

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=meter,
        mechanism_log_path=mechanism_log_path,
        seed=0,
        overrides=overrides,
        peft_config=_lora_config(),
        callbacks=[capture],
    )
    capture.trainer = trainer

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    filled_edges = []
    with torch.no_grad(), unwrapped.disable_adapter():
        for edge in _hand_built_edges():
            logp_plus, len_plus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_plus)
            logp_minus, len_minus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_minus)
            filled_edges.append(
                dataclasses.replace(
                    edge,
                    logp_ref_plus=logp_plus.item(),
                    logp_ref_minus=logp_minus.item(),
                    len_plus=len_plus,
                    len_minus=len_minus,
                )
            )
    for edge in filled_edges:
        edge_buffer.push(edge)

    lora_before = {name: p.detach().clone() for name, p in trainer.model.named_parameters() if "lora_" in name}
    assert lora_before  # sanity: the adapter actually has parameters

    trainer.train()

    assert trainer.state.global_step == 2

    lora_after = {name: p.detach() for name, p in trainer.model.named_parameters() if "lora_" in name}
    assert any(not torch.equal(lora_before[name], lora_after[name]) for name in lora_before)

    # "event"-tagged lines are the separate step_timing/veto_flush records
    # (Ethan devbox perf pass, 2026-09-11); this test's per-step assertions
    # below are about the compute_loss record, which carries no "event" key.
    records = [json.loads(line) for line in mechanism_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if "event" not in r]
    assert len(records) == 2
    expected_keys = {
        "step",
        "admitted_by_direction",
        "veto_discarded_by_direction",
        "reasons_by_direction",
        "mean_edit_distance",
        "degenerate_group_rate",
        "edges_per_million_tokens_step",
        "edges_per_million_tokens_cumulative",
        "edge_buffer_size",
        "feasible_buffer_size",
        "family_stats",
        "tokens_cumulative",
        "cumulative_admitted_edges",
    }
    for record in records:
        assert expected_keys <= record.keys()
        # K5 RUN_RESAMPLE_ZERO_STD unset (this test's default): the
        # resample_* keys must be absent, not zero, proving the flag-off
        # path never runs _resample_degenerate_groups.
        assert "resample_groups_replaced" not in record
        assert "resample_rounds" not in record
    # Both raw anchors are lifetime cumulative counters (gap 3): never
    # decrease from one step's record to the next.
    assert records[0]["tokens_cumulative"] <= records[1]["tokens_cumulative"]
    assert records[0]["cumulative_admitted_edges"] <= records[1]["cumulative_admitted_edges"]

    assert len(capture.per_step) == 2
    step1_resp_loss = capture.per_step[0]["resp_loss"]
    assert math.isfinite(step1_resp_loss)
    # theta = theta_ref at step 1 (zero-initialised LoRA): Delta_theta(e) = 0
    # for every edge, so L_resp = -logsigmoid(0) = log(2) exactly.
    assert abs(step1_resp_loss - math.log(2)) < 0.05

    assert meter.total > 0  # the clean completion tokens were charged


# ---------------------------------------------------------------------------
# arm "ifrlvr": one step, reward_additive formula checked directly
# ---------------------------------------------------------------------------


def test_ifrlvr_arm_one_step_and_reward_additive(tmp_path: Path) -> None:
    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
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
    )
    arm, overrides = resolve_arm("ifrlvr")

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=EdgeBuffer(),
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=overrides,
        peft_config=_lora_config(),
    )

    trainer.train()

    assert trainer.state.global_step == 1

    # "hello, world": all-lowercase (LOWERCASE passes) but has a comma
    # (NO_COMMA fails) -> 1 of 2 constraints satisfied -> 0.5.
    scores = trainer.reward_additive(completions=["hello, world"], key=[ROW_A.key])
    assert scores == [0.5]


# ---------------------------------------------------------------------------
# compute_loss under gradient accumulation: EDGE_PAIRS_PER_STEP is a
# per-OPTIMIZER-step budget, not a per-compute_loss-call one, and the T_c
# probe must fire once per optimizer step, not once per micro-batch.
# ---------------------------------------------------------------------------


def test_compute_loss_halves_edge_sample_and_runs_probe_once_per_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "EDGE_PAIRS_PER_STEP", 4)

    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
        use_vllm=False,
        beta=config.KL_BETA,
        loss_type="grpo",
        learning_rate=1e-3,
        report_to=[],
        logging_steps=1,
        gradient_accumulation_steps=2,
        chat_template_kwargs=config.POLICY_CHAT_TEMPLATE_KWARGS,
        seed=0,
        use_cpu=True,
        bf16=False,
        fp16=False,
    )
    # forward/reverse off: isolate compute_loss's own sampling and probe
    # cadence from the counterfactual-construction machinery.
    arm = ArmSpec(reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=True)
    edge_buffer = EdgeBuffer()
    mechanism_log_path = tmp_path / "mechanism.jsonl"

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=mechanism_log_path,
        seed=0,
        overrides=Overrides(),
        peft_config=_lora_config(),
    )

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    with torch.no_grad(), unwrapped.disable_adapter():
        for edge in _hand_built_edges():
            logp_plus, len_plus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_plus)
            logp_minus, len_minus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_minus)
            edge_buffer.push(
                dataclasses.replace(
                    edge,
                    logp_ref_plus=logp_plus.item(),
                    logp_ref_minus=logp_minus.item(),
                    len_plus=len_plus,
                    len_minus=len_minus,
                )
            )

    sample_calls: list[int] = []
    original_sample = EdgeBuffer.sample

    def spy_sample(self: EdgeBuffer, n: int, current_update: int, rng: random.Random, balanced: bool = True):
        sample_calls.append(n)
        return original_sample(self, n, current_update, rng, balanced=balanced)

    monkeypatch.setattr(EdgeBuffer, "sample", spy_sample)

    probe_calls: list[int] = []
    trainer._run_probe = lambda model, inputs, clean, current_update: probe_calls.append(current_update)

    trainer.train()

    assert trainer.state.global_step == 1
    # gradient_accumulation_steps=2 -> 2 compute_loss calls for the one
    # optimizer step, each drawing ceil(EDGE_PAIRS_PER_STEP=4 / 2) == 2, not
    # the full 4 (which 16x'd real-run memory/time before this fix).
    assert sample_calls == [2, 2]
    # Both compute_loss calls share global_step == 0 (it advances only after
    # the optimizer step completes), so the probe must fire exactly once.
    assert probe_calls == [0]

    # Round2 finding 2: pair_scoring_seconds/pair_scoring_forwards
    # (conpo/trainer.py's compute_loss) cover _score_edge_pairs's
    # score_response calls -- 2 forwards per resp edge (y_plus, y_minus).
    # Only the accumulation cycle's first compute_loss call's local totals
    # reach the mechanism record (compute_loss clears
    # _pending_mechanism_record right after writing it), and that call
    # samples the same 2 resp_edges as the second (sample_calls == [2, 2]
    # above), so the expected count does not depend on which micro-batch's
    # record survives. This arm has prompt_loss=False, so no
    # 6*len(prompt_edges) term applies.
    records = [
        json.loads(line) for line in mechanism_log_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    records = [r for r in records if "event" not in r]
    assert len(records) == 1
    assert records[0]["pair_scoring_forwards"] == 4
    assert records[0]["pair_scoring_seconds"] > 0.0
    # B2 (audit 2026-09-14): resp_evicted/pair_scoring_seconds/
    # pair_scoring_forwards are documented as single-micro-batch samples on
    # every record, this one included.
    assert set(records[0]["sampled_fields"]) == set(_SAMPLED_MECHANISM_FIELDS)


# ---------------------------------------------------------------------------
# B2 (audit 2026-09-14): pair_scoring_seconds/pair_scoring_forwards/
# resp_evicted above are single-micro-batch samples, not per-optimiser-step
# totals -- only the accumulation cycle's first compute_loss call's local
# value reaches the mechanism record, the other gradient_accumulation_steps
# - 1 calls' contributions are dropped. Fixed by accumulating all three
# across the whole cycle and surfacing the PREVIOUS cycle's total on the
# NEXT cycle's record (previous_step_totals): the current cycle's own total
# is not yet known when the first compute_loss call writes its record.
# ---------------------------------------------------------------------------


def test_mechanism_record_carries_previous_step_pair_totals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "EDGE_PAIRS_PER_STEP", 4)

    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=2,
        use_vllm=False,
        beta=config.KL_BETA,
        loss_type="grpo",
        learning_rate=1e-3,
        report_to=[],
        logging_steps=1,
        gradient_accumulation_steps=2,
        chat_template_kwargs=config.POLICY_CHAT_TEMPLATE_KWARGS,
        seed=0,
        use_cpu=True,
        bf16=False,
        fp16=False,
    )
    # forward/reverse/prompt_loss off, same isolation as
    # test_compute_loss_halves_edge_sample_and_runs_probe_once_per_step above.
    arm = ArmSpec(reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=True)
    edge_buffer = EdgeBuffer()
    mechanism_log_path = tmp_path / "mechanism.jsonl"

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=mechanism_log_path,
        seed=0,
        overrides=Overrides(),
        peft_config=_lora_config(),
        probe_enabled=False,
    )

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    with torch.no_grad(), unwrapped.disable_adapter():
        for edge in _hand_built_edges():
            logp_plus, len_plus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_plus)
            logp_minus, len_minus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_minus)
            edge_buffer.push(
                dataclasses.replace(
                    edge,
                    logp_ref_plus=logp_plus.item(),
                    logp_ref_minus=logp_minus.item(),
                    len_plus=len_plus,
                    len_minus=len_minus,
                )
            )

    # Ground truth: every compute_loss call's own resp_evicted, independent
    # of the fix under test (spies the real discard, does not replace it).
    evicted_calls: list[int] = []
    original_discard = EdgeBuffer.discard

    def spy_discard(self: EdgeBuffer, edges: object) -> int:
        n = original_discard(self, edges)
        evicted_calls.append(n)
        return n

    monkeypatch.setattr(EdgeBuffer, "discard", spy_discard)

    trainer.train()

    assert trainer.state.global_step == 2
    records = _mechanism_records(mechanism_log_path)
    assert len(records) == 2
    # gradient_accumulation_steps=2 -> 2 compute_loss calls per optimiser
    # step -> 2 discard calls per step (edge_buffer stays at 4 edges the
    # whole test: an untrained model's margins start near 0, well under
    # KAPPA*TAU=0.5, and one micro lr=1e-3 step cannot cross that).
    assert len(evicted_calls) == 4

    # Step 0 (records[0]): the run's first cycle, nothing precedes it.
    assert "previous_step_totals" not in records[0]

    # Step 1 (records[1]): step 0's totals, from BOTH of its compute_loss
    # calls, not only the first.
    prev = records[1]["previous_step_totals"]
    assert prev["step"] == 0
    assert prev["resp_evicted_step"] == sum(evicted_calls[:2])
    # ceil(EDGE_PAIRS_PER_STEP=4 / grad_accum=2) == 2 edges/call, 2
    # forwards/edge (no clamp, no span_local) -> 4 forwards/call, 8/step.
    assert prev["pair_scoring_forwards_step"] == 8
    assert prev["pair_scoring_seconds_step"] > 0.0

    for record in records:
        assert set(record["sampled_fields"]) == set(_SAMPLED_MECHANISM_FIELDS)


# ---------------------------------------------------------------------------
# B3 (audit 2026-09-14): conpo/trainer.py's compute_loss used to draw
# math.ceil(config.EDGE_PAIRS_PER_STEP / grad_accum) / ...PROMPT_PAIRS_PER_STEP
# pairs on EVERY micro-batch of the accumulation cycle, so a per-step budget
# that did not divide evenly by grad_accum was overdrawn (e.g.
# PROMPT_PAIRS_PER_STEP=8 over grad_accum=16 drew ceil(8/16)=1 * 16 == 16,
# double the configured 8; EDGE_PAIRS_PER_STEP=48/16=3 stayed exact only
# because 16 divides 48). _pairs_this_microbatch replaces the per-micro-batch
# ceil with an exact floor((i+1)*T/A)-floor(i*T/A) split.
# ---------------------------------------------------------------------------


def test_pairs_this_microbatch_schedule_sums_exact_for_any_ratio() -> None:
    for total, accumulation in ((8, 16), (48, 16), (5, 2)):
        counts = [_pairs_this_microbatch(total, i, accumulation) for i in range(accumulation)]
        assert sum(counts) == total
        assert all(c >= 0 for c in counts)

    # P=8, A=16: budget below accumulation, at most one pair per micro-batch
    # (the old ceil(8/16)=1 drawn on all 16 micro-batches summed to 16).
    assert set(_pairs_this_microbatch(8, i, 16) for i in range(16)) <= {0, 1}

    # P=48, A=16: divides evenly, exact 3 on every micro-batch -- unchanged
    # from the old ceil(48/16)=3, the one case it already got right.
    assert [_pairs_this_microbatch(48, i, 16) for i in range(16)] == [3] * 16

    # P=5, A=2: does not divide evenly; old ceil(5/2)=2 on both micro-batches
    # would have summed to 4, undercounting (a case the ceil formula could
    # also get wrong in the other direction).
    assert [_pairs_this_microbatch(5, i, 2) for i in range(2)] == [2, 3]


def test_edge_and_prompt_pairs_step_totals_exact_for_uneven_accumulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # grad_accum=2 (the same value the B2 tests above already exercise
    # end-to-end with this dataset/config shape) divides neither
    # EDGE_PAIRS_PER_STEP=3 nor PROMPT_PAIRS_PER_STEP=5 (both monkeypatched).
    monkeypatch.setattr(config, "EDGE_PAIRS_PER_STEP", 3)
    monkeypatch.setattr(config, "PROMPT_PAIRS_PER_STEP", 5)

    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=2,
        use_vllm=False,
        beta=config.KL_BETA,
        loss_type="grpo",
        learning_rate=1e-3,
        report_to=[],
        logging_steps=1,
        gradient_accumulation_steps=2,
        chat_template_kwargs=config.POLICY_CHAT_TEMPLATE_KWARGS,
        seed=0,
        use_cpu=True,
        bf16=False,
        fp16=False,
    )
    # forward/reverse off, same isolation as the B2 test above; prompt_loss
    # on this time so both edge_n and prompt_n are exercised.
    arm = ArmSpec(reward="conjunctive", forward=False, reverse=False, prompt_loss=True, resp_loss=True)
    edge_buffer = EdgeBuffer()
    mechanism_log_path = tmp_path / "mechanism.jsonl"

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=mechanism_log_path,
        seed=0,
        overrides=Overrides(),
        peft_config=_lora_config(),
        probe_enabled=False,
    )

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    with torch.no_grad(), unwrapped.disable_adapter():
        for edge in _hand_built_edges():
            logp_plus, len_plus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_plus)
            logp_minus, len_minus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_minus)
            edge_buffer.push(
                dataclasses.replace(
                    edge,
                    logp_ref_plus=logp_plus.item(),
                    logp_ref_minus=logp_minus.item(),
                    len_plus=len_plus,
                    len_minus=len_minus,
                )
            )

    # 2 forward + 2 reverse, all single-family -- both EdgeBuffer.sample
    # (balanced, requested <= 2) and .sample_where (requested <= 3, pool of
    # 4 single-family edges) always have enough edges to return exactly
    # what they are asked for, so the requested and realised counts below
    # coincide.
    edge_n_calls: list[int] = []
    original_sample = EdgeBuffer.sample

    def spy_sample(self: EdgeBuffer, n: int, current_update: int, rng: random.Random, balanced: bool = True):
        edge_n_calls.append(n)
        return original_sample(self, n, current_update, rng, balanced=balanced)

    monkeypatch.setattr(EdgeBuffer, "sample", spy_sample)

    prompt_n_calls: list[int] = []
    original_sample_where = EdgeBuffer.sample_where

    def spy_sample_where(self: EdgeBuffer, predicate, n: int, current_update: int, rng: random.Random):
        prompt_n_calls.append(n)
        return original_sample_where(self, predicate, n, current_update, rng)

    monkeypatch.setattr(EdgeBuffer, "sample_where", spy_sample_where)

    trainer.train()

    assert trainer.state.global_step == 2
    # gradient_accumulation_steps=2 -> 2 compute_loss calls per optimiser
    # step -> 2 sample()/sample_where() calls per step, 4 total over 2 steps.
    assert len(edge_n_calls) == 4
    assert len(prompt_n_calls) == 4
    # Each optimiser step's own 2 calls sum to exactly the configured
    # budget -- the old per-call ceil(3/2)=2 * 2 == 4 and ceil(5/2)=3 * 2 ==
    # 6 would both have overdrawn.
    assert sum(edge_n_calls[:2]) == sum(edge_n_calls[2:]) == config.EDGE_PAIRS_PER_STEP == 3
    assert sum(prompt_n_calls[:2]) == sum(prompt_n_calls[2:]) == config.PROMPT_PAIRS_PER_STEP == 5

    records = _mechanism_records(mechanism_log_path)
    assert len(records) == 2
    assert "previous_step_totals" not in records[0]

    # Step 1's record carries step 0's realised (post-sample) totals, which
    # match the requested totals above since the buffer never runs dry.
    prev = records[1]["previous_step_totals"]
    assert prev["step"] == 0
    assert prev["edge_pairs_step"] == 3
    assert prev["prompt_pairs_step"] == 5


# ---------------------------------------------------------------------------
# compute_loss saturated-edge eviction (conpo/trainer.py): an edge whose
# margin already clears kappa*tau carries weight 0 in resp_loss forever
# after (delta only grows as training proceeds), so compute_loss discards it
# from edge_buffer right after _score_edge_pairs instead of resampling it
# every future step until age_cap retires it.
# ---------------------------------------------------------------------------


def test_compute_loss_evicts_saturated_edge_and_records_resp_evicted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "EDGE_PAIRS_PER_STEP", 2)

    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
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
    )
    # forward/reverse/prompt_loss off: isolate compute_loss's own eviction
    # branch from the counterfactual-construction machinery and L_prompt.
    arm = ArmSpec(reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=True)
    edge_buffer = EdgeBuffer()
    # Both edges "forward" (none "reverse"): sample()'s balanced=True falls
    # back to the single present direction, so both are drawn deterministically.
    for edge in _hand_built_edges()[2:]:
        edge_buffer.push(edge)
    assert len(edge_buffer) == 2

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=Overrides(),
        peft_config=_lora_config(),
    )
    trainer._run_probe = lambda *args, **kwargs: None

    # One margin above kappa*tau (must be evicted), one at/below it (kept);
    # ages don't matter to eviction so both are 0.
    margin_high = config.KAPPA * config.TAU + 0.1
    margin_low = config.KAPPA * config.TAU - 0.1

    def fake_score_edge_pairs(
        model: torch.nn.Module, edges: list
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        assert len(edges) == 2
        return torch.tensor([margin_high, margin_low]), torch.zeros(len(edges), dtype=torch.long), {}

    trainer._score_edge_pairs = fake_score_edge_pairs

    trainer.train()

    assert trainer.state.global_step == 1
    assert len(edge_buffer) == 1  # the saturated edge was discarded
    assert trainer.last_loss_stats["resp_evicted"] == 1


# ---------------------------------------------------------------------------
# _run_probe's memory bound (task H, Ethan devbox, 2026-09-11): the clean
# side is now a fresh, small resample of the step's own batch instead of a
# retain_graph=True differentiation of the full-batch `clean` tensor.
# ---------------------------------------------------------------------------


def test_run_probe_records_a_cosine(tmp_path: Path) -> None:
    """The real (unstubbed) probe -- resampled clean-side gradient, both
    gradients taken with retain_graph=False -- runs end to end on CPU and
    records a finite T_c cosine for whichever family it picks. This does
    not prove no extra CUDA graph is kept alive (this suite is CPU-only, no
    memory introspection available), only that the resample path is real
    and correct enough to run and record, per task H's own fallback bar.
    """
    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
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
    )
    arm = ArmSpec(reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=True)
    edge_buffer = EdgeBuffer()
    stats = FamilyStats()

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=stats,
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=Overrides(),
        peft_config=_lora_config(),
        # probe_enabled defaults True: the real probe path runs, unstubbed.
    )

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    with torch.no_grad(), unwrapped.disable_adapter():
        for edge in _hand_built_edges():
            logp_plus, len_plus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_plus)
            logp_minus, len_minus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_minus)
            edge_buffer.push(
                dataclasses.replace(
                    edge,
                    logp_ref_plus=logp_plus.item(),
                    logp_ref_minus=logp_minus.item(),
                    len_plus=len_plus,
                    len_minus=len_minus,
                )
            )

    trainer.train()

    probed_families = [f for f in (NO_COMMA.iid, LOWERCASE.iid) if stats.last_probe_update(f) is not None]
    assert probed_families  # the probe fired and picked one of the two families
    for family in probed_families:
        assert stats.last_probe_update(family) == 0
        assert math.isfinite(stats.t_c(family))


# ---------------------------------------------------------------------------
# _generate_and_score_completions must push every RepairResult.feasible_endpoint
# from forward repair into feasible_buffer (review-1 major, re-verified here).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("repair_batch_generation", [True, False])
def test_forward_feasible_endpoints_reach_feasible_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repair_batch_generation: bool
) -> None:
    # Force every generated response to fail every verifier so it is always
    # routed into failed_responses, independent of what the tiny random
    # model actually generates, and forward repair always runs.
    monkeypatch.setattr("conpo.trainer.verifier_vector", lambda constraints, text: [False] * len(constraints))
    sentinel = "SENTINEL_FEASIBLE_ENDPOINT"
    # endpoint_sources=[[]]: this endpoint has no source edges (mirrors
    # repair_trajectory's already-feasible-input shape), so under the batched
    # veto (the default judge placement here) it goes straight into
    # feasible_buffer with no verdict to wait for -- see
    # test_batched_veto_forward_endpoint_withheld_until_its_edges_are_kept in
    # tests/test_batched_veto_trainer.py for the withheld-until-kept case.
    # RUN_REPAIR_BATCH_GENERATION (L2) picks which forward.py function
    # _generate_and_score_completions' per-group loop reaches:
    # repair_all_groups (default, "1") or the old per-group
    # repair_failed_group ("0") -- patch whichever one this run's flag
    # setting will call, both returning one RepairResult per group.
    monkeypatch.setenv("RUN_REPAIR_BATCH_GENERATION", "1" if repair_batch_generation else "0")
    if repair_batch_generation:
        monkeypatch.setattr(
            "conpo.forward.repair_all_groups",
            lambda groups, *args, **kwargs: [
                RepairResult(feasible_endpoints=[sentinel], endpoint_sources=[[]]) for _ in groups
            ],
        )
    else:
        monkeypatch.setattr(
            "conpo.forward.repair_failed_group",
            lambda *args, **kwargs: RepairResult(feasible_endpoints=[sentinel], endpoint_sources=[[]]),
        )

    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
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
    )
    arm, overrides = resolve_arm("forward_only")
    feasible_buffer = FeasibleBuffer()

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=EdgeBuffer(),
        feasible_buffer=feasible_buffer,
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=overrides,
        peft_config=_lora_config(),
        probe_enabled=False,
    )

    trainer.train()

    # per_device_train_batch_size == num_generations with
    # gradient_accumulation_steps=1 means one optimizer step samples exactly
    # one row's group, so only one of ROW_A/ROW_B's keys is populated -- but
    # every group is all-failing (patched verifier_vector), so the only way
    # feasible_buffer gets populated at all is via RepairResult.feasible_endpoints,
    # confirming trainer.py wires it through.
    assert feasible_buffer._by_prompt in ({ROW_A.key: [sentinel]}, {ROW_B.key: [sentinel]})


# ---------------------------------------------------------------------------
# gap 2: the H5 prompt diagnostic (prompt_gamma_theta) is computed and
# logged even when RUN_LAMBDA_P=0 zeroes out L_prompt's weighted contribution.
# ---------------------------------------------------------------------------


def test_prompt_gamma_theta_logged_when_lambda_p_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MAX_COMPLETION_TOKENS", 16)
    monkeypatch.setattr(config, "REVERSE_DRAWS_PER_STEP", 2)

    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
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
    )
    arm, _ = resolve_arm("full")
    edge_buffer = EdgeBuffer()

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=Overrides(lambda_p=0.0),  # RUN_LAMBDA_P=0
        peft_config=_lora_config(),
        probe_enabled=False,
    )
    assert trainer.overrides.lambda_p == 0.0

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    with torch.no_grad(), unwrapped.disable_adapter():
        for edge in _hand_built_edges():
            logp_plus, len_plus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_plus)
            logp_minus, len_minus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_minus)
            edge_buffer.push(
                dataclasses.replace(
                    edge,
                    logp_ref_plus=logp_plus.item(),
                    logp_ref_minus=logp_minus.item(),
                    len_plus=len_plus,
                    len_minus=len_minus,
                )
            )

    trainer.train()

    # lambda_p == 0 must not stop the diagnostic from being computed: only
    # its weighted contribution to `total` is skipped.
    assert "prompt_gamma_theta" in trainer.last_loss_stats
    assert math.isfinite(trainer.last_loss_stats["prompt_gamma_theta"])

    # The per-step mechanism record (mechanism.jsonl) carries the same
    # diagnostic, not only the in-memory last_loss_stats: compute_loss merges
    # its prompt_* stats into the record before it is written.
    # "event"-tagged lines are the separate step_timing/veto_flush records
    # (Ethan devbox perf pass, 2026-09-11); the per-step compute_loss record
    # this test checks carries no "event" key.
    records = [
        json.loads(line)
        for line in (tmp_path / "mechanism.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    step_records = [r for r in records if "event" not in r]
    assert step_records
    assert "prompt_gamma_theta" in step_records[-1]
    assert math.isfinite(step_records[-1]["prompt_gamma_theta"])


# ---------------------------------------------------------------------------
# gap 4: L_resp's edge sample stays ungated on Overrides.multi -- a
# multi-family edge is still eligible even with RUN_MULTI unset.
# ---------------------------------------------------------------------------


def test_multi_family_edge_reaches_resp_loss_sample_without_run_multi(tmp_path: Path) -> None:
    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
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
    )
    # forward/reverse off: the only edge compute_loss can possibly draw from
    # is the multi-family one pushed below.
    arm = ArmSpec(reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=True)
    edge_buffer = EdgeBuffer()

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=Overrides(),  # multi=False (RUN_MULTI unset), the default
        peft_config=_lora_config(),
        probe_enabled=False,
    )
    assert trainer.overrides.multi is False

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    with torch.no_grad(), unwrapped.disable_adapter():
        edge = _multi_family_edge(ROW_A)
        logp_plus, len_plus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_plus)
        logp_minus, len_minus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_minus)
        edge_buffer.push(
            dataclasses.replace(
                edge,
                logp_ref_plus=logp_plus.item(),
                logp_ref_minus=logp_minus.item(),
                len_plus=len_plus,
                len_minus=len_minus,
            )
        )

    trainer.train()

    # edge_buffer holds exactly this one edge, so EdgeBuffer.sample is
    # certain to draw it if L_resp's sampling is ungated on family count --
    # the per-family margin stat only appears under key "multi" if it was.
    assert "resp_margin_reverse_multi" in trainer.last_loss_stats


# ---------------------------------------------------------------------------
# gap 5: forward_only/reverse_only must never let the disabled direction's
# construction path run, whatever its own preconditions would otherwise allow.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm_name", ["forward_only", "reverse_only"])
def test_arm_direction_gating_produces_only_the_enabled_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm_name: str
) -> None:
    monkeypatch.setattr("conpo.trainer.verifier_vector", lambda constraints, text: [False] * len(constraints))
    enabled_direction = "forward" if arm_name == "forward_only" else "reverse"
    # reverse_only: patch BOTH conpo.forward functions to raise, not just
    # repair_failed_group -- RUN_REPAIR_BATCH_GENERATION (L2) defaults to
    # "1", so the default call site is repair_all_groups, and this gating
    # check must still catch a regression whichever one a future flag
    # flip reaches.
    disabled_targets = (
        ["conpo.reverse.reverse_batch"]
        if arm_name == "forward_only"
        else ["conpo.forward.repair_failed_group", "conpo.forward.repair_all_groups"]
    )

    def _raise_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"{disabled_targets} must not run when its direction is disabled")

    for disabled_target in disabled_targets:
        monkeypatch.setattr(disabled_target, _raise_if_called)

    if arm_name == "forward_only":
        # RUN_REPAIR_BATCH_GENERATION defaults to "1" (L2): the enabled
        # direction's call site is repair_all_groups, not repair_failed_group.
        monkeypatch.setattr(
            "conpo.forward.repair_all_groups",
            lambda groups, *a, **kw: [
                RepairResult(edges=[_canned_edge("forward")], feasible_endpoints=[], endpoint_sources=[])
                for _ in groups
            ],
        )
    else:
        monkeypatch.setattr(
            "conpo.reverse.reverse_batch",
            lambda *a, **kw: [
                ReverseResult(
                    edge=_canned_edge("reverse"), family=NO_COMMA.iid, reason="admitted", distance=0.1, veto_kept=True
                )
            ],
        )

    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
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
    )
    arm, _ = resolve_arm(arm_name)
    edge_buffer = EdgeBuffer()
    feasible_buffer = FeasibleBuffer()
    # Non-empty regardless of arm_name: this is what would let reverse
    # construction run if arm.reverse gating ever broke.
    feasible_buffer.add(ROW_A.key, "already feasible response")

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=feasible_buffer,
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=Overrides(no_veto=True),
        peft_config=_lora_config(),
        probe_enabled=False,
    )

    trainer.train()

    assert len(edge_buffer) >= 1
    assert all(e.direction == enabled_direction for e in edge_buffer._edges)


# ---------------------------------------------------------------------------
# RUN_VLLM_PUSH_ONCE_PER_STEP (L1, devbox perf pass 2026-09-13): the deferred
# vLLM sleep must be flushed for real before flush_pending_veto's asleep-by-
# the-time-a-burst-starts invariant (conpo/trainer.py's
# _sleep_engine_if_enabled docstring) is asked to hold.
# ---------------------------------------------------------------------------


def test_flush_deferred_vllm_sleep_runs_before_periodic_veto_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_generate_and_score_completions` must call `_flush_deferred_vllm_sleep`
    before it can trigger `flush_pending_veto` -- the periodic check here,
    plus `_save_checkpoint` and scripts/train.py's end-of-training call,
    which both fire strictly after this method has already returned for the
    step. With `max_steps=1`, HF's own end-of-training checkpoint save fires
    `_save_checkpoint`'s own `flush_pending_veto()` call in addition to the
    periodic one this test forces (`veto_period_k=1`), so two
    "flush_pending_veto" markers are expected -- both must still come after
    the single "flush_deferred_sleep" marker. `use_vllm=False` (no real
    vLLM engine, so `_flush_deferred_vllm_sleep` itself is a no-op) is
    enough: this test is only about call ORDER, both methods replaced with
    spies that record it."""
    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=8,
        max_steps=1,
        use_vllm=False,
        beta=config.KL_BETA,
        loss_type="grpo",
        learning_rate=1e-3,
        report_to=[],
        use_cpu=True,
        bf16=False,
        fp16=False,
    )
    arm, overrides = resolve_arm("solve_only")

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=EdgeBuffer(),
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=overrides,
        peft_config=_lora_config(),
    )
    trainer.veto_period_k = 1  # force the periodic branch on this one step

    call_order: list[str] = []

    def _fake_flush_deferred_sleep() -> None:
        call_order.append("flush_deferred_sleep")

    def _fake_flush_pending_veto() -> float:
        call_order.append("flush_pending_veto")
        return 0.0

    monkeypatch.setattr(trainer, "_flush_deferred_vllm_sleep", _fake_flush_deferred_sleep)
    monkeypatch.setattr(trainer, "flush_pending_veto", _fake_flush_pending_veto)

    trainer.train()

    assert call_order[0] == "flush_deferred_sleep"
    assert call_order.count("flush_deferred_sleep") == 1
    assert len(call_order) >= 2
    assert all(item == "flush_pending_veto" for item in call_order[1:])


def test_flush_pending_veto_flushes_deferred_sleep_at_its_own_top(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B5 (iii, audit 2026-09-14): flush_pending_veto must flush any deferred
    vLLM sleep itself, not rely on _generate_and_score_completions's own call
    site -- the queue-overflow trigger (_push_pending, above) can call
    flush_pending_veto mid-step, before that call site is reached, and would
    otherwise run the judge next to a still-awake engine. Calling
    flush_pending_veto directly (queue empty, no train() needed) isolates
    the method's own behaviour from its caller."""
    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=8,
        max_steps=1,
        use_vllm=False,
        beta=config.KL_BETA,
        loss_type="grpo",
        learning_rate=1e-3,
        report_to=[],
        use_cpu=True,
        bf16=False,
        fp16=False,
    )
    arm, overrides = resolve_arm("solve_only")

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=EdgeBuffer(),
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=overrides,
        peft_config=_lora_config(),
    )
    assert len(trainer.pending_queue) == 0  # the no-op early-return path below

    calls: list[str] = []
    monkeypatch.setattr(trainer, "_flush_deferred_vllm_sleep", lambda: calls.append("flush"))

    trainer.flush_pending_veto()

    assert calls == ["flush"]


# ---------------------------------------------------------------------------
# K4 RUN_KL_CLAMP (devbox 2026-09-13): ConPOTrainer._compute_loss is a full
# copy of TRL 1.9.2's own GRPOTrainer._compute_loss (no overridable KL hook
# exists there -- see conpo/trainer.py's comment above the method) with one
# added clamp on the KL diff. Both tests call _compute_loss directly on a
# hand-built GRPO-shaped inputs dict, bypassing generation/edges/judge
# entirely (none of which the KL diff touches).
# ---------------------------------------------------------------------------

_KL_TEST_VOCAB_SIZE = 151936  # matches _tiny_model_and_tokenizer's Qwen3Config


def _kl_clamp_trainer(tmp_path: Path) -> ConPOTrainer:
    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])
    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
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
    )
    arm, overrides = resolve_arm("solve_only")
    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=EdgeBuffer(),
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=overrides,
        peft_config=_lora_config(),
    )
    # _compute_loss reads current_gradient_accumulation_steps, normally set
    # per-batch by transformers.Trainer._inner_training_loop; these tests
    # call _compute_loss directly (no trainer.train()), so set it by hand.
    trainer.current_gradient_accumulation_steps = grpo_config.gradient_accumulation_steps
    return trainer


def _kl_clamp_inputs() -> dict[str, torch.Tensor]:
    """Minimal GRPO-shaped micro-batch _compute_loss needs: 2 prompts x
    3-token completions, no padding. ref_per_token_logps is set far from
    any plausible policy log-prob so the unclamped KL diff is clearly
    nonzero (needed by the unset-path equivalence test to actually
    exercise the KL branch; the clamp=0.0 test doesn't depend on this
    value at all, since the clamp forces the diff to 0 regardless)."""
    torch.manual_seed(0)
    return {
        "prompt_ids": torch.randint(0, _KL_TEST_VOCAB_SIZE, (2, 4)),
        "prompt_mask": torch.ones(2, 4, dtype=torch.long),
        "completion_ids": torch.randint(0, _KL_TEST_VOCAB_SIZE, (2, 3)),
        "completion_mask": torch.ones(2, 3, dtype=torch.long),
        "advantages": torch.tensor([1.0, -0.5]),
        "ref_per_token_logps": torch.full((2, 3), -50.0),
    }


def test_compute_loss_kl_clamp_disabled_matches_trl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_KL_CLAMP="none" (the disable spelling, decision 2026-09-14 --
    unset now defaults to config.KL_CLAMP=40.0, see
    test_kl_clamp_and_loss_type_defaults below) -> self._kl_clamp is None ->
    the clamp branch in ConPOTrainer._compute_loss is a no-op, so its result
    must be exactly what TRL's own (un-subclassed) _compute_loss returns on
    the identical trainer instance, model and inputs."""
    monkeypatch.setenv("RUN_KL_CLAMP", "none")
    trainer = _kl_clamp_trainer(tmp_path)
    assert trainer._kl_clamp is None
    inputs = _kl_clamp_inputs()

    loss_ours = trainer._compute_loss(trainer.model, inputs)
    loss_trl = GRPOTrainer._compute_loss(trainer, trainer.model, inputs)

    assert torch.equal(loss_ours, loss_trl)


def test_compute_loss_kl_clamp_zero_collapses_kl_to_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_KL_CLAMP=0.0 clamps every token's diff to [0, 0], so per_token_kl
    (exp(0) - 0 - 1) is exactly 0 for every masked token no matter how far
    ref_per_token_logps sits from the policy's own -- checked via the "kl"
    metric _compute_loss records (global_masked_mean(per_token_kl) over the
    fully-unmasked batch), which is exactly 0.0 only if every token is."""
    monkeypatch.setenv("RUN_KL_CLAMP", "0.0")
    trainer = _kl_clamp_trainer(tmp_path)
    assert trainer._kl_clamp == 0.0
    inputs = _kl_clamp_inputs()

    trainer._compute_loss(trainer.model, inputs)

    assert trainer._metrics["train"]["kl"][-1] == 0.0


def test_kl_clamp_and_loss_type_defaults(tmp_path: Path) -> None:
    """Loss-aggregation decision (Ethan, 2026-09-14): config.LOSS_TYPE's and
    config.KL_CLAMP's defaults are "dapo" and 40.0, and RUN_KL_CLAMP unset
    reaches the trainer as that same 40.0 (not None -- see
    test_compute_loss_kl_clamp_disabled_matches_trl above for the explicit
    "none" disable spelling)."""
    assert config.LOSS_TYPE == "dapo"
    assert config.KL_CLAMP == 40.0
    trainer = _kl_clamp_trainer(tmp_path)
    assert trainer._kl_clamp == config.KL_CLAMP == 40.0


# ---------------------------------------------------------------------------
# K1 RUN_KL_BETA (devbox 2026-09-15): beta=0.0 must switch the explicit KL
# term off exactly the way TRL 1.9.2 does (self.ref_model = None, every
# KL-reading line in _compute_loss gated on `self.beta != 0.0`), while the
# pair path (L_resp/L_prompt, score_response under `_reference_model`)
# keeps working unchanged for a LoRA policy -- it never reads self.beta or
# self.ref_model, scoring the frozen initial policy by disabling the
# adapter instead. This end-to-end test builds on test_full_arm_two_
# optimizer_steps's real (non-stubbed) "full" arm + hand-built-edges
# fixture, only with beta=0.0 and one optimizer step (beta itself does not
# touch edge construction/admission, so one step is enough to exercise
# compute_loss's pair-path branches).
# ---------------------------------------------------------------------------


def test_full_arm_beta_zero_disables_kl_but_keeps_pair_losses_working(tmp_path: Path) -> None:
    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])

    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
        use_vllm=False,
        beta=0.0,
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
    )
    arm, overrides = resolve_arm("full")  # resp_loss=True, prompt_loss=True
    edge_buffer = EdgeBuffer()
    meter = TokenMeter(budget=0)
    mechanism_log_path = tmp_path / "mechanism.jsonl"
    capture = _CaptureCallback()

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=meter,
        mechanism_log_path=mechanism_log_path,
        seed=0,
        overrides=overrides,
        peft_config=_lora_config(),
        callbacks=[capture],
    )
    capture.trainer = trainer

    # TRL 1.9.2 (trainer/grpo_trainer.py): beta == 0.0 -> no reference model
    # is built at all, regardless of PEFT.
    assert trainer.beta == 0.0
    assert trainer.ref_model is None

    # The pair path scores against the frozen initial policy via
    # `_reference_model` (disable_adapter for a LoRA policy), independent of
    # self.beta/self.ref_model -- fill edges the same way
    # test_full_arm_two_optimizer_steps does.
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    filled_edges = []
    with torch.no_grad(), unwrapped.disable_adapter():
        for edge in _hand_built_edges():
            logp_plus, len_plus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_plus)
            logp_minus, len_minus = score_response(unwrapped, tokenizer, edge.prompt, edge.y_minus)
            filled_edges.append(
                dataclasses.replace(
                    edge,
                    logp_ref_plus=logp_plus.item(),
                    logp_ref_minus=logp_minus.item(),
                    len_plus=len_plus,
                    len_minus=len_minus,
                )
            )
    for edge in filled_edges:
        edge_buffer.push(edge)

    lora_before = {name: p.detach().clone() for name, p in trainer.model.named_parameters() if "lora_" in name}
    assert lora_before  # sanity: the adapter actually has parameters

    trainer.train()

    assert trainer.state.global_step == 1
    assert len(capture.per_step) == 1
    step1 = capture.per_step[0]

    # beta=0.0 -> no KL term or metric at all (ConPOTrainer._compute_loss's
    # `if self.beta != 0.0:` gates, mirroring TRL's own).
    assert "kl" not in step1

    # The pair losses (L_resp/L_prompt) still ran and produced finite
    # numbers -- unaffected by beta.
    assert math.isfinite(step1["resp_loss"])
    assert math.isfinite(step1["prompt_loss"])
    # theta = theta_ref at step 1 (zero-initialised LoRA): Delta_theta(e) = 0
    # for every edge, so L_resp = -logsigmoid(0) = log(2) exactly (same
    # invariant as test_full_arm_two_optimizer_steps, unaffected by beta).
    assert abs(step1["resp_loss"] - math.log(2)) < 0.05

    lora_after = {name: p.detach() for name, p in trainer.model.named_parameters() if "lora_" in name}
    assert any(not torch.equal(lora_before[name], lora_after[name]) for name in lora_before)


# ---------------------------------------------------------------------------
# K7 RUN_PAIR_LOSS_WEIGHT (devbox 2026-09-13): config.PAIR_LOSS_WEIGHT
# multiplies loss_resp and lambda_p * loss_prompt individually before they
# join `total` in compute_loss, so it must scale the pair contribution
# (loss_resp + lambda_p * loss_prompt) without touching L_clean (the GRPO
# term, stats["clean_loss"]). _score_edge_pairs/_prompt_margins are stubbed
# to fixed-shape tensors (same idiom as
# test_compute_loss_evicts_saturated_edge_and_records_resp_evicted above),
# so compute_loss's own edge sampling from edge_buffer -- which consumes
# self.rng and would otherwise draw edges in a different order on the
# second call -- cannot change the numeric result: both directions'
# 2 edges each are always sampled in full (edge_n=48 > 4 edges available)
# and both fixture edges are single-family (qualifies for the prompt draw
# whether or not RUN_MULTI is set).
# ---------------------------------------------------------------------------


def _pair_loss_weight_trainer(
    tmp_path: Path, loss_type: str = "grpo", gradient_accumulation_steps: int = 1
) -> ConPOTrainer:
    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])
    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
        use_vllm=False,
        beta=config.KL_BETA,
        loss_type=loss_type,
        learning_rate=1e-3,
        report_to=[],
        logging_steps=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        chat_template_kwargs=config.POLICY_CHAT_TEMPLATE_KWARGS,
        seed=0,
        use_cpu=True,
        bf16=False,
        fp16=False,
    )
    arm, _ = resolve_arm("full")  # resp_loss=True, prompt_loss=True
    edge_buffer = EdgeBuffer()
    for edge in _hand_built_edges():
        edge_buffer.push(edge)

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=Overrides(),  # lambda_p = config.LAMBDA_P (1.0)
        peft_config=_lora_config(),
        probe_enabled=False,
    )
    trainer._score_edge_pairs = lambda model, edges: (  # type: ignore[method-assign]
        torch.zeros(len(edges)),
        torch.zeros(len(edges), dtype=torch.long),
        {},
    )
    trainer._prompt_margins = lambda model, edges: (  # type: ignore[method-assign]
        torch.full((len(edges),), 0.5),
        torch.full((len(edges),), 0.1),
    )
    trainer.current_gradient_accumulation_steps = grpo_config.gradient_accumulation_steps
    return trainer


def _pair_scale_inputs(loss_type: str) -> dict[str, torch.Tensor]:
    """_kl_clamp_inputs's GRPO-shaped micro-batch, plus a fixed
    num_items_in_batch for loss_type in {dapo, cispo, vespo} (_compute_loss's
    dapo branch requires it; grpo/bnpo/dr_grpo/luspo never read it)."""
    inputs = _kl_clamp_inputs()
    if loss_type not in ("grpo", "sapo", "bnpo", "dr_grpo", "luspo"):
        inputs["num_items_in_batch"] = torch.tensor(24.0)
    return inputs


def test_compute_loss_pair_loss_weight_scales_pair_contribution_not_clean_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _pair_loss_weight_trainer(tmp_path)
    inputs = _kl_clamp_inputs()

    monkeypatch.setattr(config, "PAIR_LOSS_WEIGHT", 1.0)
    total_1x = trainer.compute_loss(trainer.model, inputs)
    stats_1x = dict(trainer.last_loss_stats)

    monkeypatch.setattr(config, "PAIR_LOSS_WEIGHT", 2.0)
    total_2x = trainer.compute_loss(trainer.model, inputs)
    stats_2x = dict(trainer.last_loss_stats)

    # L_clean (the GRPO term) is untouched by PAIR_LOSS_WEIGHT.
    assert stats_1x["clean_loss"] == pytest.approx(stats_2x["clean_loss"])

    pair_1x = total_1x.item() - stats_1x["clean_loss"] * config.LAMBDA_G
    pair_2x = total_2x.item() - stats_2x["clean_loss"] * config.LAMBDA_G
    assert pair_1x != pytest.approx(0.0)
    assert pair_2x == pytest.approx(2 * pair_1x)


# ---------------------------------------------------------------------------
# RUN_PAIR_SCALE_AUTO (fix devbox 2026-09-14): neither Trainer nor TRL's own
# loss_type-specific normalisation ever divides loss_resp/loss_prompt by
# gradient_accumulation_steps (GRPOTrainer unconditionally disables
# Trainer's automatic scaling for every loss_type), so backward()
# accumulates gradient_accumulation_steps copies of that per-micro-batch
# pair-loss mean per optimiser step. "1" (default) compensates by dividing
# both pair terms by self.current_gradient_accumulation_steps.
# ---------------------------------------------------------------------------


def test_resp_pair_scale_stat_matches_grad_accum_and_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """stats["resp_pair_scale"] == 1/gradient_accumulation_steps when the
    flag is on (default), for both loss types, and == 1.0 (today's
    behaviour) when the flag is off."""
    monkeypatch.delenv("RUN_PAIR_SCALE_AUTO", raising=False)
    for loss_type in ("grpo", "dapo"):
        trainer = _pair_loss_weight_trainer(tmp_path, loss_type=loss_type, gradient_accumulation_steps=4)
        trainer.compute_loss(trainer.model, _pair_scale_inputs(loss_type))
        assert trainer.last_loss_stats["resp_pair_scale"] == pytest.approx(0.25)

    monkeypatch.setenv("RUN_PAIR_SCALE_AUTO", "0")
    trainer_off = _pair_loss_weight_trainer(tmp_path, loss_type="grpo", gradient_accumulation_steps=4)
    trainer_off.compute_loss(trainer_off.model, _pair_scale_inputs("grpo"))
    assert trainer_off.last_loss_stats["resp_pair_scale"] == pytest.approx(1.0)


def _pair_scale_grad_trainer(
    tmp_path: Path, loss_type: str, gradient_accumulation_steps: int, resp_loss_enabled: bool
) -> ConPOTrainer:
    """Same shape as _pair_loss_weight_trainer, but _score_edge_pairs is
    stubbed to a small function of one real LoRA parameter (instead of that
    fixture's autograd-DISCONNECTED constant tensors) -- needed to inspect
    .grad directly. A real (unstubbed) forward through the tiny random
    model on _hand_built_edges was tried first and produced NaN deltas
    outside trainer.train()'s own setup path (the existing tests that score
    real edges all go through trainer.train(), never raw compute_loss()
    calls); this stub keeps the pair term genuinely differentiable and
    bounded without depending on that setup."""
    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])
    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
        use_vllm=False,
        beta=config.KL_BETA,
        loss_type=loss_type,
        learning_rate=1e-3,
        report_to=[],
        logging_steps=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        chat_template_kwargs=config.POLICY_CHAT_TEMPLATE_KWARGS,
        seed=0,
        use_cpu=True,
        bf16=False,
        fp16=False,
    )
    arm = ArmSpec(reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=resp_loss_enabled)
    edge_buffer = EdgeBuffer()
    for edge in _hand_built_edges():
        edge_buffer.push(edge)

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=Overrides(),
        peft_config=_lora_config(),
        probe_enabled=False,
    )

    def stub_score_edge_pairs(
        model: torch.nn.Module, edges: list[Edge]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        lora_param = next(p for n, p in model.named_parameters() if "lora_A" in n and p.requires_grad)
        # kappa*tau (config defaults: 5.0*0.1 = 0.5) bounds where the edge
        # counts as saturated; centering near -2.0 keeps every edge's
        # weight at 1 (a real gradient path) while lora_param.sum() (a
        # freshly-initialised LoRA A row, magnitude << 1) can only nudge it
        # slightly, so delta/weights never leave a well-behaved range.
        delta = lora_param.sum().expand(len(edges)) - 2.0
        ages = torch.zeros(len(edges), dtype=torch.long)
        return delta, ages, {}

    trainer._score_edge_pairs = stub_score_edge_pairs  # type: ignore[method-assign]
    trainer.current_gradient_accumulation_steps = gradient_accumulation_steps
    return trainer


def _accumulated_pair_grad(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loss_type: str,
    gradient_accumulation_steps: int,
    pair_scale_auto: str,
) -> torch.Tensor:
    """One simulated accumulation window (gradient_accumulation_steps
    compute_loss + backward() calls, no zero_grad/optimizer.step in
    between -- matching how transformers.Trainer._inner_training_loop
    itself accumulates), run once with the pair term on and once off, on
    two independently-constructed but identically-seeded models. Both see
    an identical L_clean forward and gradient (arm never touches
    super().compute_loss), so subtracting the two accumulated-gradient
    tensors isolates exactly loss_resp's own contribution."""
    monkeypatch.setenv("RUN_PAIR_SCALE_AUTO", pair_scale_auto)
    inputs = _pair_scale_inputs(loss_type)

    def run(resp_loss_enabled: bool) -> torch.Tensor:
        trainer = _pair_scale_grad_trainer(
            tmp_path,
            loss_type=loss_type,
            gradient_accumulation_steps=gradient_accumulation_steps,
            resp_loss_enabled=resp_loss_enabled,
        )
        for _ in range(gradient_accumulation_steps):
            total = trainer.compute_loss(trainer.model, inputs)
            total.backward()
        name, param = next((n, p) for n, p in trainer.model.named_parameters() if "lora_A" in n and p.requires_grad)
        assert param.grad is not None, name
        return param.grad.clone()

    return run(True) - run(False)


def test_pair_scale_auto_default_matches_across_grad_accum_and_loss_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_PAIR_SCALE_AUTO=1 (default): the pair term's accumulated gradient
    contribution over one optimiser step is independent of
    gradient_accumulation_steps (scale-invariance) and of loss_type (grpo
    vs dapo), within floating-point tolerance -- the goal's two required
    properties."""
    baseline = _accumulated_pair_grad(tmp_path, monkeypatch, "grpo", 1, "1")
    grpo_accum4 = _accumulated_pair_grad(tmp_path, monkeypatch, "grpo", 4, "1")
    dapo_accum1 = _accumulated_pair_grad(tmp_path, monkeypatch, "dapo", 1, "1")
    dapo_accum4 = _accumulated_pair_grad(tmp_path, monkeypatch, "dapo", 4, "1")

    assert baseline.abs().sum().item() > 0.0  # the pair term does contribute a real gradient

    for other in (grpo_accum4, dapo_accum1, dapo_accum4):
        assert torch.allclose(other, baseline, rtol=1e-3, atol=1e-6)


def test_pair_scale_auto_off_reproduces_old_grad_accum_scaled_behaviour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_PAIR_SCALE_AUTO=0: today's byte-for-byte behaviour -- the pair
    term's accumulated gradient over a gradient_accumulation_steps=4 window
    is gradient_accumulation_steps times the gradient_accumulation_steps=1
    case (nothing compensates Trainer's disabled automatic scaling), for
    both loss types."""
    for loss_type in ("grpo", "dapo"):
        accum1 = _accumulated_pair_grad(tmp_path, monkeypatch, loss_type, 1, "0")
        accum4 = _accumulated_pair_grad(tmp_path, monkeypatch, loss_type, 4, "0")
        assert torch.allclose(accum4, 4 * accum1, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# K6 RUN_TOKEN_LOGRATIO_CLAMP (devbox 2026-09-13): _score_edge_pairs's flag
# branch (_score_edge_pairs_clamped/_token_clamped_q) scores both sides
# fresh, per token, and clamps the policy-minus-reference log-ratio before
# forming Delta_theta(e). Both tests run a real trainer.train() over real
# _hand_built_edges (no stubbing of _score_edge_pairs), so the flag actually
# exercises the new forward passes end to end.
# ---------------------------------------------------------------------------


def _token_logratio_clamp_trainer(tmp_path: Path) -> ConPOTrainer:
    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])
    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
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
    )
    # forward/reverse/prompt_loss off, resp_loss on: isolate _score_edge_pairs
    # from generation/repair/corruption/L_prompt, same idiom as the
    # saturated-edge eviction trainer above.
    arm = ArmSpec(reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=True)
    edge_buffer = EdgeBuffer()
    for edge in _hand_built_edges():
        edge_buffer.push(edge)

    trainer = ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=edge_buffer,
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=Overrides(),
        peft_config=_lora_config(),
        probe_enabled=False,
    )
    return trainer


def test_score_edge_pairs_token_clamp_unset_key_absent(tmp_path: Path) -> None:
    """RUN_TOKEN_LOGRATIO_CLAMP unset -> self._token_logratio_clamp is None ->
    _score_edge_pairs takes its original cached-scalar path, and
    resp_token_clamp_frac is ABSENT (not zero) from stats, proving
    byte-for-byte unchanged behaviour when the flag is unset."""
    trainer = _token_logratio_clamp_trainer(tmp_path)
    assert trainer._token_logratio_clamp is None

    trainer.train()

    assert "resp_token_clamp_frac" not in trainer.last_loss_stats


def test_score_edge_pairs_token_clamp_reports_frac_in_unit_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_TOKEN_LOGRATIO_CLAMP set -> _score_edge_pairs_clamped runs (fresh
    per-token policy/reference forwards, clamped diff), and stats carries
    resp_token_clamp_frac in [0, 1]."""
    monkeypatch.setenv("RUN_TOKEN_LOGRATIO_CLAMP", "0.5")
    trainer = _token_logratio_clamp_trainer(tmp_path)
    assert trainer._token_logratio_clamp == 0.5

    trainer.train()

    frac = trainer.last_loss_stats["resp_token_clamp_frac"]
    assert 0.0 <= frac <= 1.0


# ---------------------------------------------------------------------------
# K8 RUN_PAIR_OBJECTIVE / RUN_POS_ANCHOR_WEIGHT (devbox 2026-09-14):
# _score_edge_pairs's span_local branch (_score_edge_pairs_span_local /
# _response_logratios) replaces Delta_theta(e) with a margin over only the
# tokens where y_plus and y_minus differ (conpo/losses.py's diff_span),
# instead of the whole-sequence sum every other path uses. Reuses
# _token_logratio_clamp_trainer (same hand-built-edges / resp_loss-only
# isolation idiom above) since span_local is just another _score_edge_pairs
# branch selected by env, at construction time.
# ---------------------------------------------------------------------------


def test_pair_objective_invalid_value_raises_at_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_PAIR_OBJECTIVE", "bogus")
    with pytest.raises(ValueError, match="RUN_PAIR_OBJECTIVE"):
        _token_logratio_clamp_trainer(tmp_path)


def test_pair_objective_unset_defaults_to_sequence_and_is_byte_identical(tmp_path: Path) -> None:
    """RUN_PAIR_OBJECTIVE unset -> self._pair_objective == "sequence" ->
    _score_edge_pairs takes the exact pre-span_local branch. Forcing the
    same trainer's _pair_objective to the explicit string "sequence" and
    re-scoring the identical edges against the identical (untouched, since
    neither call takes a training step) model weights gives an identical
    delta, proving "unset" and "sequence" are indistinguishable to the
    dispatcher -- i.e. this change left the default path byte for byte.
    logp_ref_plus/minus are NaN on these hand-built edges (by design, see
    _hand_built_edges), so the comparison needs equal_nan.
    """
    trainer = _token_logratio_clamp_trainer(tmp_path)
    assert trainer._pair_objective == "sequence"
    edges = _hand_built_edges()

    with torch.no_grad():
        delta_unset, _, stats_unset = trainer._score_edge_pairs(trainer.model, edges)
        trainer._pair_objective = "sequence"
        delta_explicit, _, stats_explicit = trainer._score_edge_pairs(trainer.model, edges)

    assert torch.allclose(delta_unset, delta_explicit, equal_nan=True)
    assert stats_unset == stats_explicit == {}


def test_pair_objective_span_local_reports_new_stats_in_sane_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_PAIR_OBJECTIVE=span_local -> _score_edge_pairs_span_local runs
    (fresh per-token policy/reference forwards, diff_span-sliced margin),
    and every new stat lands in trainer.last_loss_stats via the SAME
    generic resp_-prefixing compute_loss already applies to
    token_clamp_stats/resp_stats -- no new plumbing needed for this."""
    monkeypatch.delenv("RUN_POS_ANCHOR_WEIGHT", raising=False)
    monkeypatch.setenv("RUN_PAIR_OBJECTIVE", "span_local")
    trainer = _token_logratio_clamp_trainer(tmp_path)
    assert trainer._pair_objective == "span_local"
    assert trainer._pos_anchor_weight == 0.0

    trainer.train()
    stats = trainer.last_loss_stats

    assert stats["resp_span_len_plus_mean"] >= 0.0
    assert stats["resp_span_len_minus_mean"] >= 0.0
    assert 0.0 <= stats["resp_empty_span_frac"] <= 1.0
    assert 0.0 <= stats["resp_token_clamp_frac"] <= 1.0
    assert stats["resp_pos_anchor"] >= 0.0  # mean of max(0, -r_t): never negative


def test_pair_objective_span_local_combined_with_token_logratio_clamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """span_local reuses RUN_TOKEN_LOGRATIO_CLAMP as its own span-token
    clamp (not a second, separate knob): setting both together still
    reports a valid fraction."""
    monkeypatch.setenv("RUN_PAIR_OBJECTIVE", "span_local")
    monkeypatch.setenv("RUN_TOKEN_LOGRATIO_CLAMP", "0.1")
    trainer = _token_logratio_clamp_trainer(tmp_path)
    assert trainer._pair_objective == "span_local"
    assert trainer._token_logratio_clamp == 0.1

    trainer.train()

    frac = trainer.last_loss_stats["resp_token_clamp_frac"]
    assert 0.0 <= frac <= 1.0


def test_pos_anchor_weight_env_override_reaches_trainer_attribute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RUN_POS_ANCHOR_WEIGHT", "0.3")
    trainer = _token_logratio_clamp_trainer(tmp_path)
    assert trainer._pos_anchor_weight == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# K5 RUN_RESAMPLE_ZERO_STD (devbox 2026-09-13): a degenerate (zero-reward-std)
# prompt group is replaced with a freshly drawn, freshly generated substitute
# from rows_by_key, up to RUN_RESAMPLE_MAX_ROUNDS extra rounds. Both tests
# rig reward_conjunctive at the class level (must happen before construction:
# ConPOTrainer.__init__ binds self.reward_conjunctive into GRPOConfig's
# reward_funcs once, so a post-construction instance patch would not be
# seen) and run a real trainer.train() over the tiny model, forward/reverse/
# prompt_loss/resp_loss all off to isolate _resample_degenerate_groups from
# every other mechanism.
# ---------------------------------------------------------------------------


def _resample_trainer(tmp_path: Path, rows_by_key: dict[str, PoolRow], loss_type: str = "grpo") -> ConPOTrainer:
    model, tokenizer = _tiny_model_and_tokenizer()
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": ROW_A.prompt}], "key": ROW_A.key}])
    grpo_config = GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=16,
        max_steps=1,
        use_vllm=False,
        beta=config.KL_BETA,
        loss_type=loss_type,
        learning_rate=1e-3,
        report_to=[],
        logging_steps=1,
        gradient_accumulation_steps=1,
        chat_template_kwargs=config.POLICY_CHAT_TEMPLATE_KWARGS,
        seed=0,
        use_cpu=True,
        bf16=False,
        fp16=False,
    )
    arm = ArmSpec(reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=False)
    return ConPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=EdgeBuffer(),
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=Overrides(),
        peft_config=_lora_config(),
        probe_enabled=False,
    )


def _mechanism_records(mechanism_log_path: Path) -> list[dict[str, object]]:
    lines = mechanism_log_path.read_text(encoding="utf-8").splitlines()
    return [r for r in (json.loads(line) for line in lines if line.strip()) if "event" not in r]


@pytest.mark.parametrize("loss_type", ["grpo", "dapo"])
def test_resample_zero_std_replaces_degenerate_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loss_type: str
) -> None:
    """The dataset holds only ROW_A, whose rigged reward is constant (zero
    std, degenerate); rows_by_key also carries ROW_B, whose rigged reward
    alternates within any group (nonzero std). RUN_RESAMPLE_ZERO_STD must
    draw ROW_B as ROW_A's only candidate substitute, splice its freshly
    generated group into `output`, and report the swap -- with the
    degenerate_group_rate the existing per-group loop computes afterward
    improving to 0.0. Parametrised over loss_type (B6, audit 2026-09-14):
    before this, the fixture only ever built loss_type="grpo" trainers, so
    the real _splice_group_output -> dapo-normalizer path (4dcda75's K5 x
    dapo num_items_in_batch fix) never ran end to end -- a crash in that
    path (e.g. a missing/mis-shaped num_items_in_batch reaching
    _compute_loss's dapo branch) would have gone undetected here.
    test_dapo_loss_changes_with_splice_length below asserts the actual
    loss-value effect deterministically; real generation lengths are not
    controllable enough to assert that precisely against this rigged,
    trainer.train()-driven path without risking flakiness."""

    def rigged_reward(self: ConPOTrainer, completions: object, key: list[str], **_: object) -> list[float]:
        return [1.0 if k == ROW_A.key else float(i % 2) for i, k in enumerate(key)]

    monkeypatch.setattr(ConPOTrainer, "reward_conjunctive", rigged_reward)
    monkeypatch.setenv("RUN_RESAMPLE_ZERO_STD", "1")

    trainer = _resample_trainer(tmp_path, {ROW_A.key: ROW_A, ROW_B.key: ROW_B}, loss_type=loss_type)
    assert trainer._resample_zero_std is True

    trainer.train()  # must complete without error under dapo too (exercises the real splice + dapo loss path)

    records = _mechanism_records(trainer.mechanism_log_path)
    assert len(records) == 1
    assert records[0]["resample_groups_replaced"] >= 1
    assert records[0]["degenerate_group_rate"] == 0.0


def test_resample_zero_std_stops_at_round_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every candidate row (including the original) scores a constant
    reward under this rig, so no substitute is ever informative:
    _resample_degenerate_groups must run exactly RUN_RESAMPLE_MAX_ROUNDS
    rounds, then let training proceed with the still-degenerate batch
    rather than hang or raise."""

    def always_degenerate_reward(self: ConPOTrainer, completions: object, key: list[str], **_: object) -> list[float]:
        return [1.0 for _ in completions]

    monkeypatch.setattr(ConPOTrainer, "reward_conjunctive", always_degenerate_reward)
    monkeypatch.setenv("RUN_RESAMPLE_ZERO_STD", "1")
    monkeypatch.setenv("RUN_RESAMPLE_MAX_ROUNDS", "2")

    rows_by_key = {r.key: r for r in (ROW_A, ROW_B, ROW_C, ROW_D)}
    trainer = _resample_trainer(tmp_path, rows_by_key)
    assert trainer._resample_max_rounds == 2

    trainer.train()  # must return, not hang or raise, once the round cap is hit

    assert trainer.state.global_step == 1
    records = _mechanism_records(trainer.mechanism_log_path)
    assert len(records) == 1
    assert records[0]["resample_rounds"] == 2
    assert records[0]["degenerate_group_rate"] == 1.0


def test_splice_group_output_adjusts_num_items_in_batch_by_token_delta(tmp_path: Path) -> None:
    """_splice_group_output's docstring: `num_items_in_batch` is TRL's own
    pre-resampling batch-wide valid-token count (grpo_trainer.py's
    `_generate_and_score_completions`, ~line 2427), which
    ConPOTrainer._compute_loss uses as the loss normalizer whenever
    loss_type is in {dapo, cispo, vespo}. A spliced-in replacement group of
    a different completion length must move that scalar by the resulting
    change in valid completion tokens, or RUN_LOSS_TYPE=dapo combined with
    RUN_RESAMPLE_ZERO_STD=1 trains every resampled step with a stale
    normalizer (fixer review 2026-09-13)."""
    trainer = _resample_trainer(tmp_path, {ROW_A.key: ROW_A})
    output = {
        "completion_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),  # 2 + 3 = 5 valid tokens
        "advantages": torch.tensor([0.5, -0.5]),
        "num_items_in_batch": torch.tensor(10.0),
    }
    sub_output = {
        "completion_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),  # 4 + 2 = 6 valid tokens
        "advantages": torch.tensor([1.0, -1.0]),
        "num_items_in_batch": torch.tensor(999.0),  # must be ignored: only output's own copy is updated
    }

    trainer._splice_group_output(output, sub_output, start=0, end=2)

    assert output["num_items_in_batch"].item() == pytest.approx(11.0)  # 10 + (6 - 5)


def test_dapo_loss_changes_with_splice_length(tmp_path: Path) -> None:
    """B6 (audit 2026-09-14): the K5 x dapo fix (4dcda75) is only worth
    testing if the num_items_in_batch delta _splice_group_output computes
    (proved above) reaches the loss under loss_type=dapo, where it
    is the normalizer (_compute_loss, trainer.py ~2536). Driving this
    through a real trainer.train() (parametrized loss_type above) cannot
    assert a precise loss-value effect without flakiness: the completion
    length of a freshly generated replacement group, from an untrained tiny
    model, is not controllable. So this isolates the mechanism
    deterministically instead: row 0's advantage is 0 both times, and dapo's
    per_token_loss = -min(coef_1, coef_2) * advantage is exactly 0 at every
    position regardless of mask (advantage broadcasts over T), so row 0
    contributes nothing to the loss numerator no matter how long its
    spliced-in replacement is; row 1 is untouched by the splice (start=0,
    end=1) and old_per_token_logps defaults to per_token_logps.detach(), so
    coef_1 is exactly 1.0 for it regardless of the model's actual output,
    making its contribution a fixed -1.0 per valid token too. With beta=0
    (no KL term), the numerator is therefore bit-identical between the two
    splices below, and only the normalizer (num_items_in_batch, moved by
    the differing replacement lengths) differs -- so the loss ratio must
    equal the inverse ratio of the two num_items_in_batch values exactly. A
    reverted fix (output["num_items_in_batch"] left untouched by the
    splice) would make num_items_in_batch, and so the loss, identical
    between the two calls."""
    trainer = _resample_trainer(tmp_path, {ROW_A.key: ROW_A}, loss_type="dapo")
    trainer.current_gradient_accumulation_steps = 1  # normally set per-batch by Trainer._inner_training_loop
    trainer.beta = 0.0  # isolate the normalizer effect from the KL term

    torch.manual_seed(0)
    base_inputs = {
        "prompt_ids": torch.randint(0, _KL_TEST_VOCAB_SIZE, (2, 4)),
        "prompt_mask": torch.ones(2, 4, dtype=torch.long),
        "completion_ids": torch.randint(0, _KL_TEST_VOCAB_SIZE, (2, 5)),
        "completion_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),  # row 0: 3 valid (discarded_valid)
        "advantages": torch.tensor([0.0, 1.0]),
        "ref_per_token_logps": torch.full((2, 5), -50.0),
        "num_items_in_batch": torch.tensor(20.0),
    }

    def spliced_loss(replacement_valid_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        output = {k: v.clone() for k, v in base_inputs.items()}
        mask_row = [1] * replacement_valid_tokens + [0] * (5 - replacement_valid_tokens)
        sub_output = {
            "completion_mask": torch.tensor([mask_row]),
            "advantages": torch.tensor([0.0]),
            "num_items_in_batch": torch.tensor(-1.0),  # must be ignored: only output's own copy is updated
        }
        trainer._splice_group_output(output, sub_output, start=0, end=1)
        return trainer._compute_loss(trainer.model, output), output["num_items_in_batch"]

    loss_short, num_items_short = spliced_loss(1)  # 20 + (1 - 3) = 18
    loss_long, num_items_long = spliced_loss(4)  # 20 + (4 - 3) = 21

    assert num_items_short.item() == pytest.approx(18.0)
    assert num_items_long.item() == pytest.approx(21.0)
    assert loss_short.item() != pytest.approx(loss_long.item())
    assert (loss_short / loss_long).item() == pytest.approx(num_items_long.item() / num_items_short.item())


# ---------------------------------------------------------------------------
# arms.py sanity (folded in here per the file-scope assigned to this task)
# ---------------------------------------------------------------------------


def test_resolve_arm_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        resolve_arm("not-a-real-arm")

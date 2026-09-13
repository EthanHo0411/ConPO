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
from trl import GRPOConfig

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
from conpo.trainer import ConPOTrainer, score_response
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

    def fake_score_edge_pairs(model: torch.nn.Module, edges: list) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(edges) == 2
        return torch.tensor([margin_high, margin_low]), torch.zeros(len(edges), dtype=torch.long)

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


def test_forward_feasible_endpoints_reach_feasible_buffer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    disabled_target = "conpo.reverse.reverse_batch" if arm_name == "forward_only" else "conpo.forward.repair_failed_group"

    def _raise_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"{disabled_target} must not run when its direction is disabled")

    monkeypatch.setattr(disabled_target, _raise_if_called)

    if arm_name == "forward_only":
        monkeypatch.setattr(
            "conpo.forward.repair_failed_group",
            lambda *a, **kw: RepairResult(edges=[_canned_edge("forward")], feasible_endpoints=[], endpoint_sources=[]),
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
# arms.py sanity (folded in here per the file-scope assigned to this task)
# ---------------------------------------------------------------------------


def test_resolve_arm_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        resolve_arm("not-a-real-arm")

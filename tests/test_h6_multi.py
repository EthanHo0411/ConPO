"""Tests for H6 part B: the multi-constraint compositor wired into the arm
switch (RUN_MULTI), reverse construction (conpo/reverse.py), and the trainer
(conpo/trainer.py). Part A (conpo/compositor.py's PairStats/sample_family_set,
conpo/operators/registry.py's exact_set/corrupt_set, conpo/prompts.py's
counterfactual_prompts_set) has its own coverage in tests/test_compositor.py,
tests/test_operators_multi.py and tests/test_prompts.py; this file only
exercises what part B adds on top: Overrides.multi/resolve_arm, reverse_edge's
target_indices path and reverse_batch's pair_stats draw, and the tiny
trainer's end-to-end wiring (mechanism-log fields, pair_stats.json
save/load, and _prompt_margins on a multi-family edge).

conpo.arms.HYPOTHESES carries an "H6_multi" entry (design 2026-09-03: H6 is
a built arm -- see arms.py's own comment on that entry); scripts/dashboard.py
and its tests (a separate workflow's files) read that entry directly instead
of synthesizing an H6 row, and that is locked by
tests/test_dashboard.py:test_h6_row_uses_the_real_hypotheses_entry, out of
scope for this file's own coverage.
"""

from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl import GRPOConfig

import conpo.operators  # noqa: F401  registers every corruption operator
from conpo import config
from conpo.arms import HYPOTHESES, Overrides, resolve_arm
from conpo.compositor import PairStats
from conpo.data import PoolRow, TokenMeter
from conpo.edges import Edge, EdgeBuffer, FeasibleBuffer
from conpo.engine import HFEngine
from conpo.judge import HFBatchedJudge, StaticJudge, build_judge
from conpo.operators import registry
from conpo.reverse import reverse_batch, reverse_edge
from conpo.selection import FamilyStats
from conpo.trainer import ConPOTrainer, score_response
from conpo.verifiers import Constraint, feasible, verifier_vector

NO_COMMA = Constraint("punctuation:no_comma", {})
WORD_ONCE = Constraint("keywords:word_once", {"keyword": "unique"})
MULTI_TEXT = (
    "the report highlights one unique finding from this quarter without any repeated commas in this response"
)


def _multi_row(key: str) -> PoolRow:
    return PoolRow(
        key=key,
        base_prompt="Write a short passage about the pipeline.",
        prompt="Write a short passage about the pipeline.",
        sentences=(),
        constraints=(NO_COMMA, WORD_ONCE),
        english=True,
        num_constraints=2,
    )


def _qualifying_pair_stats() -> PairStats:
    """A PairStats where (NO_COMMA, WORD_ONCE) clears config's default
    PHI_MIN/MIN_COUNT (phi = 1.0, count = 30)."""
    stats = PairStats()
    strong = ((False, False),) * 15 + ((True, True),) * 15
    stats.update((NO_COMMA, WORD_ONCE), strong)
    return stats


# ---------------------------------------------------------------------------
# arms.py: Overrides.multi / resolve_arm(RUN_MULTI)
# ---------------------------------------------------------------------------


def test_resolve_arm_run_multi_1_sets_overrides_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_MULTI", "1")
    _, overrides = resolve_arm("full")
    assert overrides.multi is True


def test_resolve_arm_without_run_multi_leaves_overrides_multi_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_MULTI", raising=False)
    _, overrides = resolve_arm("full")
    assert overrides.multi is False


def test_overrides_multi_defaults_false() -> None:
    assert Overrides().multi is False


def test_hypotheses_has_an_h6_entry() -> None:
    """See this file's module docstring: H6 is a built arm (design
    2026-09-03), so HYPOTHESES now carries an "H6_multi" entry rather than
    leaving dashboard.py to synthesize the row."""
    assert any(h.id.startswith("H6") for h in HYPOTHESES)


# ---------------------------------------------------------------------------
# reverse.py: reverse_edge(target_indices=...)
# ---------------------------------------------------------------------------


def test_reverse_edge_admits_a_multi_family_edge() -> None:
    row = _multi_row("row-multi")
    result = reverse_edge(
        row, MULTI_TEXT, FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=5,
        target_indices=(0, 1),
    )

    assert result.reason == "admitted"
    assert result.edge is not None
    edge = result.edge
    assert edge.families == frozenset({NO_COMMA.iid, WORD_ONCE.iid})
    assert edge.direction == "reverse"
    assert edge.created_update == 5
    assert result.family is not None and "," in result.family  # multi-family label

    verdicts = verifier_vector(row.constraints, edge.y_minus)
    assert verdicts == (False, False)  # both targets fail, exactly (exact_set's own contract)


def test_reverse_edge_single_family_behaviour_unchanged_by_the_target_indices_default() -> None:
    """target_indices=None (the default) is exactly the pre-H6-part-B path:
    one family drawn from eligible_families(row)."""
    row = _multi_row("row-single")
    result = reverse_edge(row, MULTI_TEXT, FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=0)

    assert result.reason == "admitted"
    assert result.edge is not None
    assert len(result.edge.families) == 1
    assert result.family in {NO_COMMA.iid, WORD_ONCE.iid}
    assert "," not in result.family


def test_reverse_edge_multi_target_raises_on_empty_target_indices() -> None:
    row = _multi_row("row-empty")
    with pytest.raises(ValueError, match="target_indices must be non-empty"):
        reverse_edge(row, MULTI_TEXT, FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=0, target_indices=())


def test_reverse_edge_multi_target_no_operator_family_reports_no_multi_candidate() -> None:
    row = PoolRow(
        key="row-lang-multi",
        base_prompt="Write a short passage.",
        prompt="Write a short passage.",
        sentences=(),
        constraints=(Constraint("language:response_language", {"language": "en"}), NO_COMMA),
        english=True,
        num_constraints=2,
    )
    result = reverse_edge(
        row, "some response", FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=0,
        target_indices=(0, 1),
    )
    assert result.edge is None
    assert result.reason == "no_multi_candidate"


# ---------------------------------------------------------------------------
# reverse.py: reverse_batch(pair_stats=...)
# ---------------------------------------------------------------------------


def test_reverse_batch_admits_a_multi_family_edge_when_pair_stats_qualifies() -> None:
    row = _multi_row("row-multi-batch")
    buffer = FeasibleBuffer()
    buffer.add(row.key, MULTI_TEXT)
    pair_stats = _qualifying_pair_stats()

    results = reverse_batch(
        {row.key: row}, buffer, 50, FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=0,
        pair_stats=pair_stats,
    )

    multi_edges = [r for r in results if r.edge is not None and len(r.edge.families) > 1]
    assert multi_edges


@pytest.mark.parametrize("pair_stats", [None, PairStats()])
def test_reverse_batch_admits_no_multi_family_edge_without_a_qualifying_pair_stats(
    pair_stats: PairStats | None,
) -> None:
    row = _multi_row("row-multi-batch-empty")
    buffer = FeasibleBuffer()
    buffer.add(row.key, MULTI_TEXT)

    results = reverse_batch(
        {row.key: row}, buffer, 50, FamilyStats(), StaticJudge(keep=True), random.Random(0), current_update=0,
        pair_stats=pair_stats,
    )

    assert all(r.edge is None or len(r.edge.families) == 1 for r in results)


# ---------------------------------------------------------------------------
# trainer.py: _prompt_margins on a multi-family edge (theta = theta_ref)
# ---------------------------------------------------------------------------


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


def _make_multi_row(key: str, base: str) -> PoolRow:
    sentence_no_comma = "Do not use any commas."
    sentence_word_once = "Use the word 'unique' exactly once."
    prompt = f"{base} {sentence_no_comma} {sentence_word_once}"
    return PoolRow(
        key=key,
        base_prompt=base,
        prompt=prompt,
        sentences=(sentence_no_comma, sentence_word_once),
        constraints=(NO_COMMA, WORD_ONCE),
        english=True,
        num_constraints=2,
    )


def _engine_factory(trainer: ConPOTrainer) -> HFEngine:
    return HFEngine(trainer.accelerator.unwrap_model(trainer.model), trainer.processing_class, device="cpu")


def _grpo_config(tmp_path: Path, max_steps: int) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        num_generations=2,
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
        save_strategy="no",
    )


def test_prompt_margins_multi_family_edge_zero_at_theta_equals_theta_ref(tmp_path: Path) -> None:
    row = _make_multi_row("row-prompt-margin", "Write a short summary.")
    corrupted = registry.corrupt_set(row.constraints, (0, 1), MULTI_TEXT, random.Random(0))
    assert corrupted is not None
    assert feasible(row.constraints, MULTI_TEXT)

    model, tokenizer = _tiny_model_and_tokenizer()
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": row.prompt}], "key": row.key}])
    arm, overrides = resolve_arm("solve_only")  # no construction; _prompt_margins is called directly
    trainer = ConPOTrainer(
        model=model,
        args=_grpo_config(tmp_path, max_steps=1),
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=arm,
        rows_by_key={row.key: row},
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

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    with torch.no_grad(), unwrapped.disable_adapter():
        logp_plus, len_plus = score_response(unwrapped, tokenizer, row.prompt, MULTI_TEXT)
        logp_minus, len_minus = score_response(unwrapped, tokenizer, row.prompt, corrupted)
    edge = Edge(
        prompt_id=row.key,
        prompt=row.prompt,
        y_plus=MULTI_TEXT,
        y_minus=corrupted,
        families=frozenset({NO_COMMA.iid, WORD_ONCE.iid}),
        direction="reverse",
        step=0,
        logp_ref_plus=logp_plus.item(),
        logp_ref_minus=logp_minus.item(),
        len_plus=len_plus,
        len_minus=len_minus,
        created_update=0,
        meta={},
    )

    m_plus, m_minus = trainer._prompt_margins(trainer.model, [edge])

    assert m_plus.shape == (1,)
    assert m_minus.shape == (1,)
    assert abs(m_plus.item()) < 1e-4
    assert abs(m_minus.item()) < 1e-4


def _build_scored_multi_family_edge(tmp_path: Path) -> tuple[PoolRow, "ConPOTrainer", Edge]:
    """A trainer plus one multi-family edge with logp_ref_plus/minus already
    filled in (score_response under disable_adapter, same convention as
    tests/test_trainer_tiny.py's _hand_built_edges seeding). forward/reverse
    are both off (dataclasses.replace) so train() cannot construct further
    edges -- edge_buffer holds exactly this one multi-family edge."""
    row = _make_multi_row("row-gate", "Write a short summary.")
    corrupted = registry.corrupt_set(row.constraints, (0, 1), MULTI_TEXT, random.Random(0))
    assert corrupted is not None

    model, tokenizer = _tiny_model_and_tokenizer()
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": row.prompt}], "key": row.key}])
    arm, overrides = resolve_arm("full")
    no_construction_arm = dataclasses.replace(arm, forward=False, reverse=False)
    edge_buffer = EdgeBuffer()
    trainer = ConPOTrainer(
        model=model,
        args=_grpo_config(tmp_path, max_steps=1),
        train_dataset=dataset,
        processing_class=tokenizer,
        arm=no_construction_arm,
        rows_by_key={row.key: row},
        edge_buffer=edge_buffer,
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

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    with torch.no_grad(), unwrapped.disable_adapter():
        logp_plus, len_plus = score_response(unwrapped, tokenizer, row.prompt, MULTI_TEXT)
        logp_minus, len_minus = score_response(unwrapped, tokenizer, row.prompt, corrupted)
    edge = Edge(
        prompt_id=row.key,
        prompt=row.prompt,
        y_plus=MULTI_TEXT,
        y_minus=corrupted,
        families=frozenset({NO_COMMA.iid, WORD_ONCE.iid}),
        direction="reverse",
        step=0,
        logp_ref_plus=logp_plus.item(),
        logp_ref_minus=logp_minus.item(),
        len_plus=len_plus,
        len_minus=len_minus,
        created_update=0,
        meta={},
    )
    edge_buffer.push(edge)
    return row, trainer, edge


def test_compute_loss_excludes_multi_family_edge_from_prompt_loss_without_run_multi(tmp_path: Path) -> None:
    """RUN_MULTI unset (overrides.multi=False): compute_loss's L_prompt
    sampling predicate stays `len(e.families) == 1`, so a multi-family edge
    in edge_buffer is skipped -- this is the H1/H3/H4/H5 arms' unchanged
    behaviour, regardless of a forward-constructed multi-family edge landing
    in edge_buffer independent of RUN_MULTI."""
    _, trainer, _ = _build_scored_multi_family_edge(tmp_path)
    assert trainer.overrides.multi is False

    trainer.train()

    assert "prompt_loss" not in trainer.last_loss_stats


def test_compute_loss_includes_multi_family_edge_in_prompt_loss_with_run_multi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RUN_MULTI=1 (overrides.multi=True): the same multi-family edge now
    joins L_prompt, per DESIGN item 4."""
    monkeypatch.setenv("RUN_MULTI", "1")
    _, trainer, _ = _build_scored_multi_family_edge(tmp_path)
    assert trainer.overrides.multi is True

    trainer.train()

    assert "prompt_loss" in trainer.last_loss_stats


# ---------------------------------------------------------------------------
# trainer.py end to end: RUN_MULTI=1, mechanism-log fields, pair_stats.json
# save/load across two constructions.
# ---------------------------------------------------------------------------

NO_COMMA_ROW = Constraint("punctuation:no_comma", {})
LOWERCASE_ROW = Constraint("change_case:english_lowercase", {})


def _make_row(key: str, base: str) -> PoolRow:
    sentence_no_comma = "Do not use any commas."
    sentence_lowercase = "Write your entire answer in lowercase letters."
    prompt = f"{base} {sentence_no_comma} {sentence_lowercase}"
    return PoolRow(
        key=key,
        base_prompt=base,
        prompt=prompt,
        sentences=(sentence_no_comma, sentence_lowercase),
        constraints=(NO_COMMA_ROW, LOWERCASE_ROW),
        english=True,
        num_constraints=2,
    )


ROW_A = _make_row("h6-row-a", "Explain how gravity works.")
ROW_B = _make_row("h6-row-b", "Describe how photosynthesis works.")


@pytest.fixture(scope="module")
def tiny_judge_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Same shape as tests/test_batched_veto.py's tiny_model_dir fixture: a
    tiny random Qwen3 checkpoint saved to disk, loadable by path like any
    real judge model name (RUN_JUDGE_MODEL)."""
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
    model = Qwen3ForCausalLM(cfg)
    out_dir = tmp_path_factory.mktemp("h6_tiny_judge_model")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def test_tiny_trainer_multi_writes_mechanism_fields_and_pair_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_judge_model_dir: Path
) -> None:
    monkeypatch.setattr(config, "REVERSE_DRAWS_PER_STEP", 2)
    monkeypatch.setenv("RUN_MULTI", "1")
    monkeypatch.setenv("RUN_USE_VLLM", "0")
    monkeypatch.setenv("RUN_JUDGE_MODE", "batched")
    monkeypatch.setenv("RUN_JUDGE_MODEL", str(tiny_judge_model_dir))
    monkeypatch.delenv("RUN_NO_VETO", raising=False)

    arm, overrides = resolve_arm("full")
    assert overrides.multi is True
    judge = build_judge(device="cpu")
    assert isinstance(judge, HFBatchedJudge)

    model, tokenizer = _tiny_model_and_tokenizer()
    rows = [ROW_A, ROW_B]
    rows_by_key = {r.key: r for r in rows}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": r.prompt}], "key": r.key} for r in rows])
    out_dir = tmp_path / "out"

    trainer = ConPOTrainer(
        model=model,
        args=_grpo_config(tmp_path, max_steps=2),
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
        veto_period_k=1,
        out_dir=out_dir,
        peft_config=_lora_config(),
    )
    assert trainer.pair_stats is not None

    trainer.train()
    trainer.flush_pending_veto()
    trainer.save_pair_stats()

    assert trainer.state.global_step == 2

    lines = [
        json.loads(line) for line in (tmp_path / "mechanism.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    step_records = [r for r in lines if "event" not in r]  # per-step records only: veto_flush, step_timing and resume events sit beside them
    assert len(step_records) == 2
    for r in step_records:
        assert "multi_attempted" in r
        assert "multi_admitted" in r
        assert "multi_edge_share_cumulative" in r
    assert any("pair_stats_top_pairs" in r for r in step_records)  # PROBE_PERIOD_S divides step 0

    pair_stats_path = out_dir / "pair_stats.json"
    assert pair_stats_path.exists()

    # A second construction with the same out_dir loads pair_stats back.
    model2, tokenizer2 = _tiny_model_and_tokenizer()
    trainer2 = ConPOTrainer(
        model=model2,
        args=_grpo_config(tmp_path / "out2", max_steps=1),
        train_dataset=dataset,
        processing_class=tokenizer2,
        arm=arm,
        rows_by_key=rows_by_key,
        edge_buffer=EdgeBuffer(),
        feasible_buffer=FeasibleBuffer(),
        stats=FamilyStats(),
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism2.jsonl",
        seed=0,
        overrides=overrides,
        out_dir=out_dir,
        peft_config=_lora_config(),
    )

    assert trainer2.pair_stats is not None
    assert trainer2.pair_stats.top_pairs(5) == trainer.pair_stats.top_pairs(5)


def test_construction_without_multi_never_creates_pair_stats(tmp_path: Path) -> None:
    model, tokenizer = _tiny_model_and_tokenizer()
    rows_by_key = {ROW_A.key: ROW_A}
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": ROW_A.prompt}], "key": ROW_A.key}])
    arm, overrides = resolve_arm("solve_only")
    assert overrides.multi is False

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
        judge=StaticJudge(keep=True),
        engine_factory=_engine_factory,
        meter=TokenMeter(budget=0),
        mechanism_log_path=tmp_path / "mechanism.jsonl",
        seed=0,
        overrides=overrides,
        out_dir=tmp_path / "out",
        peft_config=_lora_config(),
    )

    assert trainer.pair_stats is None
    trainer.save_pair_stats()  # no-op, never raises
    assert not (tmp_path / "out" / "pair_stats.json").exists()

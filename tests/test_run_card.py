"""Tests for scripts/run_card.py.

Fixtures below are built to the real shapes read by run_card.py: scripts/
train.py's run_metadata.json and pass1.json, conpo/conpo/trainer.py's
mechanism.jsonl records, scripts/eval_external.py's external.json, and
scripts/eval_internal.py's internal.json.

scripts/ is not an installed package (pyproject.toml only packages
`conpo*`), so run_card is imported by adding scripts/ to sys.path, the same
approach tests/test_eval_scripts.py uses.
"""

import json
import re
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_card  # noqa: E402

AS_OF = "2026-09-03"

# Style contract (USER DECISIONS 2026-09-03): no em dashes, no filler words.
_FILLER_WORDS = ("just", "really", "basically", "actually", "simply", "very")


def _assert_no_banned_words(text: str) -> None:
    assert "—" not in text
    for word in _FILLER_WORDS:
        assert re.search(rf"\b{word}\b", text, re.IGNORECASE) is None, word


def _run_metadata(**overrides: object) -> dict:
    base = {
        "arm": "full",
        "seed": 42,
        "dataset_size": 15000,
        "prompts_per_update": 4,
        "subset_path": "subsets/train_15000_seed42.keys",
        "subset_sha256": "abc123",
        "pass1": {"pass1_step": 3750, "fired": True},
    }
    base.update(overrides)
    return base


def _pass1() -> dict:
    return {"global_step": 3750, "tokens_so_far": 1_200_000, "wall_time_seconds": 4321.5}


def _mechanism_records() -> list[dict]:
    return [
        {
            "step": 1,
            "admitted_by_direction": {"forward": 3, "reverse": 2},
            "veto_discarded_by_direction": {"forward": 1, "reverse": 0},
            "reasons_by_direction": {"forward": {"veto": 1}, "reverse": {}},
            "mean_edit_distance": 0.12,
            "degenerate_group_rate": 0.0,
            "edges_per_million_tokens_step": 50.0,
            "edges_per_million_tokens_cumulative": 50.0,
            "edge_buffer_size": 5,
            "feasible_buffer_size": 8,
            "family_stats": {"length": {"pass_rate": 0.6}},
        },
        {
            "step": 2,
            "admitted_by_direction": {"forward": 2, "reverse": 4},
            "veto_discarded_by_direction": {"forward": 0, "reverse": 2},
            "reasons_by_direction": {"forward": {}, "reverse": {"veto": 2}},
            "mean_edit_distance": 0.18,
            "degenerate_group_rate": 0.1,
            "edges_per_million_tokens_step": 60.0,
            "edges_per_million_tokens_cumulative": 55.0,
            "edge_buffer_size": 11,
            "feasible_buffer_size": 14,
            "family_stats": {"length": {"pass_rate": 0.65}},
        },
        {
            "step": 3,
            "admitted_by_direction": {"forward": 1, "reverse": 1},
            "veto_discarded_by_direction": {"forward": 0, "reverse": 0},
            "reasons_by_direction": {"forward": {}, "reverse": {}},
            "mean_edit_distance": 0.09,
            "degenerate_group_rate": 0.0,
            "edges_per_million_tokens_step": 40.0,
            "edges_per_million_tokens_cumulative": 58.0,
            "edge_buffer_size": 13,
            "feasible_buffer_size": 16,
            "family_stats": {"length": {"pass_rate": 0.7}},
        },
    ]


def _external_json() -> dict:
    ifeval = {
        "prompt_strict": 0.6,
        "prompt_loose": 0.65,
        "instruction_strict": 0.7,
        "instruction_loose": 0.75,
        "num_prompts": 541,
        "num_instructions": 900,
        "per_instruction": {"length_constraints:number_words": 0.5},
    }
    ifbench = {
        "prompt_strict": 0.3,
        "prompt_loose": 0.35,
        "instruction_strict": 0.4,
        "instruction_loose": 0.45,
        "num_prompts": 300,
        "num_instructions": 500,
        "per_instruction": {"count:word_count_range": 0.2},
    }
    return {"ifeval": ifeval, "ifbench": ifbench, "decoding": {"temperature": 0, "max_tokens": 1024}, "model": "m", "adapter": None}


def _internal_json() -> dict:
    def summary(split: str, pacc: float, iacc: float) -> dict:
        return {
            "split": split,
            "num_prompts": 10,
            "pacc": pacc,
            "iacc": iacc,
            "family_accuracies": {"length": pacc},
            "macro_family_accuracy": pacc,
            "worst_decile_family_accuracy": pacc,
            "stratified_by_count": {"1": pacc},
            "mean_length": 42.0,
            "repetition_rate": 0.01,
            "task_relevance": None,
        }

    return {
        "eval_i": summary("eval_i", 0.5, 0.55),
        "eval_ii": summary("eval_ii", 0.6, 0.65),
        "eval_iii": summary("eval_iii", 0.7, 0.75),
    }


def _write_run_dir(tmp_path: Path, name: str, *, with_pass1: bool = True, with_mechanism: bool = True) -> Path:
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps(_run_metadata()), encoding="utf-8")
    if with_pass1:
        (run_dir / "pass1.json").write_text(json.dumps(_pass1()), encoding="utf-8")
    if with_mechanism:
        with (run_dir / "mechanism.jsonl").open("w", encoding="utf-8") as f:
            for record in _mechanism_records():
                f.write(json.dumps(record) + "\n")
    return run_dir


def _write_eval_dir(tmp_path: Path, name: str, *, with_retention: bool = False) -> Path:
    eval_dir = tmp_path / "runs" / name
    eval_dir.mkdir(parents=True)
    (eval_dir / "external.json").write_text(json.dumps(_external_json()), encoding="utf-8")
    (eval_dir / "internal.json").write_text(json.dumps(_internal_json()), encoding="utf-8")
    if with_retention:
        # correct/n counts chosen so correct/n == the fraction each test
        # already asserts (16/20=0.8, 11/20=0.55, 8/20=0.4).
        retention = {
            "gsm8k": 0.8,
            "mmlu": 0.55,
            "bbh": 0.4,
            "n": {"gsm8k": 20, "mmlu": 20, "bbh": 20},
            "correct": {"gsm8k": 16, "mmlu": 11, "bbh": 8},
        }
        (eval_dir / "retention.json").write_text(json.dumps(retention), encoding="utf-8")
    return eval_dir


# --- front matter --------------------------------------------------------------


def test_front_matter_keys_appear_in_contract_order(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_a", with_pass1=False, with_mechanism=False)
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)

    front, _, _ = text.partition("\n---\n")
    body_keys = [line.split(":", 1)[0] for line in front.splitlines() if line != "---"]
    assert body_keys == list(run_card.FRONT_MATTER_KEYS)


def test_front_matter_pending_and_updated(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_b", with_pass1=False, with_mechanism=False)
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)

    assert f"updated: {AS_OF}" in text
    assert "ifbench_prompt_loose: pending" in text
    assert "retention_gsm8k: not wired" in text
    assert "eval_dir: pending" in text


# --- eval_dir (USER DECISIONS 2026-09-03: 27-key contract) -----------------------


def test_contract_has_27_keys_with_eval_dir_immediately_before_updated() -> None:
    # The retention suite's fourth benchmark was dropped (user decision
    # 2026-09-04): 28 keys down to 27, three retention_* keys left.
    assert len(run_card.FRONT_MATTER_KEYS) == 27
    retention_keys = [k for k in run_card.FRONT_MATTER_KEYS if k.startswith("retention_")]
    assert retention_keys == ["retention_gsm8k", "retention_mmlu", "retention_bbh"]
    assert run_card.FRONT_MATTER_KEYS[-2:] == ("eval_dir", "updated")


def test_eval_dir_field_is_repo_relative_posix_path(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_evaldir")
    eval_dir = _write_eval_dir(tmp_path, "eval_tmp_run_evaldir")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, eval_dir, None, AS_OF)

    line = next(line for line in text.splitlines() if line.startswith("eval_dir:"))
    assert "\\" not in line
    assert line.split("eval_dir: ", 1)[1] != run_card.PENDING


def test_eval_dir_pending_when_no_eval_dir_given(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_no_eval")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)
    assert "eval_dir: pending" in text


# --- Wilson 95% CI column (USER DECISIONS 2026-09-03) -----------------------------


def test_wilson_ci_column_matches_stats_wilson_interval_for_a_known_count() -> None:
    """87/300 (IFBench_test's fixed n): the rendered CI must equal
    conpo.stats.wilson_interval(87, 300), formatted to run_card._fmt's
    4dp-stripped convention."""
    from conpo.stats import wilson_interval

    value = 87 / 300
    ci = run_card._wilson_ci_str(value, 300)
    lo, hi = wilson_interval(87, 300)
    assert ci == f"[{run_card._fmt(lo)}, {run_card._fmt(hi)}]"


def test_wilson_ci_column_uses_fixed_n_fallback_when_json_omits_count(tmp_path: Path) -> None:
    """The card fixtures' internal.json splits carry num_prompts=10 (not
    the real 1500/2952/6947), so this checks the fallback path directly:
    no num_prompts in the eval JSON at all falls back to FIXED_N."""
    ci = run_card._wilson_ci_str(0.29, None)
    assert ci == run_card.PENDING
    ci_fallback = run_card._n_or_fixed(None, "ifbench_prompt")
    assert ci_fallback == 300


def test_results_table_carries_wilson_ci_for_pacc_rows(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_ci")
    eval_dir = _write_eval_dir(tmp_path, "eval_tmp_run_ci")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, eval_dir, None, AS_OF)
    results_section = text.split("## 4. Results")[1].split("## 5.")[0]

    assert "95% CI (Wilson)" in results_section
    from conpo.stats import wilson_interval

    # _internal_json's eval_i: pacc=0.5, num_prompts=10 -> k=5
    lo, hi = wilson_interval(5, 10)
    assert f"[{run_card._fmt(lo)}, {run_card._fmt(hi)}]" in results_section
    # retention rows stay "not wired" in both columns (with_retention=False here)
    assert "| Retention gsm8k | not wired | not wired |" in results_section


def test_retention_ci_computed_from_correct_and_n(tmp_path: Path) -> None:
    """Retention rows get a real Wilson CI computed from retention.json's
    own "correct"/"n" counts (design choice, builder 2026-09-04). The user
    dropped the fourth retention benchmark for now (2026-09-04), leaving
    three."""
    run_dir = _write_run_dir(tmp_path, "tmp_run_ret")
    eval_dir = _write_eval_dir(tmp_path, "eval_tmp_run_ret", with_retention=True)
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, eval_dir, None, AS_OF)
    results_section = text.split("## 4. Results")[1].split("## 5.")[0]

    from conpo.stats import wilson_interval

    lo, hi = wilson_interval(16, 20)
    assert f"| Retention gsm8k | 0.8 | [{run_card._fmt(lo)}, {run_card._fmt(hi)}] |" in results_section
    retention_rows = [line for line in results_section.splitlines() if line.startswith("| Retention ")]
    assert len(retention_rows) == 3


def test_retention_ci_falls_back_to_config_retention_sizes_when_n_absent(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_ret2")
    eval_dir = tmp_path / "runs" / "eval_tmp_run_ret2"
    eval_dir.mkdir(parents=True)
    (eval_dir / "external.json").write_text(json.dumps(_external_json()), encoding="utf-8")
    (eval_dir / "internal.json").write_text(json.dumps(_internal_json()), encoding="utf-8")
    (eval_dir / "retention.json").write_text(
        json.dumps({"gsm8k": 0.5, "correct": {"gsm8k": 660}}), encoding="utf-8"
    )
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, eval_dir, None, AS_OF)
    results_section = text.split("## 4. Results")[1].split("## 5.")[0]

    from conpo import config as cfg
    from conpo.stats import wilson_interval

    lo, hi = wilson_interval(660, cfg.RETENTION_SIZES["gsm8k"])
    assert f"| Retention gsm8k | 0.5 | [{run_card._fmt(lo)}, {run_card._fmt(hi)}] |" in results_section


def test_retention_ci_pending_when_benchmark_absent(tmp_path: Path) -> None:
    """conpo/retention.py leaves a benchmark it didn't run out of
    retention.json entirely -- the row must read pending, not crash on a
    missing key or fall back to "not wired"."""
    run_dir = _write_run_dir(tmp_path, "tmp_run_ret3")
    eval_dir = tmp_path / "runs" / "eval_tmp_run_ret3"
    eval_dir.mkdir(parents=True)
    (eval_dir / "external.json").write_text(json.dumps(_external_json()), encoding="utf-8")
    (eval_dir / "internal.json").write_text(json.dumps(_internal_json()), encoding="utf-8")
    (eval_dir / "retention.json").write_text(
        json.dumps({"gsm8k": 0.5, "n": {"gsm8k": 1319}, "correct": {"gsm8k": 660}}), encoding="utf-8"
    )
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, eval_dir, None, AS_OF)
    results_section = text.split("## 4. Results")[1].split("## 5.")[0]
    assert "| Retention mmlu | pending | pending |" in results_section
    assert "| Retention bbh | pending | pending |" in results_section


# --- results table ---------------------------------------------------------------


def test_results_land_in_table(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_c")
    eval_dir = _write_eval_dir(tmp_path, "eval_tmp_run_c", with_retention=True)
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, eval_dir, None, AS_OF)

    results_section = text.split("## 4. Results")[1].split("## 5.")[0]
    assert "| IFEval prompt strict | 0.6 |" in results_section
    assert "| IFBench prompt loose | 0.35 |" in results_section
    assert "| PAcc eval_ii | 0.6 |" in results_section
    assert "| IAcc eval_iii | 0.75 |" in results_section
    assert "| Retention gsm8k | 0.8 |" in results_section

    assert "ifbench_prompt_loose: 0.35" in text
    assert "pacc_eval_i: 0.5" in text
    assert "retention_bbh: 0.4" in text


def test_mechanism_section_reports_admission_and_veto_rates(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_d")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)

    mechanism_section = text.split("## 5. Mechanism metrics")[1]
    # forward: admitted 3+2+1=6, vetoed 1+0+0=1 -> rate 6/7
    assert "forward: admission rate 0.8571" in mechanism_section
    assert "step 3750, tokens 1200000" in mechanism_section
    assert "steps: 3" in text  # last mechanism record's step


def test_mechanism_section_reads_veto_flush_records(tmp_path: Path) -> None:
    """A batched-veto run (RUN_NO_VETO unset, the default) writes per-step
    records with empty admitted_by_direction/veto_discarded_by_direction and
    puts the real per-direction counts in trailing veto_flush records
    (conpo/conpo/trainer.py flush_pending_veto); scripts/train.py also calls
    flush_pending_veto once after training ends, so a completed run's last
    record is often veto_flush. section_mechanism must still report real
    admission/veto rates and pick edges_per_million_tokens_cumulative/
    family_stats from the per-step record that carries them."""
    run_dir = tmp_path / "runs" / "tmp_run_veto_flush"
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps(_run_metadata()), encoding="utf-8")
    records = [
        {
            "step": 5,
            "admitted_by_direction": {},
            "veto_discarded_by_direction": {"forward": 0, "reverse": 0},
            "reasons_by_direction": {"forward": {}, "reverse": {}},
            "mean_edit_distance": 0.1,
            "degenerate_group_rate": 0.0,
            "edges_per_million_tokens_step": 0.0,
            "edges_per_million_tokens_cumulative": 42.0,
            "edge_buffer_size": 0,
            "feasible_buffer_size": 3,
            "family_stats": {"length": {"pass_rate": 0.5}},
        },
        {
            "event": "veto_flush",
            "step": 5,
            "pending_size_before": 10,
            "burst_seconds": 1.2,
            "burst_kept_forward": 4,
            "burst_kept_reverse": 3,
            "burst_rejected_forward": 1,
            "burst_rejected_reverse": 2,
            "veto_rate_forward": 0.2,
            "veto_rate_reverse": 0.4,
        },
    ]
    with (run_dir / "mechanism.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)
    mechanism_section = text.split("## 5. Mechanism metrics")[1]

    # forward: kept 4, rejected 1 -> admission rate 4/5 = 0.8
    assert "forward: admission rate 0.8, veto rate 0.2" in mechanism_section
    # reverse: kept 3, rejected 2 -> admission rate 3/5 = 0.6
    assert "reverse: admission rate 0.6, veto rate 0.4" in mechanism_section
    # cumulative/family_stats come from the per-step record, not "pending"
    assert "edges per million tokens (cumulative): 42" in mechanism_section
    assert "family stats: pending" not in mechanism_section
    assert "| length |" in mechanism_section  # family_stats row from the per-step record


def test_mechanism_section_pending_when_no_admission_data_at_all(tmp_path: Path) -> None:
    """A single per-step record with nothing admitted or vetoed yet must
    still print an explicit pending line, not silently omit it."""
    run_dir = tmp_path / "runs" / "tmp_run_no_data"
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps(_run_metadata()), encoding="utf-8")
    record = {
        "step": 1,
        "admitted_by_direction": {},
        "veto_discarded_by_direction": {},
        "reasons_by_direction": {"forward": {}, "reverse": {}},
        "mean_edit_distance": 0.0,
        "degenerate_group_rate": 0.0,
        "edges_per_million_tokens_step": 0.0,
        "edges_per_million_tokens_cumulative": 0.0,
        "edge_buffer_size": 0,
        "feasible_buffer_size": 0,
        "family_stats": {},
    }
    (run_dir / "mechanism.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)
    mechanism_section = text.split("## 5. Mechanism metrics")[1]
    assert "admission/veto rates: pending" in mechanism_section


# --- status derivation -----------------------------------------------------------


def test_status_planned_when_run_dir_has_no_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "never_run"
    run_dir.mkdir(parents=True)
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)
    assert "status: planned" in text


def test_status_planned_for_h6_multi_with_no_run_dir() -> None:
    plan = run_card.resolve_plan_for_run(Path("runs") / "h6_multi_s42")
    text = run_card.render_card(plan, None, None, None, AS_OF)
    assert "status: planned" in text


def test_status_running_when_metadata_present_and_no_eval(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_e")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)
    assert "status: running" in text


def test_status_done_when_external_json_present(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_f")
    eval_dir = _write_eval_dir(tmp_path, "eval_tmp_run_f")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, eval_dir, None, AS_OF)
    assert "status: done" in text


def test_status_override_wins(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_g")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, "failed", AS_OF)
    assert "status: failed" in text


# --- Notes preservation -----------------------------------------------------------


def test_notes_preserved_across_regeneration(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_h", with_pass1=False, with_mechanism=False)
    cards_dir = tmp_path / "cards"
    plan = run_card.resolve_plan_for_run(run_dir)

    text1 = run_card.render_card(plan, run_dir, None, None, AS_OF)
    path = run_card.write_card(cards_dir, plan.run_id, text1)
    assert path.read_text(encoding="utf-8").endswith(run_card.DEFAULT_NOTES)

    with path.open("a", encoding="utf-8") as f:
        f.write("hand-written observation: the veto rate looks high.\n")

    text2 = run_card.render_card(plan, run_dir, None, None, "2026-09-04")
    run_card.write_card(cards_dir, plan.run_id, text2)

    regenerated = path.read_text(encoding="utf-8")
    assert "hand-written observation: the veto rate looks high." in regenerated
    assert "updated: 2026-09-04" in regenerated  # rest of the card did regenerate


# --- --planned -----------------------------------------------------------------


def test_planned_writes_exactly_24_files_with_listed_run_ids(tmp_path: Path) -> None:
    out_dir = tmp_path / "cards"
    run_card.main(["--planned", "--out", str(out_dir), "--as-of", AS_OF])

    written = {p.stem for p in out_dir.glob("*.md")}
    expected = {plan.run_id for plan in run_card.ALL_PLANS}
    assert len(expected) == 24
    assert written == expected

    expected_train_ids = {
        "full_s42", "full_s43", "full_s44", "untrained",
        "ifrlvr_s42", "ifrlvr_s43", "ifrlvr_s44",
        "solve_only_s42",
        "full_no_exactness_s42", "full_no_distance_s42", "vf_dpo_s42",
        "forward_only_s42", "reverse_only_s42",
        "full_uniform_s42", "full_half_s42", "full_lambda_p0_s42",
        "h6_multi_s42", "edge_dpo_s42",
        "full_qwen3_8b_s42", "full_tulu3_8b_sft_s42", "full_olmo3_7b_sft_s42",
        "full_fullft_s42",
    }
    assert expected_train_ids | {"eval_standard", "eval_retention"} == written


def test_planned_estimates_sum_to_documented_totals(tmp_path: Path) -> None:
    training_total = sum(p.est_h100h for p in run_card.PLANNED_RUNS)
    eval_total = sum(p.est_h100h for p in run_card.PLANNED_EVALS)
    assert training_total == 973.0  # "untrained" adds 0.0; measured 2026-09-11 (handoff hackmd/conpo-plan.md section 6)
    assert eval_total == 7.0  # 28 x 0.24 + 7 x 0.04, measured 2026-09-11 (handoff hackmd/conpo-details.md section 5)


def test_planned_h6_multi_is_planned(tmp_path: Path) -> None:
    out_dir = tmp_path / "cards"
    run_card.main(["--planned", "--out", str(out_dir), "--as-of", AS_OF])
    text = (out_dir / "h6_multi_s42.md").read_text(encoding="utf-8")
    assert "status: planned" in text
    assert "RUN_MULTI=1" in text
    assert "H6: RUN_MULTI=1 built 2026-09-03 (compositor, multi-family edges)" in text


def test_planned_out_defaults_to_hackmd_runs_next_to_repo() -> None:
    parser = run_card.build_arg_parser()
    args = parser.parse_args(["--planned"])
    assert args.out is None
    assert run_card.DEFAULT_CARDS_DIR == run_card.REPO_ROOT.parent / "hackmd" / "runs"


# --- model-axis launch lines (scripts/train.py reads RUN_MODEL) --------------------


def test_model_axis_launch_line_names_run_model() -> None:
    """scripts/train.py:160 reads RUN_MODEL (config.env("RUN_MODEL",
    config.POLICY_MODEL)); the three model-axis planned cards must launch
    with a real RUN_MODEL assignment, not a not-implemented placeholder."""
    model_axis_ids = {"full_qwen3_8b_s42", "full_tulu3_8b_sft_s42", "full_olmo3_7b_sft_s42"}
    for plan in run_card.PLANNED_RUNS:
        if plan.run_id not in model_axis_ids:
            continue
        line = run_card.launch_line(plan)
        assert f"RUN_MODEL={plan.model}" in line
        assert "not yet implemented" not in line


def test_non_model_axis_launch_line_omits_run_model() -> None:
    plan = next(p for p in run_card.PLANNED_RUNS if p.run_id == "full_s42")
    assert "RUN_MODEL" not in run_card.launch_line(plan)


# --- DPO-trainer configuration section (scripts/train_dpo.py's real fields) --------


def test_dpo_planned_card_shows_pairs_not_subset(tmp_path: Path) -> None:
    """vf_dpo/edge_dpo (ARMS[...].trainer == "dpo") train on RUN_PAIRS via
    scripts/train_dpo.py, never on RUN_SUBSET -- the card must show pairs
    path/DPO beta, not the GRPO subset path/config table."""
    out_dir = tmp_path / "cards"
    run_card.main(["--planned", "--out", str(out_dir), "--as-of", AS_OF])
    for run_id in ("vf_dpo_s42", "edge_dpo_s42"):
        text = (out_dir / f"{run_id}.md").read_text(encoding="utf-8")
        config_section = text.split("## 2. Configuration")[1].split("## 3.")[0]
        assert "pairs path (RUN_PAIRS)" in config_section
        assert "DPO beta: 5" in config_section
        assert "subset path" not in config_section
        assert "grpo" not in config_section
        assert "loss_type" not in config_section


def test_dpo_rendered_card_reads_metadata_pairs_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "vf_dpo_s42"
    run_dir.mkdir(parents=True)
    metadata = {
        "arm": "vf_dpo",
        "pairs_path": "data/pairs_vf.jsonl",
        "n_pairs": 4000,
        "seed": 42,
        "steps": 500,
        "beta": 5.0,
        "learning_rate": 1e-5,
        "pair_build_tokens": 12345,
        "actual_h100h": 5.5,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)
    config_section = text.split("## 2. Configuration")[1].split("## 3.")[0]
    assert "pairs path (RUN_PAIRS): data/pairs_vf.jsonl" in config_section
    assert "n_pairs: 4000" in config_section
    assert "DPO beta: 5" in config_section
    assert "steps: 500" in text  # DPO metadata.json's top-level "steps" field


def test_full_arm_planned_card_still_shows_grpo_table(tmp_path: Path) -> None:
    out_dir = tmp_path / "cards"
    run_card.main(["--planned", "--out", str(out_dir), "--as-of", AS_OF])
    text = (out_dir / "full_s42.md").read_text(encoding="utf-8")
    config_section = text.split("## 2. Configuration")[1].split("## 3.")[0]
    assert "subset path" in config_section
    assert "loss_type" in config_section
    assert "pairs path" not in config_section


# --- subset path is platform-independent (posix separators) ------------------------


def test_planned_subset_path_uses_forward_slashes() -> None:
    plan = next(p for p in run_card.PLANNED_RUNS if p.run_id == "full_s42")
    text = run_card.render_card(plan, None, None, None, AS_OF)
    config_section = text.split("## 2. Configuration")[1].split("## 3.")[0]
    line = next(line for line in config_section.splitlines() if line.startswith("- subset path:"))
    assert "\\" not in line
    assert line.endswith("/subsets/train_15000_seed42.keys")


# --- edge_dpo hypotheses (conpo/arms.py registers objective_isolation_edge_dpo) ----


def test_edge_dpo_plan_names_its_hypothesis() -> None:
    plan = next(p for p in run_card.PLANNED_RUNS if p.run_id == "edge_dpo_s42")
    assert plan.hypotheses == ("objective_isolation_edge_dpo",)
    text = run_card.render_card(plan, None, None, None, AS_OF)
    assert "hypotheses: objective_isolation_edge_dpo" in text
    assert "hypotheses served: objective_isolation_edge_dpo" in text


# --- no em dash / no filler --------------------------------------------------------


def test_no_em_dash_in_planned_cards(tmp_path: Path) -> None:
    out_dir = tmp_path / "cards"
    run_card.main(["--planned", "--out", str(out_dir), "--as-of", AS_OF])
    for path in out_dir.glob("*.md"):
        _assert_no_banned_words(path.read_text(encoding="utf-8"))


def test_no_em_dash_in_rendered_run_card(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_i")
    eval_dir = _write_eval_dir(tmp_path, "eval_tmp_run_i", with_retention=True)
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, eval_dir, None, AS_OF)
    _assert_no_banned_words(text)


# --- CLI: single run_dir ----------------------------------------------------------


def test_main_single_run_dir_writes_one_card(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "full_s42")
    eval_dir = _write_eval_dir(tmp_path, "eval_full_s42")
    out_dir = tmp_path / "cards"

    run_card.main(
        [str(run_dir), "--eval-dir", str(eval_dir), "--out", str(out_dir), "--as-of", AS_OF]
    )

    card = (out_dir / "full_s42.md").read_text(encoding="utf-8")
    assert "run_id: full_s42" in card
    assert "block: hub" in card
    assert "hypotheses: H1,H3,H4,H5" in card
    assert "status: done" in card


def test_main_requires_run_dir_or_planned() -> None:
    with pytest.raises(SystemExit):
        run_card.main([])


# --- untrained-policy card (design choice, builder, 2026-09-04) --------------------


def test_untrained_planned_card_is_planned_and_out_of_arms(tmp_path: Path) -> None:
    out_dir = tmp_path / "cards"
    run_card.main(["--planned", "--out", str(out_dir), "--as-of", AS_OF])
    text = (out_dir / "untrained.md").read_text(encoding="utf-8")
    assert "arm: untrained" in text
    assert "block: floor" in text
    assert "status: planned" in text
    assert "est_h100h: 0" in text
    from conpo.arms import ARMS

    assert "untrained" not in ARMS


def test_untrained_status_done_when_eval_dir_has_external_json(tmp_path: Path) -> None:
    """The untrained card never trains (no run_dir with metadata), so its
    status must come from its own eval_dir's external.json, not from
    derive_status's run_metadata.json check (which it would never pass)."""
    plan = next(p for p in run_card.PLANNED_RUNS if p.run_id == "untrained")
    eval_dir = _write_eval_dir(tmp_path, "eval_untrained")
    text = run_card.render_card(plan, None, eval_dir, None, AS_OF)
    assert "status: done" in text


def test_untrained_status_planned_when_no_eval_dir(tmp_path: Path) -> None:
    plan = next(p for p in run_card.PLANNED_RUNS if p.run_id == "untrained")
    text = run_card.render_card(plan, None, None, None, AS_OF)
    assert "status: planned" in text


def test_untrained_status_override_still_wins(tmp_path: Path) -> None:
    plan = next(p for p in run_card.PLANNED_RUNS if p.run_id == "untrained")
    eval_dir = _write_eval_dir(tmp_path, "eval_untrained2")
    text = run_card.render_card(plan, None, eval_dir, "failed", AS_OF)
    assert "status: failed" in text


# --- H4 compute section: admitted edges (design choice, builder, 2026-09-04) -------


def test_last_mechanism_field_reads_last_record_that_carries_it() -> None:
    records = [{"step": 1, "cumulative_admitted_edges": 3}, {"step": 2}, {"step": 3, "cumulative_admitted_edges": 9}]
    assert run_card.last_mechanism_field(records, "cumulative_admitted_edges") == 9
    assert run_card.last_mechanism_field(records, "no_such_field") is None


def test_compute_section_reports_admitted_edges(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_edges")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)
    compute_section = text.split("## 3. Compute")[1].split("## 4.")[0]
    # _mechanism_records's last record with the field carries no
    # cumulative_admitted_edges key at all (the fixture predates wave 2),
    # so this must read pending rather than crash.
    assert "admitted edges: pending" in compute_section


def test_compute_section_reads_cumulative_admitted_edges(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "tmp_run_edges2"
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps(_run_metadata()), encoding="utf-8")
    records = [
        {"step": 1, "cumulative_admitted_edges": 4},
        {"step": 2, "cumulative_admitted_edges": 11},
    ]
    with (run_dir / "mechanism.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)
    compute_section = text.split("## 3. Compute")[1].split("## 4.")[0]
    assert "admitted edges: 11" in compute_section


# --- H5 diagnostic: mean prompt_gamma_theta (design choice, builder, 2026-09-04) ---


def test_mean_of_last_n_with_field_averages_only_the_last_n_that_carry_it() -> None:
    records = [{"prompt_gamma_theta": 1.0}, {"step": 2}, {"prompt_gamma_theta": 3.0}]
    assert run_card.mean_of_last_n_with_field(records, "prompt_gamma_theta", n=20) == 2.0
    assert run_card.mean_of_last_n_with_field(records, "no_such_field") is None


def test_mechanism_section_reports_prompt_gamma_theta_pending_with_no_data(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path, "tmp_run_gamma_pending")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)
    mechanism_section = text.split("## 5. Mechanism metrics")[1]
    assert "prompt gamma theta (mean, last 20 records): pending" in mechanism_section


def test_mechanism_section_reports_mean_prompt_gamma_theta_over_last_20_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "tmp_run_gamma"
    run_dir.mkdir(parents=True)
    (run_dir / "run_metadata.json").write_text(json.dumps(_run_metadata()), encoding="utf-8")
    # 25 records: the first 5 carry no prompt_gamma_theta at all (as if the
    # arm had no prompt term yet), the last 20 do, values 1..20 (mean 10.5).
    records = [{"step": i} for i in range(5)]
    records += [{"step": 5 + i, "prompt_gamma_theta": float(i + 1)} for i in range(20)]
    with (run_dir / "mechanism.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    plan = run_card.resolve_plan_for_run(run_dir)
    text = run_card.render_card(plan, run_dir, None, None, AS_OF)
    mechanism_section = text.split("## 5. Mechanism metrics")[1]
    assert "prompt gamma theta (mean, last 20 records): 10.5" in mechanism_section

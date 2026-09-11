"""Render ConPO run cards under the CARD FRONT-MATTER CONTRACT (USER DECISIONS
2026-09-03): a central hackmd/runs dashboard plus one generated markdown card
per run, produced from run/eval artefacts. `--planned` renders the 22 planned
training cards (21 trained arms plus the untrained floor) and the 2 planned
eval cards from PLANNED_RUNS / PLANNED_EVALS below (design estimates, pre-M0);
a bare run directory renders one train card from its real artefacts, enriched
with the matching planned entry when its run_id is one of the 22 (block,
hypotheses, estimate), else with pending placeholders for those fields.

Reads (see their docstrings for the exact shapes): scripts/train.py
(run_metadata.json, pass1.json), conpo/conpo/trainer.py (mechanism.jsonl
records), scripts/eval_external.py (external.json), scripts/eval_internal.py
(internal.json), conpo/conpo/scoring.py (ScoreReport), conpo/conpo/arms.py
(ARMS, Overrides), conpo/conpo/config.py.

Regenerating a card preserves everything after its "## 6. Notes" heading
byte for byte -- see write_card / apply_notes_preservation.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Any

from conpo import config as cfg
from conpo.arms import ARMS
from conpo.data import keys_digest
from conpo.stats import wilson_interval

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CARDS_DIR = REPO_ROOT.parent / "hackmd" / "runs"
DEFAULT_SUBSET_PATH = REPO_ROOT / "subsets" / "train_15000_seed42.keys"

# Mirrors scripts/train_dpo.py's _DEFAULT_PAIRS: the RUN_PAIRS fallback per
# DPO-trainer arm (ARMS[arm].trainer == "dpo"). scripts/train_dpo.py never
# reads RUN_SUBSET -- a DPO card shows this, not the GRPO subset fields.
DEFAULT_DPO_PAIRS: dict[str, Path] = {
    "vf_dpo": REPO_ROOT / "data" / "pairs_vf.jsonl",
    "relax_dpo": REPO_ROOT / "data" / "pairs_relax.jsonl",
    "edge_dpo": REPO_ROOT / "data" / "pairs_edges.jsonl",
}
# scripts/train_dpo.py hardcodes per_device_train_batch_size (2 under
# --tiny, else 4) -- DPO batch sizing has no config.py knob and is never
# written to metadata.json, so this is documentation, not a read value.
DPO_PER_DEVICE_BATCH = 4

PENDING = "pending"
NOT_WIRED = "not wired"
NOTES_HEADING = "## 6. Notes"
DEFAULT_NOTES = f"{NOTES_HEADING}\n\n(add observations here)\n"

# Card front matter: order is the contract every reader (dashboard.py) depends on.
FRONT_MATTER_KEYS: tuple[str, ...] = (
    "run_id",
    "kind",
    "arm",
    "overrides",
    "block",
    "hypotheses",
    "model",
    "seed",
    "status",
    "est_h100h",
    "est_days_2xh100",
    "actual_h100h",
    "steps",
    "tokens_generated",
    "ifbench_prompt_loose",
    "ifbench_prompt_strict",
    "ifbench_instruction_loose",
    "ifeval_prompt_loose",
    "ifeval_prompt_strict",
    "pacc_eval_i",
    "pacc_eval_ii",
    "pacc_eval_iii",
    "retention_gsm8k",
    "retention_mmlu",
    "retention_bbh",
    "eval_dir",
    "updated",
)

# Fixed prompt-set sizes (USER DECISIONS 2026-09-03), keyed by the same
# name used for the eval JSON's own num_prompts field when present -- the
# Wilson-CI fallback when a card has no real eval JSON to read a count from.
FIXED_N: dict[str, int] = {
    "ifbench_prompt": 300,
    "ifeval_prompt": 541,
    "eval_i": 1500,
    "eval_ii": 2952,
    "eval_iii": 6947,
}

OVERRIDE_MEANINGS: dict[str, str] = {
    "RUN_SELECTION": "reverse-construction selection rule override (conpo/arms.py Overrides.selection)",
    "RUN_NO_VETO": "disable the quality-veto judge gate (conpo/arms.py Overrides.no_veto)",
    "RUN_NO_EXACTNESS": "disable reverse-construction's exactness gate (conpo/arms.py Overrides.no_exactness)",
    "RUN_NO_DISTANCE": "disable reverse-construction's distance gate (conpo/arms.py Overrides.no_distance)",
    "RUN_LAMBDA_P": "override config.LAMBDA_P, the L_prompt weight (conpo/arms.py Overrides.lambda_p)",
}


@dataclasses.dataclass(frozen=True)
class CardPlan:
    """One planned or resolved run's descriptive fields (independent of
    whether it has run yet -- render_card layers real artefacts, when
    given a run_dir/eval_dir, on top of this)."""

    run_id: str
    kind: str  # "train" | "eval"
    arm: str
    overrides: dict[str, str]
    block: str
    hypotheses: tuple[str, ...]
    model: str
    seed: str
    status: str  # "planned"
    est_h100h: float | None
    detail: str = ""
    launch: str = ""  # non-empty overrides the derived local/ launch line


POLICY_MODEL = cfg.POLICY_MODEL


PLANNED_RUNS: tuple[CardPlan, ...] = (
    CardPlan("full_s42", "train", "full", {}, "hub", ("H1", "H3", "H4", "H5"), POLICY_MODEL, "42", "planned", 39.0),
    CardPlan("full_s43", "train", "full", {}, "hub", ("H1", "H3", "H4", "H5"), POLICY_MODEL, "43", "planned", 39.0),
    CardPlan("full_s44", "train", "full", {}, "hub", ("H1", "H3", "H4", "H5"), POLICY_MODEL, "44", "planned", 39.0),
    # The untrained base policy: never trains (arm kept out of conpo.arms.ARMS
    # on purpose), so its status is derived from its own eval_dir alone
    # (see render_card's _EVAL_ONLY_STATUS_RUN_IDS branch) rather than from
    # run_metadata.json/derive_status like every trained arm above.
    CardPlan(
        "untrained", "train", "untrained", {}, "floor", ("H1",), POLICY_MODEL, "-", "planned", 0.0,
        detail="eval hours are counted in the eval job rows.",
        launch="N/A: this policy never trains; bash local/eval.sh and bash local/retention.sh with no adapter",
    ),
    CardPlan("ifrlvr_s42", "train", "ifrlvr", {}, "H1", ("H1",), POLICY_MODEL, "42", "planned", 63.0),
    CardPlan("ifrlvr_s43", "train", "ifrlvr", {}, "H1", ("H1",), POLICY_MODEL, "43", "planned", 63.0),
    CardPlan("ifrlvr_s44", "train", "ifrlvr", {}, "H1", ("H1",), POLICY_MODEL, "44", "planned", 63.0),
    CardPlan("solve_only_s42", "train", "solve_only", {}, "H1 attribution", ("H1",), POLICY_MODEL, "42", "planned", 18.0),
    CardPlan(
        "full_no_exactness_s42", "train", "full", {"RUN_NO_EXACTNESS": "1"}, "H2", ("H2",), POLICY_MODEL, "42",
        "planned", 39.0,
    ),
    CardPlan(
        "full_no_distance_s42", "train", "full", {"RUN_NO_DISTANCE": "1"}, "H2", ("H2",), POLICY_MODEL, "42",
        "planned", 39.0,
    ),
    CardPlan("vf_dpo_s42", "train", "vf_dpo", {}, "H2", ("H2",), POLICY_MODEL, "42", "planned", 18.0),
    CardPlan("forward_only_s42", "train", "forward_only", {}, "H3", ("H3",), POLICY_MODEL, "42", "planned", 39.0),
    CardPlan("reverse_only_s42", "train", "reverse_only", {}, "H3", ("H3",), POLICY_MODEL, "42", "planned", 39.0),
    CardPlan(
        "full_uniform_s42", "train", "full", {"RUN_SELECTION": "uniform"}, "H4", ("H4",), POLICY_MODEL, "42",
        "planned", 39.0,
    ),
    CardPlan(
        "full_half_s42", "train", "full", {"RUN_SELECTION": "half"}, "H4", ("H4",), POLICY_MODEL, "42",
        "planned", 39.0,
    ),
    CardPlan(
        "full_lambda_p0_s42", "train", "full", {"RUN_LAMBDA_P": "0"}, "H5", ("H5",), POLICY_MODEL, "42",
        "planned", 39.0,
    ),
    CardPlan(
        "h6_multi_s42", "train", "full", {"RUN_MULTI": "1"}, "H6", ("H6",), POLICY_MODEL, "42", "planned", 39.0,
        detail="H6: RUN_MULTI=1 built 2026-09-03 (compositor, multi-family edges)",
    ),
    CardPlan(
        "edge_dpo_s42", "train", "edge_dpo", {}, "objective control", ("objective_isolation_edge_dpo",),
        POLICY_MODEL, "42", "planned", 39.0,
    ),
    CardPlan(
        "full_qwen3_8b_s42", "train", "full", {}, "model axis", (), "Qwen/Qwen3-8B", "42", "planned", 78.0,
    ),
    CardPlan(
        "full_tulu3_8b_sft_s42", "train", "full", {}, "model axis", (), "allenai/Llama-3.1-Tulu-3-8B-SFT", "42",
        "planned", 78.0,
    ),
    CardPlan(
        "full_olmo3_7b_sft_s42", "train", "full", {}, "model axis", (), "allenai/Olmo-3-7B-Instruct-SFT", "42",
        "planned", 78.0,
    ),
    CardPlan(
        "full_fullft_s42", "train", "full", {}, "full-FT point-check", (), POLICY_MODEL, "42", "planned", 46.0,
        launch="RUN_ARM=full RUN_SEED=42 RUN_FULL_FT=1 RUN_OUT_DIR=runs/full_fullft_s42 bash local/train_fullft.sh",
    ),
)

PLANNED_EVALS: tuple[CardPlan, ...] = (
    CardPlan(
        "eval_standard", "eval", "IFEval, IFBench, internal PAcc", {},
        "28 checkpoints: 21 runs + 3 IF-RLVR pass1 + 4 untrained", (), "none", "none", "planned", 6.72,
        detail=(
            "IFEval 541, IFBench_test 300, internal eval_i 1500 / eval_ii 2952 / eval_iii 6947; "
            "about 0.24 H100-h per checkpoint."
        ),
    ),
    CardPlan(
        # The retention suite's fourth benchmark was dropped (user decision
        # 2026-09-04). est_h100h is 7 checkpoints x 0.04 H100-h measured on
        # 2026-09-11 (2.4 min per checkpoint). retention.py
        # / scripts/eval_retention.py are wired as of wave 1, so the detail
        # no longer says "not wired" (design choice, builder, 2026-09-04).
        "eval_retention", "eval", "GSM8K, MMLU, BBH", {},
        "7 checkpoints: full x3, ifrlvr token-matched x3, untrained", (), "none", "none", "planned", 0.28,
        detail="about 0.04 H100-h per checkpoint.",
    ),
)

ALL_PLANS: tuple[CardPlan, ...] = PLANNED_RUNS + PLANNED_EVALS
_PLANS_BY_RUN_ID: dict[str, CardPlan] = {plan.run_id: plan for plan in ALL_PLANS}


# --- formatting --------------------------------------------------------------


def _fmt(value: Any) -> str:
    """A front-matter-safe string: "pending" for None, a string passed
    through unchanged, else a plain decimal rounded to at most 4 places."""
    if value is None:
        return PENDING
    if isinstance(value, str):
        return value
    text = f"{round(float(value), 4):.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _days(h100h: float | None) -> float | None:
    """Wall-clock days on 2x H100 with the batched veto (design decision
    2026-09-03: hours / 48, i.e. two GPUs running 24h/day)."""
    return None if h100h is None else h100h / 48


def _n_or_fixed(json_n: Any, fixed_key: str) -> int | None:
    """The eval JSON's own count field when it parses as a positive number,
    else FIXED_N[fixed_key] (or None when fixed_key has no fallback --
    the instruction-level and IAcc rows, design choice 2026-09-03: no
    fixed instruction-count fallback was given, so those CI cells fall
    back to "pending" rather than a made-up n)."""
    if isinstance(json_n, (int, float)) and json_n > 0:
        return int(json_n)
    return FIXED_N.get(fixed_key)


def _wilson_ci_str(value: float | None, n: int | None) -> str:
    """"[lo, hi]" (conpo.stats.wilson_interval, 4dp) for one accuracy row,
    or "pending" when either the value or n is unavailable.

    Design choice 2026-09-03: the eval JSON stores a fraction, not a raw
    pass count, so the count fed to wilson_interval is round(value * n).
    """
    if value is None or n is None or n <= 0:
        return PENDING
    k = max(0, min(n, round(value * n)))
    lo, hi = wilson_interval(k, n)
    return f"[{_fmt(lo)}, {_fmt(hi)}]"


def _retention_row(label: str, retention_raw: dict[str, Any] | None) -> tuple[str, str, str]:
    """One retention row's (name, value, CI): "not wired"/"not wired" when
    the run has no retention.json at all; "pending"/"pending" when
    retention.json exists but `label` is absent (a benchmark not run, see
    conpo/retention.py's shape); else the fraction plus a Wilson 95% CI from
    retention.json's own "correct"/"n" counts, falling back to
    config.RETENTION_SIZES[label] when "n" is missing (design choice,
    builder, 2026-09-04)."""
    name = f"Retention {label}"
    if retention_raw is None:
        return name, NOT_WIRED, NOT_WIRED
    if label not in retention_raw:
        return name, PENDING, PENDING
    value = retention_raw[label]
    n = (retention_raw.get("n") or {}).get(label, cfg.RETENTION_SIZES.get(label))
    correct = (retention_raw.get("correct") or {}).get(label)
    if n is None or correct is None:
        return name, _fmt(value), PENDING
    lo, hi = wilson_interval(correct, n)
    return name, _fmt(value), f"[{_fmt(lo)}, {_fmt(hi)}]"


def eval_dir_str(eval_dir: Path | None) -> str:
    """eval_dir's path relative to REPO_ROOT.parent (the reserch2 root --
    the same convention section_configuration uses for subset_path), or
    "pending" when no eval_dir was given. walk_up=True (Python 3.12) keeps
    this from raising for a path outside the repo tree (e.g. a test's
    tmp_path), rendering a "../"-prefixed relative path instead."""
    if eval_dir is None:
        return PENDING
    return eval_dir.resolve().relative_to(REPO_ROOT.parent.resolve(), walk_up=True).as_posix()


def _overrides_str(overrides: dict[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in overrides.items()) or "none"


def _hyps_str(hypotheses: tuple[str, ...]) -> str:
    return ",".join(hypotheses) or "none"


# --- artefact loading ---------------------------------------------------------


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_metadata(run_dir: Path) -> dict[str, Any] | None:
    """run_metadata.json (the GRPO/ConPOTrainer path, scripts/train.py) when
    present, else metadata.json (the DPO path, scripts/train_dpo.py)."""
    metadata = load_json(run_dir / "run_metadata.json")
    if metadata is not None:
        return metadata
    return load_json(run_dir / "metadata.json")


def load_mechanism(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def extract_external(external: dict[str, Any] | None) -> dict[str, Any]:
    ifeval = (external or {}).get("ifeval") or {}
    ifbench = (external or {}).get("ifbench") or {}
    return {
        "ifeval_prompt_strict": ifeval.get("prompt_strict"),
        "ifeval_prompt_loose": ifeval.get("prompt_loose"),
        "ifeval_instruction_strict": ifeval.get("instruction_strict"),
        "ifeval_instruction_loose": ifeval.get("instruction_loose"),
        "ifeval_num_prompts": ifeval.get("num_prompts"),
        "ifeval_num_instructions": ifeval.get("num_instructions"),
        "ifbench_prompt_strict": ifbench.get("prompt_strict"),
        "ifbench_prompt_loose": ifbench.get("prompt_loose"),
        "ifbench_instruction_strict": ifbench.get("instruction_strict"),
        "ifbench_instruction_loose": ifbench.get("instruction_loose"),
        "ifbench_num_prompts": ifbench.get("num_prompts"),
        "ifbench_num_instructions": ifbench.get("num_instructions"),
    }


def derive_status(has_run_metadata: bool, has_external: bool, planned_status: str) -> str:
    """run dir absent -> planned_status (planned); run_metadata.json present,
    no external.json -> running; external.json
    present -> done. Callers apply an explicit --status override before this."""
    if not has_run_metadata:
        return planned_status
    if has_external:
        return "done"
    return "running"


# CardPlans that never train (see PLANNED_RUNS' "untrained" comment): their
# status comes from their own eval_dir alone -- external.json present means
# done, else the plan's own planned_status -- instead of derive_status's
# run_metadata.json check, which they'd never satisfy.
_EVAL_ONLY_STATUS_RUN_IDS = frozenset({"untrained"})


def resolve_plan_for_run(run_dir: Path) -> CardPlan:
    """The matching PLANNED_RUNS entry by run_id, or a minimal plan derived
    from run_metadata.json (block/hypotheses/estimate stay unknown)."""
    run_id = run_dir.name
    known = _PLANS_BY_RUN_ID.get(run_id)
    if known is not None:
        return known
    metadata = load_metadata(run_dir) or {}
    return CardPlan(
        run_id=run_id,
        kind="train",
        arm=str(metadata.get("arm", "none")),
        overrides={},
        block="none",
        hypotheses=(),
        model=POLICY_MODEL,
        seed=str(metadata.get("seed", "none")),
        status="planned",
        est_h100h=None,
    )


# --- rendering -----------------------------------------------------------------


def front_matter(fields: dict[str, str]) -> str:
    lines = ["---"]
    lines.extend(f"{key}: {fields[key]}" for key in FRONT_MATTER_KEYS)
    lines.append("---")
    return "\n".join(lines)


def launch_line(plan: CardPlan) -> str:
    if plan.kind == "eval":
        return "launched per checkpoint by local/eval.sh; see the checkpoint list in detail below"
    if plan.launch:
        return plan.launch
    arm_spec = ARMS.get(plan.arm)
    if arm_spec is not None and arm_spec.trainer == "dpo":
        return (
            f"RUN_ARM={plan.arm} RUN_SEED={plan.seed} RUN_OUT_DIR=runs/{plan.run_id} "
            "bash local/train_dpo.sh (RUN_STEPS is unset here and local/train_dpo.sh "
            "defaults it to 500 -- set RUN_STEPS explicitly to override)"
        )
    assignments = [f"RUN_ARM={plan.arm}"]
    if plan.model != POLICY_MODEL:
        assignments.append(f"RUN_MODEL={plan.model}")
    for key, value in plan.overrides.items():
        assignments.append(f"{key}={value}")
    assignments.append(f"RUN_SEED={plan.seed}")
    assignments.append(f"RUN_OUT_DIR=runs/{plan.run_id}")
    return " ".join(assignments) + " bash local/train.sh"


def section_header(plan: CardPlan, status: str) -> str:
    lines = [
        "## 1. Header",
        "",
        f"- run_id: {plan.run_id}",
        f"- kind: {plan.kind}",
        f"- block: {plan.block}",
        f"- hypotheses served: {_hyps_str(plan.hypotheses)}",
        f"- status: {status}",
        "- owner: pending",
        "- dates: pending",
        f"- launch: {launch_line(plan)}",
    ]
    if plan.detail:
        lines.append(f"- detail: {plan.detail}")
    return "\n".join(lines)


def section_configuration(plan: CardPlan, metadata: dict[str, Any] | None) -> str:
    arm_spec = ARMS.get(plan.arm)
    is_dpo = arm_spec is not None and arm_spec.trainer == "dpo"

    override_lines = [
        f"  - {key}={value}: {OVERRIDE_MEANINGS.get(key, 'see conpo/arms.py Overrides')}"
        for key, value in plan.overrides.items()
    ] or ["  - none"]

    lines = [
        "## 2. Configuration",
        "",
        f"- model: {plan.model}",
        (
            f"- LoRA: r={cfg.LORA_R} alpha={cfg.LORA_ALPHA} dropout={cfg.LORA_DROPOUT} "
            f"targets={','.join(cfg.LORA_TARGET_MODULES)}"
        ),
    ]

    if is_dpo:
        pairs_path = metadata.get("pairs_path") if metadata else None
        n_pairs = metadata.get("n_pairs") if metadata else None
        beta = metadata.get("beta") if metadata else cfg.DPO_BETA
        if pairs_path is None and metadata is None:
            default_pairs = DEFAULT_DPO_PAIRS.get(plan.arm)
            if default_pairs is not None:
                pairs_path = default_pairs.relative_to(REPO_ROOT.parent).as_posix()
        lines += [
            f"- pairs path (RUN_PAIRS): {_fmt(pairs_path)}",
            f"- n_pairs: {_fmt(n_pairs)}",
            f"- DPO beta: {_fmt(beta)}",
            (
                f"- per_device_batch: {DPO_PER_DEVICE_BATCH} (hardcoded in scripts/train_dpo.py, "
                "not written to metadata.json)"
            ),
            "- overrides:",
            *override_lines,
        ]
        return "\n".join(lines)

    subset_path = metadata.get("subset_path") if metadata else None
    subset_sha256 = metadata.get("subset_sha256") if metadata else None
    if metadata is None and plan.kind == "train" and DEFAULT_SUBSET_PATH.is_file():
        subset_path = DEFAULT_SUBSET_PATH.relative_to(REPO_ROOT.parent).as_posix()
        subset_sha256 = keys_digest(DEFAULT_SUBSET_PATH)
    token_budget = metadata.get("token_budget") if metadata else None

    config_table = (
        "| G | temperature | top_p | max_completion_tokens | LR | effective_batch | loss_type |\n"
        "|---|---|---|---|---|---|---|\n"
        f"| {cfg.NUM_GENERATIONS} | {cfg.TEMPERATURE} | {cfg.TOP_P} | {cfg.MAX_COMPLETION_TOKENS} | "
        f"{cfg.LR} | {cfg.EFFECTIVE_BATCH} | {cfg.LOSS_TYPE} |"
    )

    lines += [
        f"- subset path: {_fmt(subset_path)}",
        f"- subset sha256: {_fmt(subset_sha256)}",
        f"- token budget: {_fmt(token_budget)}",
        "- overrides:",
        *override_lines,
        "",
        config_table,
    ]
    return "\n".join(lines)


def last_mechanism_field(mechanism: list[dict[str, Any]], field: str) -> Any:
    """`field`'s value on the last mechanism.jsonl record that carries it
    (scanned in reverse), or None when no record does -- e.g.
    cumulative_admitted_edges/edges_per_million_tokens_cumulative, which
    live only on per-step records, not on veto_flush records (see
    section_mechanism's docstring)."""
    record = next((r for r in reversed(mechanism) if field in r), None)
    return record.get(field) if record is not None else None


def mean_of_last_n_with_field(mechanism: list[dict[str, Any]], field: str, n: int = 20) -> float | None:
    """Mean of `field` over the last `n` mechanism.jsonl records that carry
    it (conpo/trainer.py merges prompt_gamma_theta into a per-step record
    only when the arm has the prompt term), or None when none do."""
    values = [r[field] for r in mechanism if field in r]
    if not values:
        return None
    tail = values[-n:]
    return sum(tail) / len(tail)


def section_compute(
    plan: CardPlan, actual_h100h: str, steps: str, tokens_generated: str, admitted_edges: str
) -> str:
    lines = [
        "## 3. Compute",
        "",
        f"- estimate: {_fmt(plan.est_h100h)} H100-h ({_fmt(_days(plan.est_h100h))} days on 2x H100)",
        f"- actual H100-h: {actual_h100h}",
        f"- steps: {steps}",
        f"- tokens generated: {tokens_generated}",
        f"- admitted edges: {admitted_edges}",
    ]
    return "\n".join(lines)


def section_results(
    ext: dict[str, Any], internal_splits: dict[str, Any], retention_raw: dict[str, Any] | None
) -> str:
    """USER DECISIONS 2026-09-03: every accuracy row carries a 95% CI
    (Wilson) column, from conpo.stats.wilson_interval via _wilson_ci_str
    (see its docstring for the count-from-fraction and fixed-n-fallback
    choices). Retention rows (GSM8K, MMLU, BBH; the suite's fourth benchmark
    dropped, user decision 2026-09-04) get their own Wilson CI
    from retention.json's "correct"/"n" counts (see _retention_row);
    instruction-level and IAcc rows read "pending" for CI when the eval
    JSON carries no instruction count (no fixed fallback exists for those
    -- design choice 2026-09-03)."""

    def acc_row(name: str, value: Any, n: int | None) -> tuple[str, str, str]:
        return name, _fmt(value), _wilson_ci_str(value, n)

    ifeval_prompt_n = _n_or_fixed(ext["ifeval_num_prompts"], "ifeval_prompt")
    ifeval_instr_n = _n_or_fixed(ext["ifeval_num_instructions"], "")
    ifbench_prompt_n = _n_or_fixed(ext["ifbench_num_prompts"], "ifbench_prompt")
    ifbench_instr_n = _n_or_fixed(ext["ifbench_num_instructions"], "")

    rows: list[tuple[str, str, str]] = [
        acc_row("IFEval prompt strict", ext["ifeval_prompt_strict"], ifeval_prompt_n),
        acc_row("IFEval prompt loose", ext["ifeval_prompt_loose"], ifeval_prompt_n),
        acc_row("IFEval instruction strict", ext["ifeval_instruction_strict"], ifeval_instr_n),
        acc_row("IFEval instruction loose", ext["ifeval_instruction_loose"], ifeval_instr_n),
        acc_row("IFBench prompt strict", ext["ifbench_prompt_strict"], ifbench_prompt_n),
        acc_row("IFBench prompt loose", ext["ifbench_prompt_loose"], ifbench_prompt_n),
        acc_row("IFBench instruction strict", ext["ifbench_instruction_strict"], ifbench_instr_n),
        acc_row("IFBench instruction loose", ext["ifbench_instruction_loose"], ifbench_instr_n),
    ]
    for split in sorted(internal_splits):
        summary = internal_splits[split]
        pacc_n = _n_or_fixed(summary.get("num_prompts"), split)
        rows.append(acc_row(f"PAcc {split}", summary.get("pacc"), pacc_n))
        rows.append((f"IAcc {split}", _fmt(summary.get("iacc")), PENDING))
    for key in ("gsm8k", "mmlu", "bbh"):
        rows.append(_retention_row(key, retention_raw))
    table = ["| metric | value | 95% CI (Wilson) |", "|---|---|---|"]
    table.extend(f"| {name} | {value} | {ci} |" for name, value, ci in rows)
    return "\n".join(["## 4. Results", ""] + table)


def section_mechanism(mechanism: list[dict[str, Any]], pass1: dict[str, Any] | None) -> str:
    """conpo/conpo/trainer.py writes two mechanism.jsonl record shapes into
    the same file: a per-step record (admitted_by_direction/
    veto_discarded_by_direction/edges_per_million_tokens_cumulative/
    family_stats) and, once per flush_pending_veto call, an
    `"event": "veto_flush"` record (burst_kept_*/burst_rejected_*/
    veto_rate_*, none of the per-step keys). Under the batched veto (the
    default: RUN_NO_VETO unset), every admitted candidate is queued and only
    counted at flush time, so admitted_by_direction/veto_discarded_by_direction
    stay empty on every per-step record and the real counts live only in the
    veto_flush records -- both shapes are read here. edges_per_million_tokens
    _cumulative/family_stats are per-step-only fields, so they come from the
    last record that carries them, not blindly the last record in
    the file (a completed run's mechanism.jsonl often ends on a
    veto_flush record: scripts/train.py calls flush_pending_veto once after
    trainer.train() returns). H5 diagnostic (design choice, builder,
    2026-09-04): prompt_gamma_theta is a per-step-only field merged in by
    conpo/trainer.py's compute_loss whenever the arm has the prompt term
    (including under RUN_LAMBDA_P=0); mean_of_last_n_with_field reports its
    mean over the last 20 records that carry it."""
    lines = ["## 5. Mechanism metrics", ""]
    if not mechanism:
        lines.append(f"- admission/veto rates: {PENDING}")
        lines.append(f"- edges per million tokens (cumulative): {PENDING}")
    else:
        admitted: dict[str, int] = {}
        vetoed: dict[str, int] = {}
        for record in mechanism:
            if record.get("event") == "veto_flush":
                for direction in ("forward", "reverse"):
                    admitted[direction] = admitted.get(direction, 0) + record.get(f"burst_kept_{direction}", 0)
                    vetoed[direction] = vetoed.get(direction, 0) + record.get(f"burst_rejected_{direction}", 0)
                continue
            for direction, count in record.get("admitted_by_direction", {}).items():
                admitted[direction] = admitted.get(direction, 0) + count
            for direction, count in record.get("veto_discarded_by_direction", {}).items():
                vetoed[direction] = vetoed.get(direction, 0) + count
        for direction in sorted(set(admitted) | set(vetoed)):
            total = admitted.get(direction, 0) + vetoed.get(direction, 0)
            admission_rate = _fmt(admitted.get(direction, 0) / total) if total else PENDING
            veto_rate = _fmt(vetoed.get(direction, 0) / total) if total else PENDING
            lines.append(f"- {direction}: admission rate {admission_rate}, veto rate {veto_rate}")
        if not admitted and not vetoed:
            lines.append(f"- admission/veto rates: {PENDING}")
        cumulative = last_mechanism_field(mechanism, "edges_per_million_tokens_cumulative")
        lines.append(f"- edges per million tokens (cumulative): {_fmt(cumulative)}")
        family_record = next((r for r in reversed(mechanism) if r.get("family_stats")), None)
        family_stats = family_record.get("family_stats", {}) if family_record else {}
        if family_stats:
            lines.append("- family stats:")
            lines.append("")
            lines.append("| family | p_c | n_c | T_c | U_c | C_c | acquisition | last_probe_update |")
            lines.append("|---|---|---|---|---|---|---|---|")
            for family in sorted(family_stats):
                s = family_stats[family]
                lines.append(
                    f"| {family} | {_fmt(s.get('p_c'))} | {_fmt(s.get('n_c'))} | {_fmt(s.get('T_c'))} | "
                    f"{_fmt(s.get('U_c'))} | {_fmt(s.get('C_c'))} | {_fmt(s.get('acquisition'))} | "
                    f"{_fmt(s.get('last_probe_update'))} |"
                )
        else:
            lines.append(f"- family stats: {PENDING}")
    gamma_mean = mean_of_last_n_with_field(mechanism, "prompt_gamma_theta")
    lines.append(f"- prompt gamma theta (mean, last 20 records): {_fmt(gamma_mean)}")
    if pass1 is not None:
        step = pass1.get("global_step", PENDING)
        tokens = pass1.get("tokens_so_far", PENDING)
        lines.append(f"- pass1 checkpoint: step {step}, tokens {tokens}")
    else:
        lines.append(f"- pass1 checkpoint: {PENDING}")
    return "\n".join(lines)


def render_card(
    plan: CardPlan,
    run_dir: Path | None,
    eval_dir: Path | None,
    status_override: str | None,
    as_of: str,
) -> str:
    """One card's full text, up to and excluding the Notes section (which
    write_card appends, preserving any existing hand-written content)."""
    metadata = load_metadata(run_dir) if run_dir is not None else None
    pass1 = load_json(run_dir / "pass1.json") if run_dir is not None else None
    mechanism = load_mechanism(run_dir / "mechanism.jsonl") if run_dir is not None else []
    external = load_json(eval_dir / "external.json") if eval_dir is not None else None
    internal = load_json(eval_dir / "internal.json") if eval_dir is not None else None
    retention = load_json(eval_dir / "retention.json") if eval_dir is not None else None

    if status_override is not None:
        status = status_override
    elif plan.run_id in _EVAL_ONLY_STATUS_RUN_IDS:
        status = "done" if external is not None else plan.status
    else:
        status = derive_status(metadata is not None, external is not None, plan.status)

    ext = extract_external(external)
    internal_splits = internal or {}

    if mechanism:
        steps = str(mechanism[-1]["step"])
    elif metadata is not None and "steps" in metadata:
        # DPO metadata.json shape (scripts/train_dpo.py): no mechanism.jsonl,
        # steps is a plain top-level field instead.
        steps = str(metadata["steps"])
    else:
        steps = PENDING
    tokens_generated = _fmt(metadata.get("tokens_generated")) if metadata is not None else PENDING
    actual_h100h = _fmt(metadata.get("actual_h100h")) if metadata is not None else PENDING
    admitted_edges = _fmt(last_mechanism_field(mechanism, "cumulative_admitted_edges"))

    fields = {
        "run_id": plan.run_id,
        "kind": plan.kind,
        "arm": plan.arm,
        "overrides": _overrides_str(plan.overrides),
        "block": plan.block,
        "hypotheses": _hyps_str(plan.hypotheses),
        "model": plan.model,
        "seed": plan.seed,
        "status": status,
        "est_h100h": _fmt(plan.est_h100h),
        "est_days_2xh100": _fmt(_days(plan.est_h100h)),
        "actual_h100h": actual_h100h,
        "steps": steps,
        "tokens_generated": tokens_generated,
        "ifbench_prompt_loose": _fmt(ext["ifbench_prompt_loose"]),
        "ifbench_prompt_strict": _fmt(ext["ifbench_prompt_strict"]),
        "ifbench_instruction_loose": _fmt(ext["ifbench_instruction_loose"]),
        "ifeval_prompt_loose": _fmt(ext["ifeval_prompt_loose"]),
        "ifeval_prompt_strict": _fmt(ext["ifeval_prompt_strict"]),
        "pacc_eval_i": _fmt(internal_splits.get("eval_i", {}).get("pacc")),
        "pacc_eval_ii": _fmt(internal_splits.get("eval_ii", {}).get("pacc")),
        "pacc_eval_iii": _fmt(internal_splits.get("eval_iii", {}).get("pacc")),
        "retention_gsm8k": NOT_WIRED if retention is None else _fmt(retention.get("gsm8k")),
        "retention_mmlu": NOT_WIRED if retention is None else _fmt(retention.get("mmlu")),
        "retention_bbh": NOT_WIRED if retention is None else _fmt(retention.get("bbh")),
        "eval_dir": eval_dir_str(eval_dir),
        "updated": as_of,
    }

    return "\n\n".join(
        [
            front_matter(fields),
            f"# Run {plan.run_id}",
            section_header(plan, status),
            section_configuration(plan, metadata),
            section_compute(plan, actual_h100h, steps, tokens_generated, admitted_edges),
            section_results(ext, internal_splits, retention),
            section_mechanism(mechanism, pass1),
        ]
    )


# --- writing (Notes preservation) --------------------------------------------


def _split_notes(text: str) -> tuple[str, str]:
    idx = text.find(NOTES_HEADING)
    if idx == -1:
        return text, ""
    return text[:idx].rstrip("\n"), text[idx:]


def apply_notes_preservation(new_text: str, existing_path: Path) -> str:
    """new_text plus its Notes section: the existing card's Notes verbatim
    when one is already on disk, else a fresh placeholder."""
    head, _ = _split_notes(new_text)
    if existing_path.is_file():
        _, existing_notes = _split_notes(existing_path.read_text(encoding="utf-8"))
        if existing_notes:
            return f"{head}\n\n{existing_notes}"
    return f"{head}\n\n{DEFAULT_NOTES}"


def write_card(out_dir: Path, run_id: str, text: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.md"
    path.write_text(apply_notes_preservation(text, path), encoding="utf-8")
    return path


def write_planned_cards(out_dir: Path, as_of: str) -> list[Path]:
    return [
        write_card(out_dir, plan.run_id, render_card(plan, None, None, None, as_of)) for plan in ALL_PLANS
    ]


# --- CLI -----------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render ConPO run cards (see the card front-matter contract).")
    parser.add_argument("run_dir", nargs="?", default=None, help="run directory (RUN_OUT_DIR) to render a card for")
    parser.add_argument(
        "--planned", action="store_true", help="render the 22 planned training cards and 2 planned eval cards"
    )
    parser.add_argument("--eval-dir", default=None, help="eval directory (external.json / internal.json / retention.json)")
    parser.add_argument("--out", default=None, help="cards output directory (default: hackmd/runs next to the repo)")
    parser.add_argument("--status", default=None, help="override the derived status")
    parser.add_argument("--as-of", default=None, help="ISO date for the 'updated' field (default: today)")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out) if args.out else DEFAULT_CARDS_DIR
    as_of = args.as_of or dt.date.today().isoformat()

    if args.planned:
        if args.run_dir is not None:
            raise SystemExit("--planned takes no run_dir")
        write_planned_cards(out_dir, as_of)
        return

    if args.run_dir is None:
        raise SystemExit("run_dir is required unless --planned is given")

    run_dir = Path(args.run_dir)
    eval_dir = Path(args.eval_dir) if args.eval_dir else None
    plan = resolve_plan_for_run(run_dir)
    text = render_card(plan, run_dir, eval_dir, args.status, as_of)
    write_card(out_dir, plan.run_id, text)


if __name__ == "__main__":
    main()

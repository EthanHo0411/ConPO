"""Build the central runs dashboard from per-run HackMD cards.

``## 0. Shared configuration`` (USER DECISION 2026-09-03, see build_configuration_section)
renders the training and data/evaluation setup common to every run, straight
from ``conpo/conpo/config.py`` and the 15k subset manifest, ahead of the
per-card sections below.

Reads every ``*.md`` card in a cards directory (``README.md`` excluded),
parses the front-matter contract each card starts with (a ``---`` line, one
``key: value`` line per contract key, a closing ``---`` line), and writes
``<cards_dir>/README.md``. The contract's keys, in order, are listed in
``CARD_KEYS`` below; a card whose front matter deviates from that exact key
set, or contains a line that is not ``key: value``, is skipped with a
message on stderr rather than raising.

Design choices not fixed by the card contract (dated 2026-09-03):

* Hypothesis headline metric: ``conpo.arms.HYPOTHESES`` carries no
  H-number-to-metric mapping of its own (every row's ``metric`` field is a
  placeholder kept parseable for ``scripts/analyze.py``'s
  ``_parse_metric``, not the real metric -- see ``conpo.arms.HYPOTHESES``'s
  own H6 comment), so this script derives the dashboard's headline metric
  from the hypothesis id's ``H<n>`` prefix: ``ifbench_prompt_loose`` for H1
  rows, ``pacc_eval_i`` for H2 to H5 rows, ``pacc_eval_iii`` for H6 rows
  (design choice 2026-09-03: eval_iii is the harder split; the Metric
  column shows "PAcc eval_ii, eval_iii" instead, see
  ``hypothesis_metric_display``), and ``ifbench_prompt_loose`` for any row
  outside that numbering (today only ``objective_isolation_edge_dpo``).
* Hypothesis claims: ``conpo.arms.HYPOTHESES`` carries no plain-language
  claim, so ``CLAIMS`` below supplies one short sentence per hypothesis id;
  an id not in ``CLAIMS`` falls back to a generic sentence built from its
  treatment, control and metric.
* Eval job "Checkpoints" and "Suite" columns: the card contract has no
  dedicated fields for these on ``kind: eval`` cards, so this script reads
  them from the same ``block`` and ``arm`` fields a train card uses for its
  block name and arm name -- an eval card's ``block`` holds a free-text
  checkpoint count description and its ``arm`` holds a free-text suite
  description.
* Retention column: the contract stores three separate ``retention_*``
  fields (GSM8K, MMLU, BBH; the retention suite's fourth benchmark was
  dropped, user decision 2026-09-04); the dashboard's single "Retention" cell shows
  ``not wired`` when all three are ``not wired``, ``pending`` when all
  three are ``pending``, and otherwise every value as a Wilson half-width
  cell (see ``retention_cell``), the same style as the IFBench/IFEval
  accuracy cells.
* Totals: an "actual" total sums whatever ``actual_h100h`` values parse as
  a number and ignores ``pending`` ones; if none parse, the total reads
  "pending" rather than 0, since 0 would misreport an all-pending set as
  a completed zero-cost run.
* Hypothesis ids with no ``HYPOTHESES`` entry (none today: H1-H6 and the
  objective-isolation row all have a real ``HYPOTHESES`` entry, H6 as of
  design choice 2026-09-03): whenever a future H-number's arm comparison
  still has no row in ``HYPOTHESES``, the dashboard synthesizes one
  directly from the run card that names it (``block`` equal to the
  H-number, or the H-number listed in the card's comma-separated
  ``hypotheses`` field). ``DEFERRED_HYPOTHESES`` supplies the row's claim,
  display metric and internal metric key; its Delta and Verdict cells run
  the same paired-bootstrap machinery as a HYPOTHESES row, against the
  ``full`` arm as control.

Statistics compute (USER DECISIONS 2026-09-02/2026-09-03, this script's
share of it):

* Run-table accuracy cells: "value +/- half-width" from
  ``conpo.stats.wilson_interval`` at the metric's ``FIXED_N`` prompt count,
  or the bare "pending" when the card's value is pending. A block whose arm
  has 2+ "done" seeds also gets a "<arm> mean +/- std over <k> seeds"
  summary row (``conpo.stats.mean_std`` over each metric's per-seed values).
* Hypothesis Delta [95% CI]: ``conpo.stats.paired_bootstrap_diff`` (seed 0,
  10,000 resamples) on the two arms' prompt-aligned per-prompt verdicts,
  read from ``<eval_dir>/external.json.verdicts.jsonl`` /
  ``internal.json.verdicts.jsonl`` (``eval_dir`` is a card's own
  repo-relative field; ``resolve_eval_dir`` resolves it against
  ``_RESERCH2_ROOT``). A multi-seed arm averages each prompt's verdict
  across its "done" seeds first (``average_verdicts``). "pending" when
  either arm has no per-prompt verdicts yet on that metric.
* H4/H5 row extras (design choice, builder, 2026-09-04): the H4 rows carry
  an extra column with each arm's tokens generated, admitted edges, and
  PAcc(eval_i) per 1e6 generated tokens / per 1e3 admitted edges
  (``h4_compute_cell``, ``rate_per_unit``); the H5 row carries an extra
  column with each arm's mean prompt_gamma_theta over its last 20
  mechanism.jsonl records (``h5_gamma_cell``). Both read mechanism.jsonl
  straight from the training run directory the run_id/RUN_OUT_DIR
  convention implies (``resolve_run_dir``), since that field is not part of
  the card contract. Every other row shows "-" in both columns.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable

from conpo import config as cfg
from conpo.arms import HYPOTHESES, Hypothesis
from conpo.stats import mean_std, paired_bootstrap_diff, verdict_from_ci, wilson_interval

# reserch2 root, two levels up from scripts/dashboard.py (conpo/scripts ->
# conpo -> reserch2) -- the base a card's repo-relative "eval_dir" field
# resolves against (same convention as scripts/run_card.py's eval_dir_str).
_RESERCH2_ROOT = Path(__file__).resolve().parent.parent.parent
_CONPO_REPO_ROOT = _RESERCH2_ROOT / "conpo"

# Default 15k subset manifest (USER DECISION 2026-09-03: the dashboard
# carries run configuration on its own, section 0 below) -- overridable via
# render_dashboard's manifest_path for tests.
SUBSET_MANIFEST_PATH = _CONPO_REPO_ROOT / "subsets" / "train_15000_seed42.manifest.json"
# The subset keys file the manifest describes; only its path is quoted here
# (the manifest carries the sha256, not this file directly).
SUBSET_KEYS_PATH_STR = "conpo/subsets/train_15000_seed42.keys"

CARD_KEYS: tuple[str, ...] = (
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

# Fixed prompt-set sizes (USER DECISIONS 2026-09-03), mirroring
# scripts/run_card.py's FIXED_N -- the Wilson half-width fallback for a run
# table cell when the card's own eval JSON carried no count (a planned card
# reads "pending" regardless; see accuracy_cell below).
FIXED_N: dict[str, int] = {
    "ifbench_prompt_loose": 300,
    "ifeval_prompt_loose": 541,
    "pacc_eval_i": 1500,
    "pacc_eval_ii": 2952,
    "pacc_eval_iii": 6947,
}

PENDING = "pending"

BLOCK_ORDER: tuple[str, ...] = (
    "hub",
    "floor",
    "H1",
    "H1 attribution",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "objective control",
    "model axis",
    "full-FT point-check",
)

CLAIMS: dict[str, str] = {
    "H1": "ConPO's conjunctive-reward counterfactual construction beats the "
    "IF-RLVR additive-reward baseline on constraint-following.",
    "H1_attribution_full_vs_solve_only": "Counterfactual construction adds value on top of the conjunctive reward alone.",
    "H1_attribution_solve_only_vs_ifrlvr": "The conjunctive reward alone beats the IF-RLVR additive reward, "
    "before counterfactual construction is added.",
    "H3_forward": "The forward counterfactual direction contributes to ConPO's full performance.",
    "H3_reverse": "The reverse counterfactual direction contributes to ConPO's full performance.",
    "H4_uniform": "Transfer-weighted selection beats uniform selection.",
    "H4_half": "Transfer-weighted selection beats half selection.",
    "H5_lambda_p_zero": "The prompt-consistency loss L_prompt contributes to ConPO's full performance.",
    "H2_vf_dpo": "ConPO beats verifier-filtered DPO on independently sampled pairs.",
    "objective_isolation_edge_dpo": "ConPO's contrastive objective beats plain DPO trained on ConPO's own edges.",
    "H6_multi": "Interaction-selected multi-constraint edges improve compositional generalisation.",
}

# Claim, display metric, and internal metric key for H-numbers HYPOTHESES
# has no entry for (no arm comparison exists yet), keyed by "H<n>". See the
# module docstring's "Hypothesis ids with no HYPOTHESES entry" note above.
# The internal metric key names the per-prompt verdict source
# hypothesis_delta reads (see _METRIC_VERDICT_SOURCE). H6 is covered by a
# real HYPOTHESES entry today (design choice 2026-09-03), so this entry is
# never read by build_deferred_hypothesis_rows; it stays here, its text
# updated to "built", for the same reason the covered_hypothesis_numbers
# skip check exists -- a future regression that ever un-registers an H6
# HYPOTHESES entry falls back to accurate synthesized text, not stale
# "being built" wording.
DEFERRED_HYPOTHESES: dict[str, tuple[str, str, str]] = {
    "H6": (
        "Multi-constraint compositional generalisation (built 2026-09-03: "
        "RUN_ARM=full RUN_MULTI=1, compositor part A and arm switch part B "
        "both done)",
        "PAcc eval_iii",
        "pacc_eval_iii",
    ),
}

# "Seeds and test" column text (USER DECISIONS 2026-09-03): the literal H1
# id gets the 3-seed mean/std-plus-bootstrap text (the only row with 3
# planned seeds per arm); every other hypothesis id -- including
# H1_attribution_full_vs_solve_only and H1_attribution_solve_only_vs_ifrlvr,
# which are 1-seed diagnostics per hackmd/conpo-plan.md section 6 -- gets the
# 1-seed bootstrap-only text.
_H1_SEEDS_AND_TEST = (
    "3 seeds per arm; mean and std over seeds; paired bootstrap 95% CI "
    "over the 300 IFBench prompts, 10,000 resamples"
)
_OTHER_SEEDS_AND_TEST = "1 seed; paired bootstrap 95% CI over the split's prompts, 10,000 resamples"


def seeds_and_test_text(hypothesis_id: str) -> str:
    return _H1_SEEDS_AND_TEST if hypothesis_id == "H1" else _OTHER_SEEDS_AND_TEST

_LINE_RE = re.compile(r"^([A-Za-z0-9_]+):\s?(.*)$")


def parse_front_matter(text: str) -> dict[str, str] | None:
    """Parse one card's front matter, or None if it deviates from the contract."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    end_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return None
    data: dict[str, str] = {}
    for line in lines[1:end_idx]:
        match = _LINE_RE.match(line)
        if match is None:
            return None
        data[match.group(1)] = match.group(2)
    if set(data.keys()) != set(CARD_KEYS):
        return None
    return data


def load_cards(cards_dir: Path) -> list[dict[str, str]]:
    """Load and parse every card in cards_dir, skipping malformed ones with a stderr note."""
    cards: list[dict[str, str]] = []
    for path in sorted(cards_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        data = parse_front_matter(path.read_text(encoding="utf-8"))
        if data is None:
            print(f"dashboard: skipping {path.name}: malformed front matter", file=sys.stderr)
            continue
        cards.append(data)
    cards.sort(key=lambda c: c["run_id"])
    return cards


def parse_float(value: str) -> float | None:
    """value as a float, or None for "pending" / "not wired" / unparsable text."""
    if value in ("pending", "not wired"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt_arm(card: dict[str, str]) -> str:
    if card["overrides"] == "none":
        return card["arm"]
    return f'{card["arm"]} {card["overrides"]}'


def _fmt_num(x: float) -> str:
    """4dp rounded, trailing zeros/dot stripped -- mirrors
    scripts/run_card.py's _fmt for numeric values (dashboard.py has no
    import path to that script; scripts/ is not an installed package)."""
    text = f"{round(x, 4):.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _half_width_cell(value: float | None, n: int | None) -> str:
    """"value +/- half-width" (Wilson 95% half-width at n), or the bare
    "pending" when value or n is unavailable -- never "pending +/- pending"."""
    if value is None or n is None:
        return PENDING
    k = max(0, min(n, round(value * n)))
    lo, hi = wilson_interval(k, n)
    return f"{_fmt_num(value)} +/- {_fmt_num((hi - lo) / 2)}"


def accuracy_cell(card: dict[str, str], metric: str) -> str:
    """One run-table accuracy cell (USER DECISIONS 2026-09-03: Wilson 95%
    half-width at FIXED_N[metric]) -- see _half_width_cell."""
    return _half_width_cell(parse_float(card[metric]), FIXED_N.get(metric))


def pacc_cell(card: dict[str, str]) -> str:
    return (
        f"{accuracy_cell(card, 'pacc_eval_i')} / "
        f"{accuracy_cell(card, 'pacc_eval_ii')} / "
        f"{accuracy_cell(card, 'pacc_eval_iii')}"
    )


def retention_cell(card: dict[str, str]) -> str:
    """"label=value +/- half-width" per benchmark (three now, the retention
    suite's fourth benchmark dropped, user decision 2026-09-04), Wilson half-width at
    conpo.config.RETENTION_SIZES[label] -- same style as accuracy_cell/the
    IFBench cell (design choice, builder, 2026-09-04)."""
    labels = ("gsm8k", "mmlu", "bbh")
    raw = tuple(card[f"retention_{label}"] for label in labels)
    if all(v == "not wired" for v in raw):
        return "not wired"
    if all(v == "pending" for v in raw):
        return "pending"
    cells = []
    for label, v in zip(labels, raw):
        value = parse_float(v)
        cell = _half_width_cell(value, cfg.RETENTION_SIZES.get(label)) if value is not None else v
        cells.append(f"{label}={cell}")
    return ", ".join(cells)


def card_link(card: dict[str, str]) -> str:
    return f'[{card["run_id"]}]({card["run_id"]}.md)'


def build_seed_summary_row(same_arm_cards: list[dict[str, str]]) -> str | None:
    """"<arm> mean +/- std over <k> seeds" (USER DECISIONS 2026-09-03,
    conpo.stats.mean_std over each metric's per-seed values) for a group of
    cards sharing one arm+overrides+model identifier, or None when fewer
    than 2 of them are "done" -- a summary needs at least 2 real seed
    values from the same model (the model axis puts several models under
    one arm+overrides pair, and those are not seed replicates of each
    other)."""
    done = [c for c in same_arm_cards if c["status"] == "done"]
    if len(done) < 2:
        return None
    label = fmt_arm(done[0])

    def summary_cell(metric: str) -> str:
        values = [v for v in (parse_float(c[metric]) for c in done) if v is not None]
        if not values:
            return PENDING
        mean, std = mean_std(values)
        return f"{_fmt_num(mean)} +/- {_fmt_num(std)}"

    pacc = " / ".join(summary_cell(m) for m in ("pacc_eval_i", "pacc_eval_ii", "pacc_eval_iii"))
    row = [
        f"{label} mean +/- std over {len(done)} seeds",
        label,
        done[0]["model"],
        f"{len(done)} seeds",
        "summary",
        "-",
        "-",
        summary_cell("ifbench_prompt_loose"),
        summary_cell("ifeval_prompt_loose"),
        pacc,
        "-",
        "-",
    ]
    return "| " + " | ".join(row) + " |"


# --- ## 0. Shared configuration ---------------------------------------------
#
# USER DECISION 2026-09-03: the dashboard is the note teammates read, so it
# must carry run configuration (model, dataset, hyperparameters) on its own
# rather than only in per-run cards. This section renders it from
# conpo/conpo/config.py and the 15k subset manifest at generation time. A
# value config.py or the manifest does not define renders "pending", never
# invented.

_SPLIT_ORDER: tuple[str, ...] = ("train", "dev", "eval_i", "eval_ii", "eval_iii")


def load_manifest(manifest_path: Path) -> dict[str, Any] | None:
    """The 15k subset manifest's parsed JSON, or None when it is not on disk."""
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def format_subset_sha256(manifest: dict[str, Any] | None) -> str:
    if manifest is None:
        return PENDING
    return str(manifest.get("keys_sha256", PENDING))


def format_per_count_sizes(manifest: dict[str, Any] | None) -> str:
    counts = manifest.get("per_count_sizes") if manifest else None
    if not counts:
        return PENDING
    return ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items(), key=lambda kv: int(kv[0])))


def format_split_sizes(manifest: dict[str, Any] | None) -> str:
    sizes = manifest.get("source_manifest_split_sizes") if manifest else None
    if not sizes:
        return PENDING
    return " / ".join(f"{split} {sizes[split]:,}" for split in _SPLIT_ORDER if split in sizes)


def _cfg_row(*attrs: str, render: Callable[..., str]) -> str:
    """`render(*values)` from named `cfg` attributes, or "pending" if any is undefined.

    Mirrors the veto-period pattern above, applied uniformly so a config.py
    that drops or renames a constant renders one "pending" cell instead of
    crashing dashboard regeneration.
    """
    values = [getattr(cfg, name, None) for name in attrs]
    if any(v is None for v in values):
        return PENDING
    return render(*values)


def build_training_table() -> list[str]:
    veto_period = getattr(cfg, "VETO_PERIOD_K", None)
    veto_row = str(veto_period) if veto_period is not None else "batched veto, K pending"
    rows = [
        ("policy model", _cfg_row("POLICY_MODEL", render=lambda v: f"{v} (non-thinking)")),
        (
            "LoRA",
            _cfg_row(
                "LORA_R",
                "LORA_ALPHA",
                "LORA_DROPOUT",
                "LORA_TARGET_MODULES",
                render=lambda r, a, d, t: f"r={r} alpha={a} dropout={d} targets={','.join(t)}",
            ),
        ),
        ("G", _cfg_row("NUM_GENERATIONS", render=str)),
        ("temperature", _cfg_row("TEMPERATURE", render=str)),
        ("top-p", _cfg_row("TOP_P", render=str)),
        (
            "max completion / prompt tokens",
            _cfg_row(
                "MAX_COMPLETION_TOKENS",
                "MAX_PROMPT_TOKENS",
                render=lambda c, p: f"{c} / {p}",
            ),
        ),
        ("max_model_len", _cfg_row("MAX_MODEL_LEN", render=str)),
        (
            "optimiser",
            _cfg_row(
                "LR",
                "WARMUP_RATIO",
                "LR_SCHEDULER",
                "MAX_GRAD_NORM",
                render=lambda lr, warmup, sched, clip: (
                    f"AdamW, LR {lr:g}, warm-up ratio {warmup}, {sched} scheduler, grad clip {clip}"
                ),
            ),
        ),
        ("effective batch", _cfg_row("EFFECTIVE_BATCH", render=str)),
        ("loss_type", _cfg_row("LOSS_TYPE", render=str)),
        ("KL beta", _cfg_row("KL_BETA", render=str)),
        (
            "repair caps",
            _cfg_row(
                "MAX_CANDIDATES",
                "MAX_STEPS",
                render=lambda cands, steps: f"{cands} candidates x {steps} steps",
            ),
        ),
        (
            "edge buffer",
            _cfg_row(
                "BUFFER_SIZE",
                "BUFFER_AGE_CAP",
                render=lambda size, cap: f"size {size}, age cap {cap}",
            ),
        ),
        ("judge model", _cfg_row("JUDGE_MODEL", render=str)),
        ("veto period K", veto_row),
    ]
    lines = ["### Training", "", "| setting | value |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines.append("")
    return lines


def build_data_and_evaluation_table(manifest: dict[str, Any] | None) -> list[str]:
    rows = [
        ("pool dataset", _cfg_row("POOL_DATASET", render=str)),
        ("15k subset path", SUBSET_KEYS_PATH_STR),
        ("15k subset sha256", format_subset_sha256(manifest)),
        ("per-constraint-count sizes", format_per_count_sizes(manifest)),
        ("split sizes", format_split_sizes(manifest)),
    ]
    lines = ["### Data and evaluation", "", "| setting | value |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows)
    lines += [
        "",
        "Eval protocol: IFEval 541 and IFBench_test 300 at temperature 0, max new tokens 2048, "
        "thinking off, no system prompt; internal PAcc on the three splits; retention suite GSM8K, "
        "MMLU, BBH on the headline pair and the untrained policy.",
        "",
        "Token-budget rule: ConPO's single pass over the subset, measured at M0; IF-RLVR first-pass "
        "checkpoint plus token-matched run.",
        "",
    ]
    return lines


def build_configuration_section(manifest_path: Path) -> list[str]:
    manifest = load_manifest(manifest_path)
    lines = ["## 0. Shared configuration", ""]
    lines += build_training_table()
    lines += build_data_and_evaluation_table(manifest)
    return lines


def build_runs_section(train_cards: list[dict[str, str]]) -> list[str]:
    lines = ["## 1. Runs", ""]
    header = (
        "| Run | Arm and overrides | Model | Seed | Status | Est H100-h | Actual H100-h "
        "| IFBench prompt-loose | IFEval prompt-loose | PAcc i / ii / iii | Retention | Card |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    for block in BLOCK_ORDER:
        block_cards = [c for c in train_cards if c["block"] == block]
        if not block_cards:
            continue
        lines.append(f"### {block}")
        lines.append("")
        lines.append(header)
        lines.append(sep)
        groups: dict[str, list[dict[str, str]]] = {}
        for c in block_cards:
            row = [
                c["run_id"],
                fmt_arm(c),
                c["model"],
                c["seed"],
                c["status"],
                c["est_h100h"],
                c["actual_h100h"],
                accuracy_cell(c, "ifbench_prompt_loose"),
                accuracy_cell(c, "ifeval_prompt_loose"),
                pacc_cell(c),
                retention_cell(c),
                card_link(c),
            ]
            lines.append("| " + " | ".join(row) + " |")
            groups.setdefault(f'{c["arm"]}|{c["overrides"]}|{c["model"]}', []).append(c)
        for group_cards in groups.values():
            summary_row = build_seed_summary_row(group_cards)
            if summary_row is not None:
                lines.append(summary_row)
        lines.append("")
    return lines


def hypothesis_metric(h: Hypothesis) -> str:
    """The internal metric key a hypothesis row's Delta/Verdict cells
    compute on (a ``_METRIC_VERDICT_SOURCE`` key), derived from the id's
    ``H<n>`` prefix, not ``h.metric`` (see the module docstring)."""
    if h.id.startswith("H1"):
        return "ifbench_prompt_loose"
    if h.id.startswith("H6"):
        # Design choice 2026-09-03: eval_iii is the harder split, so the
        # Delta/Verdict cells compute on it alone; eval_ii is reported in
        # the Metric column only (hypothesis_metric_display).
        return "pacc_eval_iii"
    if h.id[:2] in ("H2", "H3", "H4", "H5"):
        return "pacc_eval_i"
    return "ifbench_prompt_loose"


def hypothesis_metric_display(h: Hypothesis) -> str:
    """The Metric column's text for one HYPOTHESES row: hypothesis_metric's
    internal key for every id, except H6 (design choice 2026-09-03), whose
    plan row (hackmd/conpo-plan.md section 4) reads "PAcc on eval_ii and
    eval_iii" -- both splits are named here even though the Delta/Verdict
    cells compute on eval_iii alone."""
    if h.id.startswith("H6"):
        return "PAcc eval_ii, eval_iii"
    return hypothesis_metric(h)


# Blocks that share an arm/overrides pair with the hypothesis hub (arm="full",
# overrides="none") but are not part of any hypothesis graph: the model-axis
# cards and the full-FT point-check card. A bare identifier match must not
# pull these in alongside the hub's own seed replicates (round 2 finding 1).
_OFF_GRAPH_BLOCKS = {"model axis", "full-FT point-check"}


def resolve_identifier_cards(identifier: str, cards: list[dict[str, str]]) -> list[dict[str, str]]:
    """Cards matching a HYPOTHESES treatment/control identifier.

    A plain arm name (e.g. "full") matches cards with that arm and no
    overrides, excluding off-graph blocks (model axis, full-FT point-check)
    that happen to share the same arm/overrides pair. An "arm:field=value"
    identifier (e.g. "full:selection=uniform") matches cards with that arm
    whose overrides contain the corresponding RUN_<FIELD>=value assignment.
    """
    if ":" in identifier:
        arm, override_expr = identifier.split(":", 1)
        field, value = override_expr.split("=", 1)
        run_assignment = f"RUN_{field.upper()}={value}"
        return [c for c in cards if c["arm"] == arm and run_assignment in c["overrides"].split()]
    return [
        c for c in cards
        if c["arm"] == identifier and c["overrides"] == "none" and c["block"] not in _OFF_GRAPH_BLOCKS
    ]


# One metric key -> (per-prompt verdicts filename under eval_dir, the
# record field to filter on, the value that field must equal). Only the
# metrics hypothesis_metric()/DEFERRED_HYPOTHESES emit are wired:
# ifbench_prompt_loose reads scripts/eval_external.py's
# external.json.verdicts.jsonl (benchmark == "ifbench"); pacc_eval_* reads
# scripts/eval_internal.py's internal.json.verdicts.jsonl (split ==
# "eval_i"/"eval_ii"/"eval_iii"). Both verdict shapes carry a "prompt_loose"
# field per record (eval_internal.py's is the same value as its
# "prompt_strict" -- conpo.verifiers has no strict/loose distinction, see
# that script's module docstring), so every source reads that one field.
_METRIC_VERDICT_SOURCE: dict[str, tuple[str, str, str]] = {
    "ifbench_prompt_loose": ("external.json.verdicts.jsonl", "benchmark", "ifbench"),
    "pacc_eval_i": ("internal.json.verdicts.jsonl", "split", "eval_i"),
    "pacc_eval_ii": ("internal.json.verdicts.jsonl", "split", "eval_ii"),
    "pacc_eval_iii": ("internal.json.verdicts.jsonl", "split", "eval_iii"),
}


def resolve_eval_dir(card: dict[str, str]) -> Path | None:
    """card["eval_dir"] resolved against _RESERCH2_ROOT, or None when the
    card has no real eval_dir yet (still "pending")."""
    value = card.get("eval_dir", PENDING)
    if not value or value == PENDING:
        return None
    return (_RESERCH2_ROOT / value).resolve()


def load_verdict_map(eval_dir: Path, metric: str) -> dict[Any, bool] | None:
    """{prompt key: prompt_loose verdict} for one metric's per-prompt
    verdicts file under eval_dir, or None when the metric has no wired
    source or the file is not on disk yet."""
    source = _METRIC_VERDICT_SOURCE.get(metric)
    if source is None:
        return None
    filename, field, value = source
    path = eval_dir / filename
    if not path.is_file():
        return None
    out: dict[Any, bool] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get(field) == value:
                out[record["key"]] = bool(record["prompt_loose"])
    return out


def average_verdicts(cards: list[dict[str, str]], metric: str) -> dict[Any, float] | None:
    """Per-prompt verdict averaged across every "done" card's own
    per-prompt verdicts (USER DECISIONS 2026-09-03: multi-seed prompt
    averaging -- average each prompt's verdict across the done seeds
    before the paired bootstrap runs). A card that is not "done", or has
    no eval_dir yet, contributes nothing. None when no card contributed a
    verdict map, or the contributing maps share no prompt key."""
    maps: list[dict[Any, bool]] = []
    for card in cards:
        if card["status"] != "done":
            continue
        eval_dir = resolve_eval_dir(card)
        if eval_dir is None:
            continue
        vmap = load_verdict_map(eval_dir, metric)
        if vmap:
            maps.append(vmap)
    if not maps:
        return None
    common_keys = set(maps[0])
    for m in maps[1:]:
        common_keys &= set(m)
    if not common_keys:
        return None
    return {k: sum(m[k] for m in maps) / len(maps) for k in common_keys}


# --- H4/H5 row extras: run_metadata.json's tokens_generated is already on
# the card (front matter); cumulative_admitted_edges and prompt_gamma_theta
# live only in mechanism.jsonl, which is not part of the card contract, so
# these read it straight from the training run directory the run_id/
# RUN_OUT_DIR convention implies (runs/<run_id>, mirroring
# scripts/run_card.py's launch_line and slurm/train.sbatch's RUN_OUT_DIR
# default) -- design choice, builder, 2026-09-04.


def resolve_run_dir(card: dict[str, str]) -> Path:
    return _CONPO_REPO_ROOT / "runs" / card["run_id"]


def _load_mechanism(path: Path) -> list[dict[str, Any]]:
    """Mirrors scripts/run_card.py's load_mechanism (dashboard.py has no
    import path to that script, same reason as _fmt_num above)."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _last_mechanism_field(records: list[dict[str, Any]], field: str) -> Any:
    """Mirrors scripts/run_card.py's last_mechanism_field."""
    record = next((r for r in reversed(records) if field in r), None)
    return record.get(field) if record is not None else None


def admitted_edges_for_card(card: dict[str, str]) -> int | None:
    mechanism = _load_mechanism(resolve_run_dir(card) / "mechanism.jsonl")
    return _last_mechanism_field(mechanism, "cumulative_admitted_edges")


def prompt_gamma_theta_mean_for_card(card: dict[str, str]) -> float | None:
    """Mean of prompt_gamma_theta over the last 20 mechanism.jsonl records
    that carry it (mirrors scripts/run_card.py's
    mean_of_last_n_with_field), or None when none do."""
    mechanism = _load_mechanism(resolve_run_dir(card) / "mechanism.jsonl")
    values = [r["prompt_gamma_theta"] for r in mechanism if "prompt_gamma_theta" in r]
    if not values:
        return None
    tail = values[-20:]
    return sum(tail) / len(tail)


def rate_per_unit(value: float | None, denom: float | None, scale: float) -> str:
    """value / denom * scale, 4dp (mirrors _fmt_num), or "pending" when
    value or denom is missing or non-positive -- the one small helper for
    the H4 row's PAcc-per-1e6-tokens / PAcc-per-1e3-edges arithmetic."""
    if value is None or denom is None or denom <= 0:
        return PENDING
    return _fmt_num(value / denom * scale)


def _first_card(cards: list[dict[str, str]]) -> dict[str, str] | None:
    # ponytail: reports one representative seed's compute/gamma stats, not a
    # multi-seed mean, so a 3-seed arm's H4/H5 columns reflect only 1 of 3
    # seeds; revisit with a real mean if per-seed compute/gamma varies.
    return next((c for c in cards if c["status"] == "done"), cards[0] if cards else None)


def h4_compute_cell(treatment_cards: list[dict[str, str]], control_cards: list[dict[str, str]]) -> str:
    """"tok T/C edges T/C PAcc/1e6tok T/C PAcc/1e3edges T/C" for an H4 row,
    or "-" when either arm resolves to no card at all."""
    t_card, c_card = _first_card(treatment_cards), _first_card(control_cards)
    if t_card is None or c_card is None:
        return "-"
    t_tokens, c_tokens = parse_float(t_card["tokens_generated"]), parse_float(c_card["tokens_generated"])
    t_edges, c_edges = admitted_edges_for_card(t_card), admitted_edges_for_card(c_card)
    t_pacc, c_pacc = parse_float(t_card["pacc_eval_i"]), parse_float(c_card["pacc_eval_i"])

    def fmt_or_pending(v: float | int | None) -> str:
        return _fmt_num(v) if v is not None else PENDING

    tok = f"{fmt_or_pending(t_tokens)}/{fmt_or_pending(c_tokens)}"
    edges = f"{fmt_or_pending(t_edges)}/{fmt_or_pending(c_edges)}"
    per_tok = f"{rate_per_unit(t_pacc, t_tokens, 1e6)}/{rate_per_unit(c_pacc, c_tokens, 1e6)}"
    per_edge = f"{rate_per_unit(t_pacc, t_edges, 1e3)}/{rate_per_unit(c_pacc, c_edges, 1e3)}"
    return f"tok {tok} edges {edges} PAcc/1e6tok {per_tok} PAcc/1e3edges {per_edge}"


def h5_gamma_cell(treatment_cards: list[dict[str, str]], control_cards: list[dict[str, str]]) -> str:
    """"treatment/control" mean prompt_gamma_theta for the H5 row, or "-"
    when either arm resolves to no card at all."""
    t_card, c_card = _first_card(treatment_cards), _first_card(control_cards)
    if t_card is None or c_card is None:
        return "-"
    t_mean = prompt_gamma_theta_mean_for_card(t_card)
    c_mean = prompt_gamma_theta_mean_for_card(c_card)
    t_str = _fmt_num(t_mean) if t_mean is not None else PENDING
    c_str = _fmt_num(c_mean) if c_mean is not None else PENDING
    return f"{t_str}/{c_str}"


def hypothesis_delta(
    treatment_cards: list[dict[str, str]], control_cards: list[dict[str, str]], metric: str, expected_sign: str
) -> tuple[str, str]:
    """(delta string, verdict) for one hypothesis row (USER DECISIONS
    2026-09-03): "+0.034 [+0.010, +0.058]" (3dp, signed) via
    conpo.stats.paired_bootstrap_diff (seed 0, 10,000 resamples) on the
    two arms' prompt-aligned averaged verdict vectors, or ("pending",
    "pending") when either arm has no per-prompt verdicts on `metric`."""
    t_map = average_verdicts(treatment_cards, metric)
    c_map = average_verdicts(control_cards, metric)
    if not t_map or not c_map:
        return PENDING, PENDING
    keys = sorted(set(t_map) & set(c_map), key=str)
    if not keys:
        return PENDING, PENDING
    a = [t_map[k] for k in keys]
    b = [c_map[k] for k in keys]
    point, lo, hi = paired_bootstrap_diff(a, b, n_resamples=10000, seed=0)
    delta_str = f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    return delta_str, verdict_from_ci(lo, hi, expected_sign)


def covered_hypothesis_numbers() -> set[str]:
    """H-number prefixes (e.g. "H1") already covered by a HYPOTHESES entry."""
    numbers: set[str] = set()
    for h in HYPOTHESES:
        match = re.match(r"H\d+", h.id)
        if match is not None:
            numbers.add(match.group(0))
    return numbers


def find_deferred_card(h_number: str, train_cards: list[dict[str, str]]) -> dict[str, str] | None:
    """First train card naming h_number via its block or hypotheses field, or None."""
    for c in train_cards:
        if c["block"] == h_number or h_number in c["hypotheses"].split(","):
            return c
    return None


def build_deferred_hypothesis_rows(train_cards: list[dict[str, str]]) -> list[str]:
    """Synthesized rows for H1..H6 ids HYPOTHESES does not cover, same
    columns as a HYPOTHESES row (design choice 2026-09-03: control is
    always the `full` arm, no overrides -- the hub every H-number
    compares back to per the plan note's section 4)."""
    covered = covered_hypothesis_numbers()
    rows: list[str] = []
    for n in range(1, 7):
        h_number = f"H{n}"
        if h_number in covered or h_number not in DEFERRED_HYPOTHESES:
            continue
        card = find_deferred_card(h_number, train_cards)
        if card is None:
            continue
        claim, metric_display, metric_key = DEFERRED_HYPOTHESES[h_number]
        control_cards = resolve_identifier_cards("full", train_cards)
        delta_str, verdict = hypothesis_delta([card], control_cards, metric_key, ">")
        seeds_and_test = seeds_and_test_text(h_number)
        # No deferred H-number is H4 or H5 today (H4/H5 both have real
        # HYPOTHESES entries), so these two columns are always "-" here.
        rows.append(
            f"| {h_number} | {claim} | {card['run_id']} vs full | {metric_display} | {seeds_and_test} "
            f"| {delta_str} | {verdict} | - | - |"
        )
    return rows


def build_hypotheses_section(train_cards: list[dict[str, str]]) -> list[str]:
    lines = [
        "## 2. Hypothesis verdicts",
        "",
        "| H | Claim | Arms compared | Metric | Seeds and test | Delta [95% CI] | Verdict "
        "| H4 compute tok/edges/rates (T/C) | H5 prompt gamma theta (T/C) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for h in HYPOTHESES:
        metric = hypothesis_metric(h)
        treatment_cards = resolve_identifier_cards(h.treatment, train_cards)
        control_cards = resolve_identifier_cards(h.control, train_cards)
        claim = CLAIMS.get(h.id, f"{h.treatment} outperforms {h.control} on {metric}.")
        arms = f"{h.treatment} vs {h.control}"
        delta_str, verdict = hypothesis_delta(treatment_cards, control_cards, metric, h.expected_sign)
        seeds_and_test = seeds_and_test_text(h.id)
        metric_display = hypothesis_metric_display(h)
        h4_cell = h4_compute_cell(treatment_cards, control_cards) if h.id.startswith("H4") else "-"
        h5_cell = h5_gamma_cell(treatment_cards, control_cards) if h.id.startswith("H5") else "-"
        lines.append(
            f"| {h.id} | {claim} | {arms} | {metric_display} | {seeds_and_test} | {delta_str} | {verdict} "
            f"| {h4_cell} | {h5_cell} |"
        )
    lines += build_deferred_hypothesis_rows(train_cards)
    lines.append("")
    return lines


def build_eval_section(eval_cards: list[dict[str, str]]) -> list[str]:
    lines = [
        "## 3. Evaluation jobs",
        "",
        "| Job | Checkpoints | Suite | Est H100-h | Actual | Status | Card |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in eval_cards:
        row = [c["run_id"], c["block"], c["arm"], c["est_h100h"], c["actual_h100h"], c["status"], card_link(c)]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def sum_numeric(cards: list[dict[str, str]], key: str) -> tuple[float, bool]:
    """(sum of the parseable values, whether any value parsed)."""
    values = [v for v in (parse_float(c[key]) for c in cards) if v is not None]
    return sum(values), len(values) > 0


def _fmt_total(value: float, has_value: bool) -> str:
    return f"{value:.4f}" if has_value else "pending"


def build_totals_section(train_cards: list[dict[str, str]], eval_cards: list[dict[str, str]]) -> list[str]:
    train_est, _ = sum_numeric(train_cards, "est_h100h")
    train_actual, train_has = sum_numeric(train_cards, "actual_h100h")
    eval_est, _ = sum_numeric(eval_cards, "est_h100h")
    eval_actual, eval_has = sum_numeric(eval_cards, "actual_h100h")
    grand_est = train_est + eval_est
    grand_actual = train_actual + eval_actual
    grand_has = train_has or eval_has
    return [
        "## 4. Totals",
        "",
        f"Training: est {train_est:.4f} H100-h, actual {_fmt_total(train_actual, train_has)} H100-h.",
        "",
        f"Evaluation: est {eval_est:.4f} H100-h, actual {_fmt_total(eval_actual, eval_has)} H100-h.",
        "",
        f"Grand total: est {grand_est:.4f} H100-h, actual {_fmt_total(grand_actual, grand_has)} H100-h.",
        "",
        f"Days on 2x H100 (est): {grand_est / 48:.4f}.",
        "",
    ]


def build_statistics_section() -> list[str]:
    """USER DECISIONS 2026-09-03: what the statistics cost, in GPU-hours,
    stated once here rather than repeated per hypothesis row."""
    return [
        "## 5. Statistics compute",
        "",
        "Intervals and bootstraps cost 0 GPU-hours (CPU, seconds, from the stored per-prompt verdicts).",
        "",
        "Seed replication is the only compute that buys statistical power: the 2 extra seeds on the "
        "headline pair cost 48 H100-h of training (2 x (18 + 6)) plus about 4 H100-h of eval and "
        "retention, about 52 H100-h, about 1.1 days on 2x H100, already inside the total; each further "
        "seed on the headline pair costs about 27 H100-h (24 training, 3 eval), about 0.6 days on 2x "
        "H100, and narrows the seed-level std only.",
        "",
        "The prompt-level bootstrap gets tighter only with more eval prompts, and IFBench_test has 300.",
        "",
    ]


def render_dashboard(cards: list[dict[str, str]], as_of: str, manifest_path: Path | None = None) -> str:
    train_cards = [c for c in cards if c["kind"] == "train"]
    eval_cards = [c for c in cards if c["kind"] == "eval"]
    lines = [
        "# ConPO runs dashboard",
        "",
        f"Updated {as_of}. Estimates are pre-M0, factor-of-two accuracy. "
        "Hours are H100-hours; days are on 2x H100 with the batched veto.",
        "",
    ]
    lines += build_configuration_section(manifest_path if manifest_path is not None else SUBSET_MANIFEST_PATH)
    lines += build_runs_section(train_cards)
    lines += build_hypotheses_section(train_cards)
    lines += build_eval_section(eval_cards)
    lines += build_totals_section(train_cards, eval_cards)
    lines += build_statistics_section()
    return "\n".join(lines).rstrip("\n") + "\n"


def default_cards_dir() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return (repo_root / ".." / "hackmd" / "runs").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the ConPO runs dashboard from run cards.")
    parser.add_argument("--cards", type=Path, default=None, help="cards directory (default: ../hackmd/runs)")
    parser.add_argument("--as-of", default=None, help="dashboard date stamp (default: today)")
    args = parser.parse_args()

    cards_dir = args.cards if args.cards is not None else default_cards_dir()
    as_of = args.as_of if args.as_of is not None else date.today().isoformat()

    cards = load_cards(cards_dir)
    content = render_dashboard(cards, as_of)
    (cards_dir / "README.md").write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()

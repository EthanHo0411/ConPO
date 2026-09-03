"""Build the central runs dashboard from per-run HackMD cards.

Reads every ``*.md`` card in a cards directory (``README.md`` excluded),
parses the front-matter contract each card starts with (a ``---`` line, one
``key: value`` line per contract key, a closing ``---`` line), and writes
``<cards_dir>/README.md``. The contract's keys, in order, are listed in
``CARD_KEYS`` below; a card whose front matter deviates from that exact key
set, or contains a line that is not ``key: value``, is skipped with a
message on stderr rather than raising.

Design choices not fixed by the card contract (dated 2026-09-03):

* Hypothesis headline metric: ``conpo.arms.HYPOTHESES`` carries no
  H-number-to-metric mapping of its own (every row's ``metric`` field
  reads ``ifbench_prompt_loose`` today), so this script derives the
  dashboard's headline metric from the hypothesis id's ``H<n>`` prefix:
  ``ifbench_prompt_loose`` for H1 rows, ``pacc_eval_i`` for H2 to H5 rows,
  and ``ifbench_prompt_loose`` for any row outside that numbering (today
  only ``objective_isolation_edge_dpo``).
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
* Retention column: the contract stores four separate ``retention_*``
  fields; the dashboard's single "Retention" cell shows ``not wired`` when
  all four are ``not wired``, ``pending`` when all four are ``pending``, and
  otherwise every value as ``label=value`` pairs.
* Totals: an "actual" total sums whatever ``actual_h100h`` values parse as
  a number and ignores ``pending`` ones; if none parse, the total reads
  "pending" rather than 0, since 0 would misreport an all-pending set as
  a completed zero-cost run.
* Hypothesis ids with no ``HYPOTHESES`` entry (today only H6, being built:
  compositor part A in progress, arm switch part B next): ``HYPOTHESES`` has
  no row to compute a treatment/control delta from, so the dashboard
  synthesizes one row per such H-number directly from the run card that
  names it (``block`` equal to the H-number, or the H-number listed in the
  card's comma-separated ``hypotheses`` field). ``DEFERRED_HYPOTHESES``
  supplies the row's claim and metric text; its Delta and Verdict cells are
  always "pending".
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

from conpo.arms import HYPOTHESES, Hypothesis

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
    "retention_alpacaeval2",
    "updated",
)

BLOCK_ORDER: tuple[str, ...] = (
    "hub",
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
}

# Claim and metric text for H-numbers HYPOTHESES has no entry for (no arm
# comparison exists yet), keyed by "H<n>". See the module docstring's
# "Hypothesis ids with no HYPOTHESES entry" note above.
DEFERRED_HYPOTHESES: dict[str, tuple[str, str]] = {
    "H6": (
        "Multi-constraint compositional generalisation (being built: "
        "compositor part A in progress, arm switch part B next)",
        "PAcc eval_iii",
    ),
}

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


def pacc_cell(card: dict[str, str]) -> str:
    return f'{card["pacc_eval_i"]} / {card["pacc_eval_ii"]} / {card["pacc_eval_iii"]}'


def retention_cell(card: dict[str, str]) -> str:
    labels = ("gsm8k", "mmlu", "bbh", "alpacaeval2")
    values = tuple(card[f"retention_{label}"] for label in labels)
    if all(v == "not wired" for v in values):
        return "not wired"
    if all(v == "pending" for v in values):
        return "pending"
    return ", ".join(f"{label}={v}" for label, v in zip(labels, values))


def card_link(card: dict[str, str]) -> str:
    return f'[{card["run_id"]}]({card["run_id"]}.md)'


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
        for c in block_cards:
            row = [
                c["run_id"],
                fmt_arm(c),
                c["model"],
                c["seed"],
                c["status"],
                c["est_h100h"],
                c["actual_h100h"],
                c["ifbench_prompt_loose"],
                c["ifeval_prompt_loose"],
                pacc_cell(c),
                retention_cell(c),
                card_link(c),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return lines


def hypothesis_metric(h: Hypothesis) -> str:
    if h.id.startswith("H1"):
        return "ifbench_prompt_loose"
    if h.id[:2] in ("H2", "H3", "H4", "H5"):
        return "pacc_eval_i"
    return "ifbench_prompt_loose"


def resolve_identifier_cards(identifier: str, cards: list[dict[str, str]]) -> list[dict[str, str]]:
    """Cards matching a HYPOTHESES treatment/control identifier.

    A plain arm name (e.g. "full") matches cards with that arm and no
    overrides. An "arm:field=value" identifier (e.g. "full:selection=uniform")
    matches cards with that arm whose overrides contain the corresponding
    RUN_<FIELD>=value assignment.
    """
    if ":" in identifier:
        arm, override_expr = identifier.split(":", 1)
        field, value = override_expr.split("=", 1)
        run_assignment = f"RUN_{field.upper()}={value}"
        return [c for c in cards if c["arm"] == arm and run_assignment in c["overrides"].split()]
    return [c for c in cards if c["arm"] == identifier and c["overrides"] == "none"]


def mean_metric(cards: list[dict[str, str]], metric: str) -> float | None:
    values = [v for v in (parse_float(c[metric]) for c in cards) if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


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
    """Synthesized rows for H1..H6 ids HYPOTHESES does not cover."""
    covered = covered_hypothesis_numbers()
    rows: list[str] = []
    for n in range(1, 7):
        h_number = f"H{n}"
        if h_number in covered or h_number not in DEFERRED_HYPOTHESES:
            continue
        card = find_deferred_card(h_number, train_cards)
        if card is None:
            continue
        claim, metric = DEFERRED_HYPOTHESES[h_number]
        rows.append(f"| {h_number} | {claim} | {card['run_id']} | {metric} | pending | pending |")
    return rows


def build_hypotheses_section(train_cards: list[dict[str, str]]) -> list[str]:
    lines = [
        "## 2. Hypothesis verdicts",
        "",
        "| H | Claim | Arms compared | Metric | Delta | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for h in HYPOTHESES:
        metric = hypothesis_metric(h)
        t_mean = mean_metric(resolve_identifier_cards(h.treatment, train_cards), metric)
        c_mean = mean_metric(resolve_identifier_cards(h.control, train_cards), metric)
        claim = CLAIMS.get(h.id, f"{h.treatment} outperforms {h.control} on {metric}.")
        arms = f"{h.treatment} vs {h.control}"
        if t_mean is None or c_mean is None:
            delta_str = "pending"
            verdict = "pending"
        else:
            delta = t_mean - c_mean
            delta_str = f"{delta:.4f}"
            sign_ok = delta > 0 if h.expected_sign == ">" else delta < 0
            verdict = "supports" if sign_ok else "null"
        lines.append(f"| {h.id} | {claim} | {arms} | {metric} | {delta_str} | {verdict} |")
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


def render_dashboard(cards: list[dict[str, str]], as_of: str) -> str:
    train_cards = [c for c in cards if c["kind"] == "train"]
    eval_cards = [c for c in cards if c["kind"] == "eval"]
    lines = [
        "# ConPO runs dashboard",
        "",
        f"Updated {as_of}. Estimates are pre-M0, factor-of-two accuracy. "
        "Hours are H100-hours; days are on 2x H100 with the batched veto.",
        "",
    ]
    lines += build_runs_section(train_cards)
    lines += build_hypotheses_section(train_cards)
    lines += build_eval_section(eval_cards)
    lines += build_totals_section(train_cards, eval_cards)
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

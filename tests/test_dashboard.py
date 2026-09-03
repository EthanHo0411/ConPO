"""Tests for scripts/dashboard.py.

Six synthetic cards (four train, two eval) cover the planned/running/done/
deferred statuses; a seventh malformed card checks the skip-not-crash path.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import dashboard  # noqa: E402

_REPO_ROOT = Path(__file__).parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "dashboard.py"

# Style contract (USER DECISIONS 2026-09-03): no em dashes, no filler words.
_FILLER_WORDS = ("just", "really", "basically", "actually", "simply", "very")


def _assert_no_banned_words(text: str) -> None:
    assert "—" not in text
    for word in _FILLER_WORDS:
        assert re.search(rf"\b{word}\b", text, re.IGNORECASE) is None, word

_TRAIN_KEYS = (
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
    "eval_dir",
    "updated",
)


def _card(**fields: str) -> str:
    values = {key: "pending" for key in _TRAIN_KEYS}
    values.update(fields)
    lines = ["---"]
    for key in _TRAIN_KEYS:
        lines.append(f"{key}: {values[key]}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {values['run_id']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    return "\n".join(lines)


def _write_cards(cards_dir: Path) -> None:
    cards_dir.mkdir(parents=True, exist_ok=True)

    (cards_dir / "full_s42.md").write_text(
        _card(
            run_id="full_s42",
            kind="train",
            arm="full",
            overrides="none",
            block="hub",
            hypotheses="H1,H3,H4,H5",
            model="Qwen/Qwen3-4B",
            seed="42",
            status="done",
            est_h100h="18",
            est_days_2xh100="0.375",
            actual_h100h="19.5",
            steps="1000",
            tokens_generated="200000",
            ifbench_prompt_loose="0.62",
            ifbench_prompt_strict="0.55",
            ifbench_instruction_loose="0.70",
            ifeval_prompt_loose="0.65",
            ifeval_prompt_strict="0.60",
            pacc_eval_i="0.80",
            pacc_eval_ii="0.75",
            pacc_eval_iii="0.70",
            retention_gsm8k="not wired",
            retention_mmlu="not wired",
            retention_bbh="not wired",
            retention_alpacaeval2="not wired",
            updated="2026-09-03",
        ),
        encoding="utf-8",
    )

    (cards_dir / "ifrlvr_s42.md").write_text(
        _card(
            run_id="ifrlvr_s42",
            kind="train",
            arm="ifrlvr",
            overrides="none",
            block="H1",
            hypotheses="H1",
            model="Qwen/Qwen3-4B",
            seed="42",
            status="done",
            est_h100h="6",
            est_days_2xh100="0.125",
            actual_h100h="6.2",
            steps="500",
            tokens_generated="100000",
            ifbench_prompt_loose="0.50",
            ifbench_prompt_strict="0.45",
            ifbench_instruction_loose="0.55",
            ifeval_prompt_loose="0.52",
            ifeval_prompt_strict="0.48",
            pacc_eval_i="0.60",
            pacc_eval_ii="0.58",
            pacc_eval_iii="0.55",
            retention_gsm8k="not wired",
            retention_mmlu="not wired",
            retention_bbh="not wired",
            retention_alpacaeval2="not wired",
            updated="2026-09-03",
        ),
        encoding="utf-8",
    )

    (cards_dir / "solve_only_s42.md").write_text(
        _card(
            run_id="solve_only_s42",
            kind="train",
            arm="solve_only",
            overrides="none",
            block="H1 attribution",
            hypotheses="H1_attribution_full_vs_solve_only,H1_attribution_solve_only_vs_ifrlvr",
            model="Qwen/Qwen3-4B",
            seed="42",
            status="planned",
            est_h100h="6",
            est_days_2xh100="0.125",
            updated="2026-09-03",
        ),
        encoding="utf-8",
    )

    (cards_dir / "h6_multi_s42.md").write_text(
        _card(
            run_id="h6_multi_s42",
            kind="train",
            arm="h6_multi",
            overrides="none",
            block="H6",
            hypotheses="none",
            model="Qwen/Qwen3-4B",
            seed="42",
            status="deferred",
            est_h100h="18",
            est_days_2xh100="0.375",
            updated="2026-09-03",
        ),
        encoding="utf-8",
    )

    (cards_dir / "eval_standard.md").write_text(
        _card(
            run_id="eval_standard",
            kind="eval",
            arm="IFEval, IFBench, internal PAcc",
            overrides="none",
            block="28 checkpoints: 21 runs + 3 IF-RLVR pass1 + 4 untrained",
            hypotheses="none",
            model="none",
            seed="none",
            status="running",
            est_h100h="20",
            est_days_2xh100="0.4167",
            updated="2026-09-03",
        ),
        encoding="utf-8",
    )

    (cards_dir / "eval_retention.md").write_text(
        _card(
            run_id="eval_retention",
            kind="eval",
            arm="GSM8K, MMLU, BBH, AlpacaEval 2",
            overrides="none",
            block="7 checkpoints: full x3, ifrlvr token-matched x3, untrained",
            hypotheses="none",
            model="none",
            seed="none",
            status="planned",
            est_h100h="7",
            est_days_2xh100="0.1458",
            updated="2026-09-03",
        ),
        encoding="utf-8",
    )

    (cards_dir / "broken.md").write_text(
        "---\nrun_id: broken\nthis line has no colon\n---\n",
        encoding="utf-8",
    )


@pytest.fixture
def cards_dir(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    _write_cards(d)
    return d


def test_blocks_appear_in_order(cards_dir: Path) -> None:
    cards = dashboard.load_cards(cards_dir)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    order = ["### hub", "### H1", "### H1 attribution", "### H6"]
    positions = [out.index(marker) for marker in order]
    assert positions == sorted(positions)


# --- Delta [95% CI] / Verdict from synthetic per-prompt verdict files ------------
#
# USER DECISIONS 2026-09-03: the hypothesis Delta/Verdict cells come from
# conpo.stats.paired_bootstrap_diff over each arm's per-prompt verdicts
# (scripts/eval_external.py's external.json.verdicts.jsonl), read from the
# card's own "eval_dir" field, not from a mean of the cards' front-matter
# values. These fixtures write that verdicts file directly, one prompt key
# per record, "ifbench" benchmark, and point the card's eval_dir at it
# (an absolute path here; resolve_eval_dir accepts one unchanged since
# pathlib's "/" treats an absolute right-hand side as a full replacement --
# only scripts/run_card.py's real cards store a repo-relative path).


def _write_ifbench_verdicts(eval_dir: Path, keys: list[int], passed: list[bool]) -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "key": k,
            "benchmark": "ifbench",
            "prompt_strict": p,
            "prompt_loose": p,
            "instruction_id_list": [],
            "instruction_strict": [p],
            "instruction_loose": [p],
        }
        for k, p in zip(keys, passed)
    ]
    with (eval_dir / "external.json.verdicts.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _h1_card(cards_root: Path, run_id: str, arm: str, status: str, eval_dir: Path | None) -> None:
    fields: dict[str, str] = {
        "run_id": run_id,
        "kind": "train",
        "arm": arm,
        "overrides": "none",
        "block": "hub" if arm == "full" else "H1",
        "hypotheses": "H1",
        "model": "Qwen/Qwen3-4B",
        "seed": "42",
        "status": status,
    }
    if eval_dir is not None:
        fields["eval_dir"] = eval_dir.resolve().as_posix()
    (cards_root / f"{run_id}.md").write_text(_card(**fields), encoding="utf-8")


def test_h1_delta_supports_when_treatment_clearly_beats_control(tmp_path: Path) -> None:
    cards_root = tmp_path / "runs"
    cards_root.mkdir()
    keys = list(range(20))
    _write_ifbench_verdicts(tmp_path / "eval_full", keys, [True] * 16 + [False] * 4)  # 0.8
    _write_ifbench_verdicts(tmp_path / "eval_ifrlvr", keys, [True] * 4 + [False] * 16)  # 0.2
    _h1_card(cards_root, "full_s42", "full", "done", tmp_path / "eval_full")
    _h1_card(cards_root, "ifrlvr_s42", "ifrlvr", "done", tmp_path / "eval_ifrlvr")

    cards = dashboard.load_cards(cards_root)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    row = next(line for line in out.splitlines() if line.startswith("| H1 |"))
    assert "+0.600" in row
    assert "supports" in row
    assert "pending" not in row


def test_h1_delta_null_when_arms_are_identical(tmp_path: Path) -> None:
    cards_root = tmp_path / "runs"
    cards_root.mkdir()
    keys = list(range(20))
    same = [True] * 10 + [False] * 10
    _write_ifbench_verdicts(tmp_path / "eval_full", keys, same)
    _write_ifbench_verdicts(tmp_path / "eval_ifrlvr", keys, same)
    _h1_card(cards_root, "full_s42", "full", "done", tmp_path / "eval_full")
    _h1_card(cards_root, "ifrlvr_s42", "ifrlvr", "done", tmp_path / "eval_ifrlvr")

    cards = dashboard.load_cards(cards_root)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    row = next(line for line in out.splitlines() if line.startswith("| H1 |"))
    assert "+0.000" in row
    assert "| null |" in row


def test_h1_delta_null_opposite_sign_when_control_beats_treatment(tmp_path: Path) -> None:
    cards_root = tmp_path / "runs"
    cards_root.mkdir()
    keys = list(range(20))
    _write_ifbench_verdicts(tmp_path / "eval_full", keys, [True] * 4 + [False] * 16)  # 0.2
    _write_ifbench_verdicts(tmp_path / "eval_ifrlvr", keys, [True] * 16 + [False] * 4)  # 0.8
    _h1_card(cards_root, "full_s42", "full", "done", tmp_path / "eval_full")
    _h1_card(cards_root, "ifrlvr_s42", "ifrlvr", "done", tmp_path / "eval_ifrlvr")

    cards = dashboard.load_cards(cards_root)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    row = next(line for line in out.splitlines() if line.startswith("| H1 |"))
    assert "-0.600" in row
    assert "null (opposite sign)" in row


def test_h1_delta_pending_when_one_arm_has_no_verdicts_yet(tmp_path: Path) -> None:
    cards_root = tmp_path / "runs"
    cards_root.mkdir()
    _write_ifbench_verdicts(tmp_path / "eval_full", list(range(20)), [True] * 16 + [False] * 4)
    _h1_card(cards_root, "full_s42", "full", "done", tmp_path / "eval_full")
    _h1_card(cards_root, "ifrlvr_s42", "ifrlvr", "planned", None)  # not yet run

    cards = dashboard.load_cards(cards_root)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    row = next(line for line in out.splitlines() if line.startswith("| H1 |"))
    assert row.endswith("| pending | pending |")
    assert "supports" not in row


def test_multi_seed_averages_per_prompt_before_the_bootstrap(tmp_path: Path) -> None:
    """Two "done" seeds of the treatment arm: average_verdicts must average
    each prompt's verdict across both seeds before paired_bootstrap_diff
    runs -- the point diff is exact (mean(a) - mean(b) on the full data,
    no resampling), so it is asserted precisely."""
    cards_root = tmp_path / "runs"
    cards_root.mkdir()
    keys = [0, 1, 2, 3]
    _write_ifbench_verdicts(tmp_path / "eval_full_s42", keys, [True, True, False, False])
    _write_ifbench_verdicts(tmp_path / "eval_full_s43", keys, [True, False, True, False])
    # per-key average: 1.0, 0.5, 0.5, 0.0 -> mean 0.5
    _write_ifbench_verdicts(tmp_path / "eval_ifrlvr", keys, [False, False, False, False])  # mean 0.0
    _h1_card(cards_root, "full_s42", "full", "done", tmp_path / "eval_full_s42")
    _h1_card(cards_root, "full_s43", "full", "done", tmp_path / "eval_full_s43")
    _h1_card(cards_root, "ifrlvr_s42", "ifrlvr", "done", tmp_path / "eval_ifrlvr")

    cards = dashboard.load_cards(cards_root)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    row = next(line for line in out.splitlines() if line.startswith("| H1 |"))
    assert "+0.500" in row


def test_off_graph_full_card_excluded_from_hub_delta(tmp_path: Path) -> None:
    """A "full"/no-overrides card from an off-graph block (model axis or
    full-FT point-check) must not be pulled into the hub's per-prompt
    averaging alongside the hub's own seed replicates: the H1 delta with
    the off-graph card present must equal the hub-only delta (round 2
    finding 1)."""
    cards_root = tmp_path / "runs"
    cards_root.mkdir()
    keys = list(range(20))
    _write_ifbench_verdicts(tmp_path / "eval_full", keys, [True] * 16 + [False] * 4)  # 0.8
    _write_ifbench_verdicts(tmp_path / "eval_ifrlvr", keys, [True] * 4 + [False] * 16)  # 0.2
    _write_ifbench_verdicts(tmp_path / "eval_off_graph", keys, [True] * 2 + [False] * 18)  # 0.1
    _h1_card(cards_root, "full_s42", "full", "done", tmp_path / "eval_full")
    _h1_card(cards_root, "ifrlvr_s42", "ifrlvr", "done", tmp_path / "eval_ifrlvr")
    (cards_root / "full_qwen3_8b_s42.md").write_text(
        _card(
            run_id="full_qwen3_8b_s42",
            kind="train",
            arm="full",
            overrides="none",
            block="model axis",
            hypotheses="none",
            model="Qwen/Qwen3-8B",
            seed="42",
            status="done",
            eval_dir=(tmp_path / "eval_off_graph").resolve().as_posix(),
        ),
        encoding="utf-8",
    )

    cards = dashboard.load_cards(cards_root)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    row = next(line for line in out.splitlines() if line.startswith("| H1 |"))
    assert "+0.600" in row
    assert "supports" in row


# --- "Seeds and test" column text -------------------------------------------------


def test_seeds_and_test_text_h1_vs_attribution_rows(cards_dir: Path) -> None:
    """The literal H1 id gets the 3-seed text; the two H1_attribution_* ids
    (1 seed each per hackmd/conpo-plan.md section 6) must not match on the
    "H1" prefix and fall through to the 1-seed text."""
    cards = dashboard.load_cards(cards_dir)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    h1_row = next(line for line in out.splitlines() if line.startswith("| H1 |"))
    attr_rows = [
        line
        for line in out.splitlines()
        if line.startswith("| H1_attribution_full_vs_solve_only |")
        or line.startswith("| H1_attribution_solve_only_vs_ifrlvr |")
    ]
    assert len(attr_rows) == 2
    assert "3 seeds per arm; mean and std over seeds" in h1_row
    for row in attr_rows:
        assert "3 seeds per arm" not in row
        assert "1 seed; paired bootstrap 95% CI over the split's prompts, 10,000 resamples" in row


# --- Seed summary row ("<arm> mean +/- std over <k> seeds") -----------------------


def _seed_card(
    cards_root: Path, run_id: str, model: str, seed: str, status: str, ifbench_prompt_loose: str
) -> None:
    fields: dict[str, str] = {
        "run_id": run_id,
        "kind": "train",
        "arm": "full",
        "overrides": "none",
        "block": "model axis",
        "hypotheses": "none",
        "model": model,
        "seed": seed,
        "status": status,
        "ifbench_prompt_loose": ifbench_prompt_loose,
    }
    (cards_root / f"{run_id}.md").write_text(_card(**fields), encoding="utf-8")


def test_seed_summary_row_needs_two_done_seeds(tmp_path: Path) -> None:
    cards_root = tmp_path / "runs"
    cards_root.mkdir()
    _seed_card(cards_root, "full_s42", "Qwen/Qwen3-4B", "42", "done", "0.60")
    _seed_card(cards_root, "full_s43", "Qwen/Qwen3-4B", "43", "planned", "pending")

    cards = dashboard.load_cards(cards_root)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    assert "mean +/- std over" not in out

    _seed_card(cards_root, "full_s43", "Qwen/Qwen3-4B", "43", "done", "0.70")
    cards = dashboard.load_cards(cards_root)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    assert "full mean +/- std over 2 seeds" in out


def test_seed_summary_row_not_synthesized_across_different_models(tmp_path: Path) -> None:
    """Two "done" cards sharing arm+overrides but not model (the model-axis
    block) are not seed replicates of one experiment and must not be
    grouped into a false summary row."""
    cards_root = tmp_path / "runs"
    cards_root.mkdir()
    _seed_card(cards_root, "full_qwen3_8b_s42", "Qwen/Qwen3-8B", "42", "done", "0.60")
    _seed_card(cards_root, "full_tulu3_8b_sft_s42", "allenai/Llama-3.1-Tulu-3-8B-SFT", "42", "done", "0.65")

    cards = dashboard.load_cards(cards_root)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    assert "mean +/- std over" not in out


# --- ## 5. Statistics compute -----------------------------------------------------


def test_statistics_compute_section_present(cards_dir: Path) -> None:
    cards = dashboard.load_cards(cards_dir)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    assert "## 5. Statistics compute" in out
    assert "Intervals and bootstraps cost 0 GPU-hours" in out
    assert "48 H100-h" in out
    assert "about 1.1 days on 2x H100" in out
    assert "about 27 H100-h" in out
    assert "IFBench_test has 300" in out


def test_pending_propagates(cards_dir: Path) -> None:
    cards = dashboard.load_cards(cards_dir)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    lines = [line for line in out.splitlines() if line.startswith("| H3_forward |")]
    assert len(lines) == 1
    row = lines[0]
    assert "pending" in row
    assert "supports" not in row
    assert "null" not in row


def test_readme_links_target_existing_cards(cards_dir: Path) -> None:
    cards = dashboard.load_cards(cards_dir)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    for card in cards:
        link = f"({card['run_id']}.md)"
        assert link in out
        assert (cards_dir / f"{card['run_id']}.md").exists()


def test_malformed_card_skipped_and_reported(cards_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cards = dashboard.load_cards(cards_dir)
    run_ids = [card["run_id"] for card in cards]
    assert "broken" not in run_ids
    assert len(cards) == 6
    captured = capsys.readouterr()
    assert "broken.md" in captured.err


def test_regeneration_is_byte_identical(cards_dir: Path) -> None:
    cards = dashboard.load_cards(cards_dir)
    first = dashboard.render_dashboard(cards, "2026-09-03")
    second = dashboard.render_dashboard(cards, "2026-09-03")
    assert first.encode("utf-8") == second.encode("utf-8")


def test_no_em_dash_in_output(cards_dir: Path) -> None:
    cards = dashboard.load_cards(cards_dir)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    _assert_no_banned_words(out)


def test_eval_jobs_section(cards_dir: Path) -> None:
    cards = dashboard.load_cards(cards_dir)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    assert "## 3. Evaluation jobs" in out
    assert "eval_standard" in out
    assert "eval_retention" in out


def test_totals_section_present(cards_dir: Path) -> None:
    cards = dashboard.load_cards(cards_dir)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    assert "## 4. Totals" in out
    assert "Grand total" in out


def test_h6_row_synthesized_h1_to_h5_unchanged(cards_dir: Path) -> None:
    """h6_multi_s42 (block H6, no HYPOTHESES entry) gets a synthesized row;
    every HYPOTHESES id (H1-H5, plus H2/objective-isolation) keeps its own
    unaltered row."""
    cards = dashboard.load_cards(cards_dir)
    out = dashboard.render_dashboard(cards, "2026-09-03")

    h6_lines = [line for line in out.splitlines() if line.startswith("| H6 |")]
    assert len(h6_lines) == 1
    h6_row = h6_lines[0]
    assert "Multi-constraint compositional generalisation" in h6_row
    assert "being built" in h6_row
    assert "compositor part A in progress, arm switch part B next" in h6_row
    assert "h6_multi_s42" in h6_row
    assert "PAcc eval_iii" in h6_row
    assert h6_row.count("pending") == 2
    assert "deferred" not in h6_row

    for h in dashboard.HYPOTHESES:
        rows = [line for line in out.splitlines() if line.startswith(f"| {h.id} |")]
        assert len(rows) == 1


def test_cli_writes_readme(cards_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--cards", str(cards_dir), "--as-of", "2026-09-03"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    readme = cards_dir / "README.md"
    assert readme.exists()
    content = readme.read_text(encoding="utf-8")
    assert content.startswith("# ConPO runs dashboard")

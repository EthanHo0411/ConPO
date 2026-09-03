"""Tests for scripts/dashboard.py.

Six synthetic cards (four train, two eval) cover the planned/running/done/
deferred statuses; a seventh malformed card checks the skip-not-crash path.
"""

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


def test_done_pair_produces_numeric_delta_and_verdict(cards_dir: Path) -> None:
    cards = dashboard.load_cards(cards_dir)
    out = dashboard.render_dashboard(cards, "2026-09-03")
    lines = [line for line in out.splitlines() if line.startswith("| H1 |")]
    assert len(lines) == 1
    row = lines[0]
    assert "0.1200" in row
    assert "supports" in row
    assert "pending" not in row


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


def test_h6_deferred_row_synthesized_h1_to_h5_unchanged(cards_dir: Path) -> None:
    """h6_multi_s42 (block H6, no HYPOTHESES entry) gets a synthesized row;
    every HYPOTHESES id (H1-H5, plus H2/objective-isolation) keeps its own
    unaltered row."""
    cards = dashboard.load_cards(cards_dir)
    out = dashboard.render_dashboard(cards, "2026-09-03")

    h6_lines = [line for line in out.splitlines() if line.startswith("| H6 |")]
    assert len(h6_lines) == 1
    h6_row = h6_lines[0]
    assert "Multi-constraint compositional generalisation" in h6_row
    assert "phi-coefficient compositor not built" in h6_row
    assert "h6_multi_s42" in h6_row
    assert "PAcc eval_iii" in h6_row
    assert h6_row.count("pending") == 1
    assert "deferred" in h6_row

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

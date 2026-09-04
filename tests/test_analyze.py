"""Tests for scripts/analyze.py on a synthetic runs directory.

scripts/ is not an installed package, so it is imported by adding scripts/
to sys.path, same pattern as tests/test_eval_scripts.py.
"""

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import analyze  # noqa: E402

IFBENCH_KEYS = [str(i) for i in range(10)]


def _write_run(base: Path, arm: str, seed: int, pacc: float, ifbench_pass: bool) -> None:
    run_dir = base / f"{arm}_s{seed}"
    run_dir.mkdir(parents=True)

    external = {
        "ifeval": None,
        "ifbench": {
            "prompt_strict": 1.0 if ifbench_pass else 0.0,
            "prompt_loose": 1.0 if ifbench_pass else 0.0,
            "instruction_strict": 1.0 if ifbench_pass else 0.0,
            "instruction_loose": 1.0 if ifbench_pass else 0.0,
        },
    }
    (run_dir / "eval_external.json").write_text(json.dumps(external), encoding="utf-8")
    with (run_dir / "eval_external.json.verdicts.jsonl").open("w", encoding="utf-8") as f:
        for key in IFBENCH_KEYS:
            f.write(json.dumps({"key": key, "benchmark": "ifbench", "prompt_strict": ifbench_pass, "prompt_loose": ifbench_pass}) + "\n")

    internal = {"dev": {"pacc": pacc}}
    (run_dir / "eval_internal.json").write_text(json.dumps(internal), encoding="utf-8")
    with (run_dir / "eval_internal.json.verdicts.jsonl").open("w", encoding="utf-8") as f:
        for key in ["d0", "d1"]:
            f.write(json.dumps({"key": key, "split": "dev", "prompt_strict": True, "prompt_loose": True}) + "\n")


def _build_synthetic_runs(tmp_path: Path) -> Path:
    runs_dir = tmp_path / "runs"
    for seed in (42, 43):
        _write_run(runs_dir, "full", seed, pacc=0.8, ifbench_pass=True)
        _write_run(runs_dir, "ifrlvr", seed, pacc=0.4, ifbench_pass=False)
    return runs_dir


# --- discovery ---------------------------------------------------------------


def test_discover_runs_parses_arm_and_seed(tmp_path: Path) -> None:
    runs_dir = _build_synthetic_runs(tmp_path)
    refs = analyze.discover_runs(runs_dir)
    found = {(r.arm, r.seed) for r in refs}
    assert found == {("full", 42), ("full", 43), ("ifrlvr", 42), ("ifrlvr", 43)}


def test_parse_metric() -> None:
    assert analyze._parse_metric("ifbench_prompt_loose") == ("ifbench", "prompt_loose")
    assert analyze._parse_metric("dev_prompt_strict") == ("dev", "prompt_strict")
    with pytest.raises(ValueError):
        analyze._parse_metric("nonsense")


def test_matched_seed_pairs_common_seeds() -> None:
    class Fake:
        def __init__(self, seed: int) -> None:
            self.seed = seed

    treatment = [Fake(42), Fake(43), Fake(44)]
    control = [Fake(43)]
    pairs = analyze._matched_seed_pairs(treatment, control)
    assert len(pairs) == 1
    assert pairs[0][0].seed == 43
    assert pairs[0][1].seed == 43


def test_matched_seed_pairs_single_fallback() -> None:
    class Fake:
        def __init__(self, seed: int) -> None:
            self.seed = seed

    pairs = analyze._matched_seed_pairs([Fake(7)], [Fake(9)])
    assert len(pairs) == 1  # no shared seed value, but exactly one run each side


def test_matched_seed_pairs_no_pairs() -> None:
    class Fake:
        def __init__(self, seed: int) -> None:
            self.seed = seed

    assert analyze._matched_seed_pairs([Fake(1), Fake(2)], []) == []


# --- end-to-end ----------------------------------------------------------


def test_analyze_end_to_end_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_dir = _build_synthetic_runs(tmp_path)
    out_path = tmp_path / "report.md"

    monkeypatch.setattr(sys, "argv", ["analyze.py", "--runs", str(runs_dir), "--out", str(out_path)])
    analyze.main()

    assert out_path.exists()
    result = json.loads((tmp_path / "report.md.json").read_text(encoding="utf-8"))

    # (1) per-benchmark table: expected means from the synthetic data
    ext_by_arm = {(r["arm"], r["benchmark"]): r for r in result["external_table"]}
    assert ext_by_arm[("full", "ifbench")]["n_seeds"] == 2
    assert ext_by_arm[("full", "ifbench")]["prompt_loose_mean"] == 1.0
    assert ext_by_arm[("full", "ifbench")]["prompt_loose_std"] == 0.0
    assert ext_by_arm[("ifrlvr", "ifbench")]["prompt_loose_mean"] == 0.0
    assert ("full", "ifeval") not in ext_by_arm  # no ifeval data written

    internal_by_arm = {(r["arm"], r["split"]): r for r in result["internal_table"]}
    assert internal_by_arm[("full", "dev")]["pacc_mean"] == 0.8
    assert internal_by_arm[("ifrlvr", "dev")]["pacc_mean"] == 0.4

    # (2) hypothesis table: H1 = full vs ifrlvr on ifbench_prompt_loose
    hyp_by_id = {r["id"]: r for r in result["hypothesis_table"]}
    h1 = hyp_by_id["H1"]
    assert h1["treatment"] == "full"
    assert h1["control"] == "ifrlvr"
    assert h1["n_seed_pairs"] == 2
    assert h1["mean_diff"] == 1.0
    assert h1["bootstrap_diff"] == 1.0
    assert h1["ci_lo"] == 1.0
    assert h1["ci_hi"] == 1.0
    assert h1["verdict"] == "supports"

    # a hypothesis row naming an arm with no runs at all is reported, not dropped
    unmatched = hyp_by_id["H3_forward"]  # full vs forward_only; forward_only has no runs here
    assert unmatched["n_seed_pairs"] == 0
    assert unmatched["verdict"] == "pending"
    assert unmatched["note"] == "no matched seed pairs"


def test_analyze_extras(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_dir = _build_synthetic_runs(tmp_path)
    out_path = tmp_path / "report.md"

    monkeypatch.setattr(sys, "argv", ["analyze.py", "--runs", str(runs_dir), "--out", str(out_path), "--extras"])
    analyze.main()

    result = json.loads((tmp_path / "report.md.json").read_text(encoding="utf-8"))
    extras = result["extras"]
    assert extras is not None

    wilson_by_arm = {w["arm"]: w for w in extras["wilson_per_arm"]}
    assert wilson_by_arm["full"]["k"] == 20  # 10 keys x 2 seeds, all pass
    assert wilson_by_arm["full"]["n"] == 20
    assert wilson_by_arm["ifrlvr"]["k"] == 0

    mc = extras["mcnemar_headline"]
    assert mc["treatment"] == "full"
    assert mc["control"] == "ifrlvr"
    assert mc["n10"] == 20  # full passes, ifrlvr fails, every pooled prompt
    assert mc["n01"] == 0

    holm_ids = [row["id"] for row in extras["holm_across_hypotheses"]]
    from conpo.arms import HYPOTHESES

    assert holm_ids == [h.id for h in HYPOTHESES]


def test_manifest_loading(tmp_path: Path) -> None:
    runs_dir = _build_synthetic_runs(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([{"arm": "full", "seed": 42, "dir": str(runs_dir / "full_s42")}]), encoding="utf-8"
    )
    refs = analyze.load_manifest(manifest_path)
    assert len(refs) == 1
    assert refs[0].arm == "full"
    assert refs[0].seed == 42
    assert refs[0].dir == runs_dir / "full_s42"

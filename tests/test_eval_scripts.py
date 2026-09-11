"""Tests for scripts/eval_external.py and scripts/eval_internal.py, --score-only paths.

Real generation (--backend hf without --score-only, and --backend vllm) needs
a downloaded policy model or a GPU respectively and is not exercised here;
`generate_greedy` itself is checked directly with conpo.engine.FakeEngine.

scripts/ is not an installed package (pyproject.toml only packages `conpo*`),
so the two scripts are imported by adding scripts/ to sys.path -- the
simplest way to unit-test a standalone CLI script without restructuring it.
"""

import dataclasses
import json
import sys
import types
from pathlib import Path

import pytest

from conpo import config, metrics, scoring
from conpo.engine import FakeEngine

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_external  # noqa: E402
import eval_internal  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# --- generate_greedy (both scripts' copies) ---------------------------------


@pytest.mark.parametrize("module", [eval_external, eval_internal])
def test_generate_greedy_requests_one_greedy_completion_per_prompt(module: types.ModuleType) -> None:
    engine = FakeEngine(lambda messages, n: [f"reply to {messages[-1]['content']}"] * n)
    out = module.generate_greedy(engine, ["hello", "world"], max_tokens=16)
    assert out == ["reply to hello", "reply to world"]
    assert engine.calls == [
        ([{"role": "user", "content": "hello"}], 1),
        ([{"role": "user", "content": "world"}], 1),
    ]


@pytest.mark.parametrize("module", [eval_external, eval_internal])
def test_generate_greedy_empty_prompts(module: types.ModuleType) -> None:
    assert module.generate_greedy(FakeEngine([]), [], max_tokens=16) == []


# --- build_parser / left_truncate_prompt (both scripts' copies) ------------


@pytest.mark.parametrize("module", [eval_external, eval_internal])
def test_max_tokens_defaults_to_eval_max_new_tokens(module: types.ModuleType) -> None:
    args = module.build_parser().parse_args(["--out", "unused.json"])
    assert args.max_tokens == config.EVAL_MAX_NEW_TOKENS


class _CharTokenizer:
    """Character-level fake tokenizer (id == ord(char)): no network fetch, no
    GPU, only the surface (__call__, decode) left_truncate_prompt needs to
    exercise its truncation logic independent of any real tokenizer."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids)


@pytest.mark.parametrize("module", [eval_external, eval_internal])
def test_left_truncate_prompt_over_cap_keeps_last_n_tokens(module: types.ModuleType) -> None:
    truncated, was_truncated = module.left_truncate_prompt(_CharTokenizer(), "abcdefghij", max_tokens=4)
    assert was_truncated is True
    assert truncated == "ghij"  # left truncation: head dropped, tail kept


@pytest.mark.parametrize("module", [eval_external, eval_internal])
def test_left_truncate_prompt_under_cap_is_unchanged(module: types.ModuleType) -> None:
    truncated, was_truncated = module.left_truncate_prompt(_CharTokenizer(), "short", max_tokens=100)
    assert was_truncated is False
    assert truncated == "short"


# --- eval_external.py --score-only ------------------------------------------


def test_eval_external_score_only_ifbench_reproduces_direct_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_rows = scoring.load_ifbench_test()
    sample_rows = scoring.load_sample_output()
    response_by_prompt = {row["prompt"]: row["response"] for row in sample_rows}
    assert all(row["prompt"] in response_by_prompt for row in test_rows)  # every prompt has a canned response

    canned_path = tmp_path / "canned_ifbench.jsonl"
    with canned_path.open("w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps({"key": row["key"], "response": response_by_prompt[row["prompt"]]}) + "\n")

    out_path = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["eval_external.py", "--score-only", str(canned_path), "--out", str(out_path)])
    eval_external.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))

    expected_records = [
        {
            "key": row["key"],
            "prompt": row["prompt"],
            "response": response_by_prompt[row["prompt"]],
            "instruction_id_list": row["instruction_id_list"],
            "kwargs": row["kwargs"],
        }
        for row in test_rows
    ]
    expected = dataclasses.asdict(scoring.score_ifbench(expected_records))
    assert result["ifbench"] == expected
    assert result["ifeval"] is None  # no IFBench key coincides with an IFEval key
    assert result["decoding"] == {"temperature": 0, "max_tokens": result["decoding"]["max_tokens"]}


def test_eval_external_score_only_ifeval_minimal_record_real_dataset_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises the _clean_kwargs codepath that the hand-written-record test
    above skips: a minimal {key, response} canned record joined (route_canned
    / _merge) against a real loaded google/IFEval row. The raw row's kwargs
    carry the union-schema's None-valued keys, which a per-instruction
    checker's build_description rejects as unrecognised unless cleaned first
    -- this is the one scenario _clean_kwargs exists for."""
    reference_row = eval_external._load_ifeval_rows()[0]
    response = "A response."

    canned_path = tmp_path / "canned_minimal.jsonl"
    canned_path.write_text(
        json.dumps({"key": reference_row["key"], "response": response}) + "\n", encoding="utf-8"
    )

    out_path = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["eval_external.py", "--score-only", str(canned_path), "--out", str(out_path)])
    eval_external.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))

    cleaned_record = eval_external._merge(reference_row, response)
    expected = dataclasses.asdict(scoring.score_ifeval([cleaned_record]))
    assert result["ifeval"] == expected
    assert result["ifbench"] is None


def test_route_canned_unmatched_key_raises_keyerror() -> None:
    """A minimal {key, response} record whose key matches neither dataset
    fails loudly instead of silently shrinking the scored set."""
    ifeval_rows = [
        {"key": 1, "prompt": "p1", "instruction_id_list": ["keywords:existence"], "kwargs": [{"keywords": ["a"]}]}
    ]
    ifbench_rows = [{"key": "0", "prompt": "p2", "instruction_id_list": ["length_constraints:number_words"], "kwargs": [{}]}]
    canned = [{"key": "no-such-key", "response": "irrelevant"}]
    with pytest.raises(KeyError):
        eval_external.route_canned(canned, ifeval_rows, ifbench_rows)


def test_eval_external_join_by_prompt_reproduces_ifbench_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The vendored sample_output.jsonl carries only prompt/response, no key
    -- --join-by-prompt joins it to IFBench_test by exact prompt text."""
    sample_rows = scoring.load_sample_output()
    ifbench_rows = scoring.load_ifbench_test()
    response_by_prompt = {row["prompt"]: row["response"] for row in sample_rows}

    canned_path = tmp_path / "sample_output.jsonl"
    with canned_path.open("w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps({"prompt": row["prompt"], "response": row["response"]}) + "\n")

    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_external.py", "--score-only", str(canned_path), "--join-by-prompt", "--out", str(out_path)],
    )
    eval_external.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))

    expected_records = [eval_external._merge(row, response_by_prompt[row["prompt"]]) for row in ifbench_rows]
    expected = dataclasses.asdict(scoring.score_ifbench(expected_records))
    assert result["ifbench"] == expected
    assert result["ifeval"] is None


@pytest.mark.parametrize("module", [eval_external, eval_internal])
def test_build_vllm_engine_memoises_per_model_adapter(
    module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_generate_vllm's engine construction is cached per (model, adapter)
    within a process instead of rebuilt on every call."""
    module._build_vllm_engine.cache_clear()
    calls: list[tuple[str, bool]] = []

    class FakeLLM:
        def __init__(self, model: str, enable_lora: bool, max_lora_rank: int) -> None:
            calls.append((model, enable_lora, max_lora_rank))

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    first = module._build_vllm_engine("model-a", None)
    second = module._build_vllm_engine("model-a", None)  # same key -> cached, no new FakeLLM
    third = module._build_vllm_engine("model-a", "adapter-dir")  # different key -> new FakeLLM

    assert first is second
    assert first is not third
    assert calls == [("model-a", False, 64), ("model-a", True, 64)]


def test_eval_external_score_only_ifeval_hand_written_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {
            "key": 0,
            "prompt": "p1",
            "response": "The quick brown fox jumps. Wonderful day.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["quick", "fox"]}],
        },
        {
            "key": 1,
            "prompt": "p2",
            "response": "Nothing relevant here.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["giraffe"]}],
        },
        {
            "key": 2,
            "prompt": "p3",
            "response": "Some text.\nBest regards",
            "instruction_id_list": ["keywords:existence", "startend:end_checker"],
            "kwargs": [{"keywords": ["text"]}, {"end_phrase": "Some text."}],
        },
    ]
    canned_path = tmp_path / "canned_ifeval.jsonl"
    with canned_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    out_path = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["eval_external.py", "--score-only", str(canned_path), "--out", str(out_path)])
    eval_external.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))

    expected = dataclasses.asdict(scoring.score_ifeval(records))
    assert result["ifeval"] == expected
    assert result["ifbench"] is None  # all 3 records use classic IFEval ids


# --- eval_internal.py --score-only ------------------------------------------


def test_eval_internal_score_only_five_dev_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = eval_internal.load_split("dev", limit=5)
    assert len(rows) == 5
    responses = [f"canned response number {i}" for i in range(5)]

    canned_path = tmp_path / "canned_dev.jsonl"
    with canned_path.open("w", encoding="utf-8") as f:
        for row, response in zip(rows, responses):
            f.write(json.dumps({"key": row.key, "response": response}) + "\n")

    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_internal.py",
            "--splits",
            "dev",
            "--limit",
            "5",
            "--score-only",
            str(canned_path),
            "--out",
            str(out_path),
        ],
    )
    eval_internal.main()
    result = json.loads(out_path.read_text(encoding="utf-8"))

    assert list(result.keys()) == ["dev"]
    # round-trip `expected` through JSON too: json.dumps stringifies dict
    # keys (stratified_by_count's int keys), so compare like for like.
    expected = json.loads(json.dumps(metrics.summarise(rows, responses, "dev")))
    assert result["dev"] == expected
    for key in (
        "split",
        "num_prompts",
        "pacc",
        "iacc",
        "family_accuracies",
        "macro_family_accuracy",
        "worst_decile_family_accuracy",
        "stratified_by_count",
        "mean_length",
        "repetition_rate",
        "task_relevance",
    ):
        assert key in result["dev"]


# --- .verdicts.jsonl (both scripts) -----------------------------------------


def test_eval_external_writes_verdicts_jsonl_matching_ifbench_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_rows = scoring.load_ifbench_test()
    sample_rows = scoring.load_sample_output()
    response_by_prompt = {row["prompt"]: row["response"] for row in sample_rows}

    canned_path = tmp_path / "canned_ifbench.jsonl"
    with canned_path.open("w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps({"key": row["key"], "response": response_by_prompt[row["prompt"]]}) + "\n")

    out_path = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["eval_external.py", "--score-only", str(canned_path), "--out", str(out_path)])
    eval_external.main()

    verdicts_path = tmp_path / "out.json.verdicts.jsonl"
    assert verdicts_path.exists()
    lines = [json.loads(line) for line in verdicts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == len(test_rows)

    result = json.loads(out_path.read_text(encoding="utf-8"))
    # prompt_loose fraction over the verdicts must reproduce the aggregate score
    assert sum(1 for v in lines if v["prompt_loose"]) / len(lines) == result["ifbench"]["prompt_loose"]
    assert sum(1 for v in lines if v["prompt_strict"]) / len(lines) == result["ifbench"]["prompt_strict"]
    for v in lines:
        assert v["benchmark"] == "ifbench"
        assert len(v["instruction_strict"]) == len(v["instruction_id_list"])
        assert len(v["instruction_loose"]) == len(v["instruction_id_list"])


def test_eval_external_record_verdicts_hand_written_records() -> None:
    records = [
        {
            "key": 2,
            "prompt": "p3",
            "response": "Some text.\nBest regards",
            "instruction_id_list": ["keywords:existence", "startend:end_checker"],
            "kwargs": [{"keywords": ["text"]}, {"end_phrase": "Some text."}],
        }
    ]
    verdicts = eval_external.record_verdicts(records, "ifeval")
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v["key"] == 2
    assert v["benchmark"] == "ifeval"
    assert v["instruction_id_list"] == ["keywords:existence", "startend:end_checker"]
    assert isinstance(v["prompt_strict"], bool)
    assert isinstance(v["prompt_loose"], bool)
    assert len(v["instruction_strict"]) == 2
    # "startend:end_checker" fails: response does not end with "Some text."
    assert v["instruction_strict"][1] is False


def test_eval_internal_writes_verdicts_jsonl_matching_pacc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = eval_internal.load_split("dev", limit=5)
    responses = [f"canned response number {i}" for i in range(5)]

    canned_path = tmp_path / "canned_dev.jsonl"
    with canned_path.open("w", encoding="utf-8") as f:
        for row, response in zip(rows, responses):
            f.write(json.dumps({"key": row.key, "response": response}) + "\n")

    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_internal.py", "--splits", "dev", "--limit", "5", "--score-only", str(canned_path), "--out", str(out_path)],
    )
    eval_internal.main()

    verdicts_path = tmp_path / "out.json.verdicts.jsonl"
    assert verdicts_path.exists()
    lines = [json.loads(line) for line in verdicts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 5

    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert sum(1 for v in lines if v["prompt_strict"]) / len(lines) == result["dev"]["pacc"]
    for v in lines:
        assert v["split"] == "dev"
        assert v["prompt_strict"] == v["prompt_loose"]  # no strict/loose distinction internally
        assert v["instruction_strict"] == v["instruction_loose"]


def test_eval_internal_record_verdicts_matches_verifier_vector() -> None:
    rows = eval_internal.load_split("dev", limit=1)
    responses = ["a plausible response"]
    verdicts = eval_internal.record_verdicts(rows, responses, "dev")
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v["key"] == rows[0].key
    assert v["split"] == "dev"
    assert v["instruction_id_list"] == [c.iid for c in rows[0].constraints]
    from conpo.verifiers import verifier_vector

    expected = [bool(x) for x in verifier_vector(rows[0].constraints, responses[0])]
    assert v["instruction_strict"] == expected
    assert v["prompt_strict"] == all(expected)


def test_eval_internal_score_only_missing_key_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = eval_internal.load_split("dev", limit=5)
    canned_path = tmp_path / "canned_dev_short.jsonl"
    with canned_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"key": rows[0].key, "response": "only one response"}) + "\n")

    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_internal.py",
            "--splits",
            "dev",
            "--limit",
            "5",
            "--score-only",
            str(canned_path),
            "--out",
            str(out_path),
        ],
    )
    with pytest.raises(KeyError):
        eval_internal.main()

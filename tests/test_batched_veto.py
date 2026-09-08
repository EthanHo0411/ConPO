"""Tests for the batched-veto judge side (conpo/judge.py): PendingQueue,
HFBatchedJudge on the tiny random model fixture, build_judge's selection
matrix, and VLLMBatchedJudge's lazy vllm import.

Trainer-side flush behaviour (PendingQueue wired into ConPOTrainer, the
mechanism JSONL burst fields, edge age at construction step) is out of
scope here; it belongs to tests/test_trainer_tiny.py.
"""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from conpo import config, judge
from conpo.edges import Edge

# ---------------------------------------------------------------------------
# PendingPair / PendingQueue
# ---------------------------------------------------------------------------


def _edge(prompt_id: str = "row-a", direction: str = "reverse") -> Edge:
    return Edge(
        prompt_id=prompt_id,
        prompt="Explain how gravity works.",
        y_plus="Gravity pulls things down.",
        y_minus="Gravity.",
        families=frozenset({"keywords:existence"}),
        direction=direction,
        step=0,
        logp_ref_plus=float("nan"),
        logp_ref_minus=float("nan"),
        len_plus=0,
        len_minus=0,
        created_update=0,
    )


def _req() -> judge.VetoRequest:
    return judge.VetoRequest(
        prompt="Explain how gravity works.",
        y_plus="Gravity pulls things down.",
        y_minus="Gravity.",
        families=("keywords:existence",),
        descriptions=("Include the word 'apple'.",),
    )


def _pair(direction: str = "reverse") -> judge.PendingPair:
    return judge.PendingPair(edge=_edge(direction=direction), request=_req(), direction=direction)


def test_pending_pair_rejects_bad_direction() -> None:
    with pytest.raises(ValueError):
        judge.PendingPair(edge=_edge(), request=_req(), direction="sideways")


def test_pending_queue_starts_empty() -> None:
    q = judge.PendingQueue(max_size=10)
    assert len(q) == 0
    assert q.drain() == []


def test_pending_queue_push_and_len() -> None:
    q = judge.PendingQueue(max_size=10)
    q.push(_pair())
    q.push(_pair())
    assert len(q) == 2


def test_pending_queue_drain_empties_and_returns_pushed_pairs() -> None:
    q = judge.PendingQueue(max_size=10)
    p1, p2 = _pair(), _pair()
    q.push(p1)
    q.push(p2)
    drained = q.drain()
    assert drained == [p1, p2]
    assert len(q) == 0
    assert q.drain() == []


def test_pending_queue_max_size_raises() -> None:
    q = judge.PendingQueue(max_size=1)
    q.push(_pair())
    with pytest.raises(ValueError):
        q.push(_pair())


def test_pending_queue_push_after_drain_is_not_blocked_by_a_stale_full_count() -> None:
    q = judge.PendingQueue(max_size=1)
    q.push(_pair())
    q.drain()
    q.push(_pair())  # would raise if drain() left a stale size behind
    assert len(q) == 1


# ---------------------------------------------------------------------------
# Judge protocol: veto is veto_many([req])[0] on every judge in this module
# ---------------------------------------------------------------------------


def test_static_judge_satisfies_veto_via_veto_many() -> None:
    j = judge.StaticJudge(keep=False)
    assert j.veto(_req()) == j.veto_many([_req()])[0]


# ---------------------------------------------------------------------------
# HFBatchedJudge, on the tiny random model fixture (test_trainer_tiny.py's
# shape: real tokenizer, random tiny Qwen3 config, saved to a local
# directory so HFBatchedJudge can load it by path like any model name).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
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
    out_dir = tmp_path_factory.mktemp("tiny_judge_model")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def test_hf_batched_judge_returns_one_verdict_per_request(tiny_model_dir: Path) -> None:
    hfj = judge.HFBatchedJudge(str(tiny_model_dir), device="cpu", max_new_tokens=16, batch_size=2)
    reqs = [_req(), _req(), _req()]
    verdicts = hfj.veto_many(reqs)
    assert len(verdicts) == len(reqs)
    assert all(isinstance(v, judge.Verdict) for v in verdicts)


def test_hf_batched_judge_fails_closed_on_garbage_and_counts_it(tiny_model_dir: Path) -> None:
    # An untrained tiny random model cannot land on the exact
    # {"degenerate": ...} JSON schema in 16 tokens: every reply here fails
    # to parse and must reject (keep=False), never raise.
    hfj = judge.HFBatchedJudge(str(tiny_model_dir), device="cpu", max_new_tokens=16, batch_size=2)
    verdicts = hfj.veto_many([_req(), _req()])
    assert all(v.keep is False for v in verdicts)
    assert all("fail-closed" in v.reason for v in verdicts)
    assert all(v.parse_ok is False for v in verdicts)
    assert all(len(v.raw) <= 500 for v in verdicts)
    assert hfj.counts["discard"] == 2


def test_hf_batched_judge_veto_matches_veto_many(tiny_model_dir: Path) -> None:
    hfj = judge.HFBatchedJudge(str(tiny_model_dir), device="cpu", max_new_tokens=16, batch_size=2)
    v = hfj.veto(_req())
    assert isinstance(v, judge.Verdict)


def test_hf_batched_judge_batches_more_requests_than_batch_size(tiny_model_dir: Path) -> None:
    hfj = judge.HFBatchedJudge(str(tiny_model_dir), device="cpu", max_new_tokens=8, batch_size=2)
    verdicts = hfj.veto_many([_req() for _ in range(5)])
    assert len(verdicts) == 5


def test_hf_batched_judge_veto_many_of_empty_is_empty(tiny_model_dir: Path) -> None:
    hfj = judge.HFBatchedJudge(str(tiny_model_dir), device="cpu")
    assert hfj.veto_many([]) == []


# ---------------------------------------------------------------------------
# VLLMBatchedJudge: construction must not import vllm; that only happens
# inside veto_many. vllm is not installed in this CPU test environment, so
# this also doubles as the "runs without vllm present" check.
# ---------------------------------------------------------------------------


def _block_vllm_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any `import vllm` raise, whether or not vllm is installed or already
    loaded. On a GPU box vllm sits in sys.modules as soon as TRL's GRPO trainer
    is imported by an earlier test, so `"vllm" not in sys.modules` cannot prove
    that construction is lazy; a blocked import can."""
    real_import = builtins.__import__

    def _blocking_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "vllm" or name.startswith("vllm."):
            raise ModuleNotFoundError(f"No module named {name!r} (blocked by test)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)


def test_vllm_batched_judge_construction_does_not_import_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_vllm_imports(monkeypatch)
    j = judge.VLLMBatchedJudge("some/judge-model")
    assert j.model_name == "some/judge-model"


def test_vllm_batched_judge_veto_many_imports_vllm_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """The only import of vllm on veto_many's path happens inside _vllm_burst,
    the function the spawned child runs. A real child starts a fresh
    interpreter that a blocked builtins.__import__ here cannot reach, so this
    calls _vllm_burst directly under the block: it must fail on that import
    (ModuleNotFoundError), never earlier and never with another error."""
    _block_vllm_imports(monkeypatch)
    with pytest.raises(ModuleNotFoundError):
        judge._vllm_burst("some/judge-model", 4096, 0.6, None, [judge.build_veto_prompt(_req())])


def test_vllm_batched_judge_veto_many_of_empty_never_touches_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty batch returns [] without importing vllm at all, so a flush
    with nothing pending never pays vllm's load cost."""
    _block_vllm_imports(monkeypatch)
    j = judge.VLLMBatchedJudge("some/judge-model")
    assert j.veto_many([]) == []


class _FakeLLMOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [type("_Choice", (), {"text": text})()]


class _FakeVLLM_LLM:
    """Stands in for vllm.LLM: records its own construction kwargs and the
    SamplingParams it is called with, and answers every chat() with a
    always-keep verdict."""

    last_sampling_params: object | None = None

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def chat(self, messages_batch, sampling_params, chat_template_kwargs=None):  # type: ignore[no-untyped-def]
        _FakeVLLM_LLM.last_sampling_params = sampling_params
        return [_FakeLLMOutput('{"degenerate": false, "reason": "fine"}') for _ in messages_batch]


class _FakeSamplingParams:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _FakeStructuredOutputsParams:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_vllm_burst_uses_structured_outputs_not_guided_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review fix, retained under the new seam: the installed trl==1.9.2
    package's own vLLM integration uses
    SamplingParams(structured_outputs=StructuredOutputsParams(...)), never
    guided_decoding/GuidedDecodingParams -- _vllm_burst must match that API
    or the first real veto burst fails outright. _vllm_burst is what a
    spawned child actually runs, so this exercises it directly, in-process,
    with no real subprocess involved."""
    fake_sampling_params_mod = types.ModuleType("vllm.sampling_params")
    fake_sampling_params_mod.StructuredOutputsParams = _FakeStructuredOutputsParams  # type: ignore[attr-defined]
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _FakeVLLM_LLM  # type: ignore[attr-defined]
    fake_vllm.SamplingParams = _FakeSamplingParams  # type: ignore[attr-defined]
    fake_vllm.sampling_params = fake_sampling_params_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling_params_mod)

    raw_replies = judge._vllm_burst("some/judge-model", 4096, 0.35, None, [judge.build_veto_prompt(_req())])

    assert raw_replies == ['{"degenerate": false, "reason": "fine"}']
    sampling = _FakeVLLM_LLM.last_sampling_params
    assert "structured_outputs" in sampling.kwargs
    assert isinstance(sampling.kwargs["structured_outputs"], _FakeStructuredOutputsParams)
    assert "guided_decoding" not in sampling.kwargs


def test_vllm_batched_judge_fails_closed_on_garbage_and_sets_parse_ok_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed parsing is veto_many's own logic, applied to whatever
    raw text _run_burst returns; a fake standing in for that seam is
    enough, no vllm or subprocess involved."""
    j = judge.VLLMBatchedJudge("some/judge-model")
    monkeypatch.setattr(j, "_run_burst", lambda prompts: ["not json at all" for _ in prompts])

    verdicts = j.veto_many([_req()])

    assert len(verdicts) == 1
    assert verdicts[0].keep is False
    assert verdicts[0].parse_ok is False
    assert "fail-closed" in verdicts[0].reason
    assert j.counts["discard"] == 1


def test_vllm_batched_judge_calls_after_unload_when_run_burst_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review fix, retained under the new seam: before_load()/after_unload()
    must bracket _run_burst() itself, not only the code that runs after it
    succeeds -- otherwise a burst failure (e.g. the child's LLM()
    constructor hitting GPU memory still held by a not-yet-fully-freed
    colocated training engine) leaves whatever before_load() did (e.g.
    sleeping the training engine) permanently unreversed."""
    calls: list[str] = []
    j = judge.VLLMBatchedJudge(
        "some/judge-model",
        before_load=lambda: calls.append("before_load"),
        after_unload=lambda: calls.append("after_unload"),
    )

    def _raising_run_burst(prompts: list[str]) -> list[str]:
        raise RuntimeError("out of memory (simulated)")

    monkeypatch.setattr(j, "_run_burst", _raising_run_burst)

    with pytest.raises(RuntimeError, match="out of memory"):
        j.veto_many([_req()])

    assert calls == ["before_load", "after_unload"]


class _FakeConn:
    """One end of the parent/child pipe as _run_burst sees it."""

    def __init__(self, poll_result: bool, message: object = None) -> None:
        self.poll_result, self.message, self.closed = poll_result, message, False

    def poll(self, timeout: float | None = None) -> bool:
        return self.poll_result

    def recv(self) -> object:
        if self.message is None:
            raise EOFError  # the child died before sending anything
        return self.message

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    """Stands in for the spawned child: never runs the target."""

    exitcode: int | None = -9

    def __init__(self, target: object, args: tuple[object, ...]) -> None:
        self.args, self.killed, self.joins = args, False, 0

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        self.joins += 1

    def is_alive(self) -> bool:
        return not self.killed

    def kill(self) -> None:
        self.killed = True


def _fake_spawn(monkeypatch: pytest.MonkeyPatch, parent: _FakeConn) -> list[_FakeProcess]:
    """Route _run_burst's get_context("spawn") to fakes; returns the processes it creates."""
    procs: list[_FakeProcess] = []

    class _Ctx:
        def Pipe(self, duplex: bool = True) -> tuple[_FakeConn, _FakeConn]:  # noqa: N802
            return parent, _FakeConn(False)

        def Process(self, target: object, args: tuple[object, ...]) -> _FakeProcess:  # noqa: N802
            procs.append(_FakeProcess(target, args))
            return procs[-1]

    monkeypatch.setattr(judge.multiprocessing, "get_context", lambda kind: _Ctx())
    return procs


def _judge_with_hooks(calls: list[str], **kwargs: object) -> judge.VLLMBatchedJudge:
    return judge.VLLMBatchedJudge(
        "some/judge-model",
        before_load=lambda: calls.append("before_load"),
        after_unload=lambda: calls.append("after_unload"),
        **kwargs,  # type: ignore[arg-type]
    )


def test_vllm_batched_judge_burst_timeout_raises_kills_child_and_still_unloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """A child wedged in the driver never reports; the burst must fail loud
    after burst_timeout_s, SIGKILL the child after a bounded join, and still
    run after_unload."""
    calls: list[str] = []
    procs = _fake_spawn(monkeypatch, _FakeConn(poll_result=False))
    j = _judge_with_hooks(calls, burst_timeout_s=0.01)
    with pytest.raises(RuntimeError, match="returned no result"):
        j.veto_many([_req()])
    assert calls == ["before_load", "after_unload"]
    assert procs[0].killed and procs[0].joins == 2


def test_vllm_batched_judge_child_death_raises_and_still_unloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """A child killed by the OOM killer closes the pipe without a message
    (EOFError on recv): a clear RuntimeError, hooks still bracketed."""
    calls: list[str] = []
    _fake_spawn(monkeypatch, _FakeConn(poll_result=True, message=None))
    j = _judge_with_hooks(calls)
    with pytest.raises(RuntimeError, match="exited without a result"):
        j.veto_many([_req()])
    assert calls == ["before_load", "after_unload"]


def test_vllm_batched_judge_child_failure_reraises_with_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _fake_spawn(monkeypatch, _FakeConn(poll_result=True, message=(False, "Traceback: simulated OOM")))
    j = _judge_with_hooks(calls)
    with pytest.raises(RuntimeError, match="simulated OOM"):
        j.veto_many([_req()])
    assert calls == ["before_load", "after_unload"]


def test_vllm_batched_judge_child_result_parses_into_verdicts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    procs = _fake_spawn(monkeypatch, _FakeConn(poll_result=True, message=(True, ['{"degenerate": false, "reason": "fine"}'])))
    j = _judge_with_hooks(calls)
    verdicts = j.veto_many([_req()])
    assert [v.keep for v in verdicts] == [True] and verdicts[0].parse_ok
    assert procs[0].args[1] == "some/judge-model" and procs[0].args[-1][0].startswith(judge.build_veto_prompt(_req())[:20])
    assert calls == ["before_load", "after_unload"]


def test_vllm_batched_judge_veto_many_uses_run_burst_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """The seam veto_many calls to get raw replies is _run_burst, never
    vllm or multiprocessing directly: a fake standing in for it proves
    veto_many's own contract -- before_load then after_unload bracket the
    burst, replies parse into Verdicts in order, and counts update."""
    calls: list[str] = []
    j = judge.VLLMBatchedJudge(
        "some/judge-model",
        before_load=lambda: calls.append("before_load"),
        after_unload=lambda: calls.append("after_unload"),
    )
    canned = ['{"degenerate": false, "reason": "fine"}', '{"degenerate": true, "reason": "bad"}']

    def _fake_run_burst(prompts: list[str]) -> list[str]:
        assert len(prompts) == 2
        calls.append("run_burst")
        return canned

    monkeypatch.setattr(j, "_run_burst", _fake_run_burst)

    verdicts = j.veto_many([_req(), _req()])

    assert calls == ["before_load", "run_burst", "after_unload"]
    assert [v.keep for v in verdicts] == [True, False]
    assert all(v.parse_ok for v in verdicts)
    assert j.counts["keep"] == 1
    assert j.counts["discard"] == 1


# ---------------------------------------------------------------------------
# build_judge selection matrix
# ---------------------------------------------------------------------------


def _clear_judge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RUN_NO_VETO", "RUN_JUDGE_MODE", "RUN_JUDGE_ENDPOINT", "RUN_USE_VLLM", "RUN_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_build_judge_no_veto_wins_over_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RUN_NO_VETO", "1")
    monkeypatch.setenv("RUN_JUDGE_MODE", "http")
    j = judge.build_judge()
    assert isinstance(j, judge.StaticJudge)
    assert j.keep is True


def test_build_judge_http_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RUN_JUDGE_MODE", "http")
    monkeypatch.setenv("RUN_JUDGE_ENDPOINT", "http://judge-host:1234")
    j = judge.build_judge()
    assert isinstance(j, judge.HttpJudge)
    assert j.base_url == "http://judge-host:1234"
    assert j.model == config.JUDGE_MODEL


def test_build_judge_http_mode_with_model_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RUN_JUDGE_MODE", "http")
    monkeypatch.setenv("RUN_JUDGE_ENDPOINT", "http://judge-host:1234")
    monkeypatch.setenv("RUN_JUDGE_MODEL", "some/other-judge")
    j = judge.build_judge()
    assert isinstance(j, judge.HttpJudge)
    assert j.model == "some/other-judge"


def test_build_judge_batched_mode_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_judge_env(monkeypatch)
    j = judge.build_judge()
    assert isinstance(j, judge.HFBatchedJudge)


def test_build_judge_batched_mode_explicit_uses_hf_without_vllm_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RUN_JUDGE_MODE", "batched")
    monkeypatch.setenv("RUN_USE_VLLM", "0")
    j = judge.build_judge()
    assert isinstance(j, judge.HFBatchedJudge)


def test_build_judge_batched_mode_with_vllm_flag_selects_vllm_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RUN_JUDGE_MODE", "batched")
    monkeypatch.setenv("RUN_USE_VLLM", "1")
    _block_vllm_imports(monkeypatch)  # construction alone must not import it
    j = judge.build_judge()
    assert isinstance(j, judge.VLLMBatchedJudge)


def test_build_judge_model_override_applies_to_batched_judges(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RUN_JUDGE_MODEL", "some/other-judge")
    j = judge.build_judge()
    assert isinstance(j, judge.HFBatchedJudge)
    assert j.model_name == "some/other-judge"

    monkeypatch.setenv("RUN_USE_VLLM", "1")
    j2 = judge.build_judge()
    assert isinstance(j2, judge.VLLMBatchedJudge)
    assert j2.model_name == "some/other-judge"


def test_build_judge_default_model_is_config_judge_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_judge_env(monkeypatch)
    j = judge.build_judge()
    assert j.model_name == config.JUDGE_MODEL


def test_build_judge_unknown_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RUN_JUDGE_MODE", "carrier-pigeon")
    with pytest.raises(ValueError):
        judge.build_judge()


def test_build_judge_passes_device_and_dtype_to_hf_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_judge_env(monkeypatch)
    j = judge.build_judge(device="cuda:0", dtype=torch.bfloat16)
    assert isinstance(j, judge.HFBatchedJudge)
    assert j.device == "cuda:0"
    assert j.dtype is torch.bfloat16


def test_build_judge_passes_hooks_to_vllm_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("RUN_USE_VLLM", "1")
    before = lambda: None  # noqa: E731
    after = lambda: None  # noqa: E731
    j = judge.build_judge(before_load=before, after_unload=after)
    assert isinstance(j, judge.VLLMBatchedJudge)
    assert j.before_load is before
    assert j.after_unload is after

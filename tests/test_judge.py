"""Tests for conpo.judge: verdict parsing, prompt construction, judges, transport."""

from __future__ import annotations

import contextlib
import http.server
import json
import threading
from pathlib import Path

import pytest

from conpo import judge

# ---------------------------------------------------------------------------
# parse_verdict
# ---------------------------------------------------------------------------


def test_parse_verdict_clean_json() -> None:
    v = judge.parse_verdict('{"degenerate": false, "reason": "fine"}')
    assert v.keep is True
    assert v.reason == "fine"


def test_parse_verdict_json_in_prose() -> None:
    raw = 'Sure, here is my answer.\n{"degenerate": true, "reason": "truncated"}\nDone.'
    v = judge.parse_verdict(raw)
    assert v.keep is False
    assert v.reason == "truncated"


def test_parse_verdict_after_think_block() -> None:
    raw = "<think>\nlet me consider this carefully\n</think>\n" '{"degenerate": false, "reason": "ok"}'
    v = judge.parse_verdict(raw)
    assert v.keep is True
    assert v.reason == "ok"


def test_parse_verdict_garbage_raises() -> None:
    with pytest.raises(ValueError):
        judge.parse_verdict("not json at all, sorry")


def test_parse_verdict_missing_key_raises() -> None:
    with pytest.raises(ValueError):
        judge.parse_verdict('{"reason": "no degenerate key here"}')


def test_parse_verdict_sets_parse_ok_true() -> None:
    v = judge.parse_verdict('{"degenerate": false, "reason": "fine"}')
    assert v.parse_ok is True


def test_parse_verdict_truncates_raw_to_500_chars() -> None:
    long_raw = '{"degenerate": false, "reason": "fine"}' + "x" * 600
    v = judge.parse_verdict(long_raw)
    assert len(v.raw) == 500
    assert v.raw == long_raw[:500]


# ---------------------------------------------------------------------------
# build_veto_prompt
# ---------------------------------------------------------------------------


def _req(prompt: str = "Write a short poem about rivers.") -> judge.VetoRequest:
    return judge.VetoRequest(
        prompt=prompt,
        y_plus="Rivers flow, apple trees grow, three lines just so.",
        y_minus="Rivers flow.",
        families=("keywords:existence", "length_constraints:number_sentences"),
        descriptions=(
            "Include keywords apple and river in your response.",
            "Your response must have at least 3 sentences.",
        ),
    )


def _label_a_text(prompt_text: str) -> str:
    start = prompt_text.index("Response A:\n") + len("Response A:\n")
    end = prompt_text.index("\n\nResponse B:")
    return prompt_text[start:end]


def test_build_veto_prompt_contains_responses_and_descriptions() -> None:
    req = _req()
    text = judge.build_veto_prompt(req)
    assert req.y_plus in text
    assert req.y_minus in text
    for d in req.descriptions:
        assert d in text


def test_build_veto_prompt_is_deterministic_for_the_same_request() -> None:
    req = _req()
    assert judge.build_veto_prompt(req) == judge.build_veto_prompt(req)


def test_build_veto_prompt_randomizes_label_assignment() -> None:
    seen: set[bool] = set()
    for i in range(30):
        req = judge.VetoRequest(
            prompt=f"prompt {i}",
            y_plus="PLUS_TEXT",
            y_minus="MINUS_TEXT",
            families=("keywords:existence",),
            descriptions=("desc",),
        )
        seen.add(_label_a_text(judge.build_veto_prompt(req)) == "PLUS_TEXT")
    assert seen == {True, False}


def test_build_veto_prompt_names_the_minus_label() -> None:
    req = _req()
    text = judge.build_veto_prompt(req)
    label = "A" if _label_a_text(text) == req.y_minus else "B"
    assert f"whether response {label} is" in text


# ---------------------------------------------------------------------------
# StaticJudge
# ---------------------------------------------------------------------------


def test_static_judge_keep_true() -> None:
    j = judge.StaticJudge(keep=True)
    v = j.veto(_req())
    assert v.keep is True
    assert v.parse_ok is True
    assert j.counts["keep"] == 1
    assert j.counts["discard"] == 0


def test_static_judge_keep_false_and_veto_many() -> None:
    j = judge.StaticJudge(keep=False)
    verdicts = j.veto_many([_req(), _req()])
    assert all(v.keep is False for v in verdicts)
    assert j.counts["discard"] == 2
    assert j.counts["keep"] == 0


# ---------------------------------------------------------------------------
# read_endpoint / resolve_endpoint
# ---------------------------------------------------------------------------


def test_read_endpoint_parses_well_formed_file(tmp_path: Path) -> None:
    ep = tmp_path / "judge.endpoint"
    ep.write_text("gpu-node.example.org 24451 999\n")
    assert judge.read_endpoint(ep) == "http://gpu-node.example.org:24451"


def test_read_endpoint_rejects_torn_file(tmp_path: Path) -> None:
    ep = tmp_path / "judge.endpoint"
    ep.write_text("garbage\n")
    with pytest.raises(ValueError):
        judge.read_endpoint(ep)


def test_resolve_endpoint_prefers_env_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_JUDGE_ENDPOINT", "http://envhost:1234")
    ep = tmp_path / "judge.endpoint"
    ep.write_text("filehost 555 1\n")
    assert judge.resolve_endpoint(ep) == "http://envhost:1234"


def test_resolve_endpoint_falls_back_to_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_JUDGE_ENDPOINT", raising=False)
    ep = tmp_path / "judge.endpoint"
    ep.write_text("filehost 555 1\n")
    assert judge.resolve_endpoint(ep) == "http://filehost:555"


# ---------------------------------------------------------------------------
# HttpJudge end to end against an in-process stub server
# ---------------------------------------------------------------------------


class _CannedHandler(http.server.BaseHTTPRequestHandler):
    """Returns a canned chat-completions body; content/status set per-subclass."""

    RESPONSE_CONTENT = ""
    STATUS = 200

    def log_message(self, format: str, *args: object) -> None:  # silence test output
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.STATUS != 200:
            self.send_response(self.STATUS)
            self.end_headers()
            self.wfile.write(b"internal error")
            return
        body = json.dumps({"choices": [{"message": {"content": self.RESPONSE_CONTENT}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def _stub_server(content: str, status: int = 200):
    handler_cls = type("Handler", (_CannedHandler,), {"RESPONSE_CONTENT": content, "STATUS": status})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_judge_end_to_end_keep() -> None:
    with _stub_server('{"degenerate": false, "reason": "looks fine"}') as base:
        j = judge.HttpJudge(base)
        v = j.veto(_req())
        assert v.keep is True
        assert v.reason == "looks fine"
        assert v.parse_ok is True
        assert j.counts["keep"] == 1


def test_http_judge_end_to_end_discard() -> None:
    with _stub_server('{"degenerate": true, "reason": "truncated mid-sentence"}') as base:
        j = judge.HttpJudge(base)
        v = j.veto(_req())
        assert v.keep is False
        assert j.counts["discard"] == 1


def test_http_judge_veto_many_sequential() -> None:
    with _stub_server('{"degenerate": false, "reason": "fine"}') as base:
        j = judge.HttpJudge(base)
        verdicts = j.veto_many([_req("a"), _req("b"), _req("c")])
        assert len(verdicts) == 3
        assert all(v.keep for v in verdicts)
        assert j.counts["keep"] == 3


def test_http_judge_fails_closed_on_unparseable_reply() -> None:
    """A reply that arrives (200 OK) but never parses into the veto JSON --
    the common failure with a small judge model -- must reject rather than
    raise, with parse_ok=False marking it apart from a real rejection."""
    with _stub_server("not valid json at all") as base:
        j = judge.HttpJudge(base)
        v = j.veto(_req())
        assert v.keep is False
        assert v.parse_ok is False
        assert j.counts["discard"] == 1
        assert j.counts["keep"] == 0


def test_http_judge_fail_loud_on_server_error() -> None:
    """A 500 response raises; no verdict is recorded as a silent keep."""
    with _stub_server("", status=500) as base:
        j = judge.HttpJudge(base)
        with pytest.raises(Exception):
            j.veto(_req())
        assert j.counts["keep"] == 0
        assert j.counts["discard"] == 0


def test_fit_gpu_fraction_caps_at_free_memory_minus_margin() -> None:
    from conpo.judge import fit_gpu_fraction

    gib = 1 << 30
    # 39.4 GiB free of 79.2 (the 2026-09-11 end-of-training flush): 0.6 does not fit.
    assert abs(fit_gpu_fraction(0.6, int(39.4 * gib), int(79.2 * gib)) - (38.4 / 79.2)) < 1e-6
    # plenty free: the requested value is returned unchanged
    assert fit_gpu_fraction(0.6, 70 * gib, 80 * gib) == 0.6
    # nothing free: never negative
    assert fit_gpu_fraction(0.6, 0, 80 * gib) == 0.0
    assert fit_gpu_fraction(0.6, 10, 0) == 0.0

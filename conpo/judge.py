"""Quality-veto judge (ConPO.pdf section 2.3): an LLM judge served by vLLM as
an OpenAI-compatible HTTP server decides whether y_minus is a degenerate
response compared with y_plus, ignoring the target constraint families. The
veto only discards admitted pairs; it never relabels.

HTTP transport ported and adapted from
C:/Users/leong/Desktop/reserch2/stackelberg-if/serve_client.py:
  - proxy-bypass opener            source lines 35-40
  - strip_think                    source lines 53-58
  - read_endpoint                  source lines 61-77
  - wait_for_endpoint              source lines 80-98
  - healthy                        source lines 101-111
  - chat                           source lines 126-168 (adapted: `model` and
    `base` become explicit parameters instead of module-global state set by
    configure(), since each HttpJudge owns one endpoint and one model name).
configure(), completions(), and the ReplayLogger are not ported: nothing in
this module's contract calls them.
"""

from __future__ import annotations

import json
import random
import re
import time
import urllib.request
import zlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conpo import config

# Bypass the site proxy explicitly (serve_client.py:35-40): a request to an
# internal compute node handed to an institutional proxy gets a 503, since
# the proxy cannot connect back to it.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

API_PATH = "/v1/chat/completions"

_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def strip_think(text: str) -> str:
    """Remove a leading (possibly empty) think block a chat template inserts."""
    return _THINK_BLOCK.sub("", text, count=1)


def read_endpoint(path: Path) -> str:
    """Parse an endpoint file ("hostname port jobid") into a base URL.

    All three fields are required: a torn write truncated inside the port
    still leaves two tokens with a numeric second field and must fail
    closed.
    """
    text = path.read_text().strip()
    parts = text.split()
    if len(parts) < 3 or not parts[1].isdigit():
        raise ValueError(
            f"malformed endpoint file {path}: {text!r} (expected 'hostname port jobid')"
        )
    return f"http://{parts[0]}:{parts[1]}"


def wait_for_endpoint(path: Path, wait: float = 3600.0) -> str:
    """Poll for the judge server's endpoint file and return its base URL.

    A malformed file mid-poll is a partial write; a file that vanishes
    between the existence check and the read is a server relaunch. Polling
    continues through both.
    """
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < wait:
        if path.is_file():
            try:
                return read_endpoint(path)
            except (ValueError, OSError):
                pass
        time.sleep(5)
    raise TimeoutError(f"no endpoint at {path} after {wait}s. Is the judge server running?")


def healthy(base: str, timeout: float = 10.0) -> bool:
    """True if the judge server answers /health, False on any failure."""
    try:
        with _OPENER.open(f"{base}/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def resolve_endpoint(path_or_url: str | Path) -> str:
    """Resolve the judge's base URL.

    RUN_JUDGE_ENDPOINT, when set, is used verbatim as the base URL.
    Otherwise `path_or_url` is read as an endpoint file in the read_endpoint
    format, written by the judge server's SLURM job.
    """
    url = config.env("RUN_JUDGE_ENDPOINT", "")
    if url:
        return url
    return read_endpoint(Path(path_or_url))


def chat(
    prompt: str,
    seed: int,
    max_tokens: int,
    temperature: float,
    *,
    base: str,
    model: str = config.JUDGE_MODEL,
    timeout: float = 120.0,
    thinking: bool = False,
    structured_outputs: dict[str, Any] | None = None,
) -> str:
    """One chat exchange against the judge's vLLM OpenAI endpoint.

    Fail-loud contract, unchanged from the ported source: an HTTP error
    raises (OpenerDirector's default error handling), and a response body
    missing choices[0].message.content raises ValueError rather than
    returning an empty string.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "seed": seed,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    if structured_outputs is not None:
        payload["structured_outputs"] = structured_outputs
    req = urllib.request.Request(
        f"{base}{API_PATH}",
        method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with _OPENER.open(req, timeout=timeout) as r:
        data = json.loads(r.read())
    content = data["choices"][0]["message"].get("content")
    if content is None:
        raise ValueError(f"chat response has no content: {json.dumps(data)[:300]}")
    return strip_think(content)


# ---------------------------------------------------------------------------
# Veto request/verdict and prompt construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VetoRequest:
    """One quality-veto question: is y_minus degenerate relative to y_plus."""

    prompt: str
    y_plus: str
    y_minus: str
    families: tuple[str, ...]
    descriptions: tuple[str, ...]


@dataclass(frozen=True)
class Verdict:
    """keep = not degenerate; the veto only discards, never relabels."""

    keep: bool
    reason: str
    raw: str


def _prompt_seed(req: VetoRequest) -> int:
    """Deterministic RNG seed derived from the request content.

    Same convention as conpo.verifiers._kwargs_seed (zlib.crc32 over a
    canonical string), so the label assignment below is reproducible for a
    given request without carrying extra state.
    """
    key = f"veto:{req.prompt}:{req.y_plus}:{req.y_minus}:{','.join(req.families)}"
    return zlib.crc32(key.encode())


def build_veto_prompt(req: VetoRequest) -> str:
    """Build the veto prompt, showing y_plus/y_minus as Response A/B.

    Which response is labelled A is decided by a seed derived from the
    request content, so the judge cannot learn a fixed position bias across
    requests, and a repeated call for the same request is stable. The prompt
    text then names the letter carrying y_minus explicitly, so the judge is
    told exactly which response to evaluate rather than asked to infer it.
    """
    swap = random.Random(_prompt_seed(req)).random() < 0.5
    a_text, b_text = (req.y_minus, req.y_plus) if swap else (req.y_plus, req.y_minus)
    minus_label, plus_label = ("A", "B") if swap else ("B", "A")
    ignore_list = "\n".join(f"- {d}" for d in req.descriptions)
    return (
        "You are judging response quality for a constraint-following task.\n\n"
        f"Task prompt:\n{req.prompt}\n\n"
        f"Response A:\n{a_text}\n\n"
        f"Response B:\n{b_text}\n\n"
        "The following constraint requirements are IN PLAY for this task, but "
        "you must IGNORE them when judging quality: a response that violates "
        "one of these is expected and is NOT a quality defect.\n"
        f"{ignore_list}\n\n"
        f"Response {minus_label} is a deliberately modified variant of "
        f"response {plus_label}. Judge only whether response {minus_label} is "
        "DEGENERATE: truncated, garbled, off-task, or changed in meaning "
        f"beyond the ignored constraints above, compared with response "
        f"{plus_label}.\n\n"
        "Answer with a single JSON object and nothing else: "
        '{"degenerate": true or false, "reason": "one short sentence"}'
    )


def parse_verdict(raw: str) -> Verdict:
    """Parse a judge reply into a Verdict.

    Strips a leading think block, then scans for the first JSON object
    anywhere in the remaining text, tolerating surrounding prose. Raises
    ValueError when no JSON object is found or the "degenerate" key is
    missing.
    """
    text = strip_think(raw)
    decoder = json.JSONDecoder()
    obj: Any = None
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            obj, _ = decoder.raw_decode(text, start)
            break
        except json.JSONDecodeError:
            idx = start + 1
    if not isinstance(obj, dict):
        raise ValueError(f"no JSON object found in judge response: {raw!r}")
    if "degenerate" not in obj:
        raise ValueError(f"judge JSON missing 'degenerate' key: {obj!r}")
    degenerate = bool(obj["degenerate"])
    reason = str(obj.get("reason", ""))
    return Verdict(keep=not degenerate, reason=reason, raw=raw)


# ---------------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------------

_VETO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "degenerate": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["degenerate", "reason"],
}


class HttpJudge:
    """Quality-veto judge served over HTTP by vLLM.

    veto_many runs sequentially; batching the HTTP calls is the upgrade path
    once judge latency is measured against training throughput.
    """

    def __init__(
        self,
        base_url: str,
        model: str = config.JUDGE_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 256,
        seed: int = 0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        # ponytail: VetoRequest carries no forward/reverse edge tag, so this
        # counts keep vs discard outcomes only; a caller tracking edge
        # direction keeps its own per-direction split on top of these counts.
        self.counts: Counter[str] = Counter()

    def veto(self, req: VetoRequest) -> Verdict:
        prompt = build_veto_prompt(req)
        raw = chat(
            prompt,
            self.seed,
            self.max_tokens,
            self.temperature,
            base=self.base_url,
            model=self.model,
            structured_outputs={"json": _VETO_SCHEMA},
        )
        verdict = parse_verdict(raw)
        self.counts["keep" if verdict.keep else "discard"] += 1
        return verdict

    def veto_many(self, reqs: Sequence[VetoRequest]) -> list[Verdict]:
        return [self.veto(req) for req in reqs]


class StaticJudge:
    """Fixed-verdict judge for tests and the no-veto ablation."""

    def __init__(self, keep: bool = True) -> None:
        self.keep = keep
        self.counts: Counter[str] = Counter()

    def veto(self, req: VetoRequest) -> Verdict:
        verdict = Verdict(keep=self.keep, reason="StaticJudge", raw="")
        self.counts["keep" if verdict.keep else "discard"] += 1
        return verdict

    def veto_many(self, reqs: Sequence[VetoRequest]) -> list[Verdict]:
        return [self.veto(req) for req in reqs]

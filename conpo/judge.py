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
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from conpo import config
from conpo.edges import Edge

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


class Judge(Protocol):
    """Structural contract every judge implements.

    veto is defined in terms of veto_many on every judge in this module
    (veto(req) == veto_many([req])[0]), so a caller that only has one
    request still goes through the same code path as a batch of one.
    """

    def veto(self, req: VetoRequest) -> Verdict: ...

    def veto_many(self, reqs: Sequence[VetoRequest]) -> list[Verdict]: ...


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


# ---------------------------------------------------------------------------
# Batched veto (user decision 2026-09-02): verifier-admitted pairs wait in a
# PendingQueue instead of calling judge.veto inline; the trainer flushes the
# queue in one judge burst every config.VETO_PERIOD_K optimiser steps (or
# earlier, see conpo/trainer.py). Two H100s, one training run per card, so
# the judge cannot sit resident beside training: VLLMBatchedJudge and
# HFBatchedJudge both load their model only for the duration of veto_many
# and delete it before returning.
# ---------------------------------------------------------------------------

_DIRECTIONS = frozenset({"forward", "reverse"})


@dataclass(frozen=True)
class PendingPair:
    """One verifier-admitted (exactness + distance gate) pair awaiting a
    batched veto verdict. `edge` is stamped with its construction step
    already, so queueing time does not affect w(e) age decay."""

    edge: Edge
    request: VetoRequest
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)}, got {self.direction!r}")


class PendingQueue:
    """FIFO queue of PendingPair awaiting one judge burst.

    Bounded by max_size: a caller is expected to flush before the queue
    reaches it (config.PENDING_QUEUE_MAX), so push() raising past max_size
    is a fail-fast signal of a missed flush, not a normal code path.
    """

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self._items: deque[PendingPair] = deque()

    def __len__(self) -> int:
        return len(self._items)

    def push(self, pair: PendingPair) -> None:
        if len(self._items) >= self.max_size:
            raise ValueError(f"PendingQueue is full (max_size={self.max_size}); flush before pushing more")
        self._items.append(pair)

    def drain(self) -> list[PendingPair]:
        """Return every queued pair and empty the queue."""
        items = list(self._items)
        self._items.clear()
        return items


class VLLMBatchedJudge:
    """Quality-veto judge run locally with vllm.LLM, loaded once per flush burst.

    vllm is imported inside veto_many only: construction never imports it,
    so build_judge can select this judge on a machine (or in a test) with no
    vllm install, and the training loop can construct it long before the
    first flush. before_load/after_unload let the caller sleep and wake a
    colocated training vLLM engine around the burst (GRPOConfig's
    vllm_enable_sleep_mode, see conpo/trainer.py) so the two engines never
    both hold GPU memory at once.
    """

    def __init__(
        self,
        model: str,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.35,
        quantization: str | None = None,
        before_load: Callable[[], None] | None = None,
        after_unload: Callable[[], None] | None = None,
    ) -> None:
        self.model_name = model
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.quantization = quantization
        self.before_load = before_load
        self.after_unload = after_unload
        self.counts: Counter[str] = Counter()

    def veto(self, req: VetoRequest) -> Verdict:
        return self.veto_many([req])[0]

    def veto_many(self, reqs: Sequence[VetoRequest]) -> list[Verdict]:
        if not reqs:
            return []
        import gc

        import torch
        # ponytail: no GPU in this environment to run vllm against, so the
        # exact call shape (LLM.chat + SamplingParams(structured_outputs=...))
        # is unverified here; verify against the real judge model the first
        # time this path runs on the H100 estate, upgrade if the installed
        # vllm version's API differs. structured_outputs (not guided_decoding)
        # confirmed against the installed trl==1.9.2 package's own vLLM
        # integration, which uses only StructuredOutputsParams/
        # structured_outputs= (trl/generation/vllm_generation.py,
        # trl/scripts/vllm_serve.py) -- the vllm version paired with this trl
        # release has moved past the guided_decoding field name.
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        if self.before_load is not None:
            self.before_load()
        # llm is constructed inside the try so a failure here (plausible: GPU
        # memory pressure from a not-yet-fully-freed colocated training
        # engine is exactly the scenario this class exists to handle) still
        # runs after_unload() in the finally below, mirroring before_load().
        llm = None
        try:
            llm = LLM(
                model=self.model_name,
                max_model_len=self.max_model_len,
                gpu_memory_utilization=self.gpu_memory_utilization,
                quantization=self.quantization,
            )
            messages_batch = [[{"role": "user", "content": build_veto_prompt(r)}] for r in reqs]
            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=256,
                structured_outputs=StructuredOutputsParams(json=_VETO_SCHEMA),
            )
            outputs = llm.chat(messages_batch, sampling, chat_template_kwargs={"enable_thinking": False})
            verdicts: list[Verdict] = []
            for out in outputs:
                raw = out.outputs[0].text
                try:
                    verdict = parse_verdict(raw)
                except ValueError:
                    verdict = Verdict(keep=False, reason="judge reply did not parse (fail-closed)", raw=raw)
                verdicts.append(verdict)
                self.counts["keep" if verdict.keep else "discard"] += 1
            return verdicts
        finally:
            if llm is not None:
                del llm
                gc.collect()
                torch.cuda.empty_cache()
            if self.after_unload is not None:
                self.after_unload()


class HFBatchedJudge:
    """Quality-veto judge run locally with a plain transformers model,
    loaded once per flush burst (same load-then-delete shape as
    VLLMBatchedJudge, for a machine or test with no vllm install)."""

    def __init__(
        self,
        model: str,
        device: str = "cpu",
        dtype: Any = None,
        max_new_tokens: int = 64,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.counts: Counter[str] = Counter()

    def veto(self, req: VetoRequest) -> Verdict:
        return self.veto_many([req])[0]

    def veto_many(self, reqs: Sequence[VetoRequest]) -> list[Verdict]:
        if not reqs:
            return []
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs: dict[str, Any] = {} if self.dtype is None else {"dtype": self.dtype}
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        model.to(self.device)
        model.eval()

        verdicts: list[Verdict] = []
        try:
            for start in range(0, len(reqs), self.batch_size):
                batch = reqs[start : start + self.batch_size]
                prompts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": build_veto_prompt(r)}],
                        add_generation_prompt=True,
                        tokenize=False,
                        enable_thinking=False,
                    )
                    for r in batch
                ]
                encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.no_grad():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                prompt_len = encoded["input_ids"].shape[1]
                replies = tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
                for raw in replies:
                    raw = strip_think(raw)
                    try:
                        verdict = parse_verdict(raw)
                    except ValueError:
                        verdict = Verdict(keep=False, reason="judge reply did not parse (fail-closed)", raw=raw)
                    verdicts.append(verdict)
                    self.counts["keep" if verdict.keep else "discard"] += 1
        finally:
            del model
            if self.device != "cpu":
                torch.cuda.empty_cache()
        return verdicts


def build_judge(
    *,
    endpoint_path: str | Path = "",
    device: str = "cpu",
    dtype: Any = None,
    before_load: Callable[[], None] | None = None,
    after_unload: Callable[[], None] | None = None,
) -> HttpJudge | StaticJudge | VLLMBatchedJudge | HFBatchedJudge:
    """Select a Judge from the RUN_* environment and config (design 2026-09-03).

    RUN_NO_VETO=1 wins over everything else. Otherwise RUN_JUDGE_MODE picks
    the placement ("http" for the always-resident judge server, or the
    default "batched" for the per-flush-burst judge, config.JUDGE_MODE_DEFAULT);
    RUN_USE_VLLM picks which batched implementation loads the local model;
    RUN_JUDGE_MODEL overrides config.JUDGE_MODEL for whichever judge is
    built. `device`/`dtype`/`before_load`/`after_unload` are the trainer's
    own knobs (unavailable as RUN_* strings) for the batched judges.
    """
    if config.env("RUN_NO_VETO", "0") == "1":
        return StaticJudge(keep=True)

    model = config.env("RUN_JUDGE_MODEL", "") or config.JUDGE_MODEL
    mode = config.env("RUN_JUDGE_MODE", getattr(config, "JUDGE_MODE_DEFAULT", "batched"))
    if mode == "http":
        return HttpJudge(resolve_endpoint(endpoint_path), model=model)
    if mode != "batched":
        raise ValueError(f"unknown RUN_JUDGE_MODE: {mode!r} (expected 'http' or 'batched')")

    if config.env("RUN_USE_VLLM", "0") == "1":
        return VLLMBatchedJudge(model, before_load=before_load, after_unload=after_unload)
    return HFBatchedJudge(model, device=device, dtype=dtype)

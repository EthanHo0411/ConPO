"""ConPOTrainer: GRPOTrainer subclass implementing ConPO (ConPO.pdf sections 2.2-2.7).

`_generate_and_score_completions` runs the stock GRPO clean-group generation
via `super()`, then (per `arm`) drives forward repair (conpo/forward.py) and
reverse corruption (conpo/reverse.py) over that group's completions. Under
the batched veto (USER DECISIONS 2026-09-02, the default judge placement),
every verifier-admitted pair goes into `pending_queue` instead of straight
into `edge_buffer`; `flush_pending_veto` scores the whole queue in one judge
burst every `veto_period_k` steps, before a checkpoint save, or on an
explicit call, caching reference log-probs and pushing only kept edges. A
forward trajectory's feasible endpoint waits the same way (`_PendingFeasible`,
`_defer_feasible_endpoint`/`_resolve_feasible_tracker`): it only reaches
`feasible_buffer` once every edge its trajectory built is confirmed kept, so
`reverse.reverse_batch`'s same-step sampling of `feasible_buffer` never draws
a response the real judge has not yet approved. Both paths write one
mechanism-log record per step or per burst.
`compute_loss` adds L_resp and L_prompt (conpo/losses.py) on top of TRL's own
GRPO objective, sampled from `edge_buffer`, plus an optional gradient-alignment
probe (conpo/selection.py section 2.6).
"""

from __future__ import annotations

import dataclasses
import json
import math
import random
import time
import warnings
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from trl import GRPOTrainer

from conpo import config, forward, losses, prompts, reverse, selection
from conpo.arms import ArmSpec, Overrides
from conpo.data import PoolRow, TokenMeter
from conpo.edges import Edge, EdgeBuffer, FeasibleBuffer
from conpo.engine import Engine
from conpo.judge import PendingPair, PendingQueue, Verdict, VetoRequest, strip_think
from conpo.selection import FamilyStats
from conpo.verifiers import feasible, verifier_vector


class _QueueingVetoCollector:
    """Judge-shaped adapter handed to forward.py/reverse.py while the batched
    veto is active (Judge protocol: veto/veto_many).

    Every call is answered with a tentative keep=True while the VetoRequest
    is recorded in call order. forward.py/reverse.py append one edge
    immediately after each veto() call that keeps, so `captured` lines up
    pairwise, in order, with the edges the caller collects: the trainer zips
    them back together into PendingPair entries for the real judge to score
    later, in one burst (ConPOTrainer.flush_pending_veto). This is how
    "forward and reverse construction stop calling judge.veto inline" is
    achieved without editing forward.py/reverse.py at all: they still call
    `judge.veto(...)`, only never on the real judge.

    This is NOT exactly what a same-step immediate admission would have
    looked like for forward.py's per-step candidate search
    (`repair_trajectory`'s `for candidate in proposals` loop): with a real
    judge, a veto reject on the first monotone-admitted candidate lets that
    loop try the next proposal in the same step; answering every call
    keep=True means the loop always takes the first monotone-admitted
    candidate and `break`s before a later, possibly better, candidate is
    even looked at. The real verdict for that first candidate arrives only
    later, at flush time, with no fallback to the candidates that were never
    tried.
    # ponytail: single-path collection (queue the first monotone-admitted
    # candidate per step, decide later) instead of deferring the whole
    # per-step candidate search into the batch; upgrade path is to have
    # repair_trajectory queue every monotone-admitted candidate for a step
    # and only commit to one -- and only advance y_t -- once that step's
    # batch verdicts are back, if the lost retry measurably hurts edge yield
    # on the real judge.
    """

    def __init__(self) -> None:
        self.captured: list[VetoRequest] = []

    def veto(self, req: VetoRequest) -> Verdict:
        self.captured.append(req)
        return Verdict(keep=True, reason="pending_veto", raw="")

    def veto_many(self, reqs: Sequence[VetoRequest]) -> list[Verdict]:
        return [self.veto(r) for r in reqs]


@dataclasses.dataclass
class _PendingFeasible:
    """One forward trajectory's feasible endpoint, withheld from
    feasible_buffer until every edge its trajectory built has a real
    keep=True veto verdict (review4/round2 finding 1): a placeholder
    keep=True from `_QueueingVetoCollector` let `y_t` advance during
    construction, so the endpoint text is only as trustworthy as its
    trajectory's edges turn out to be once the real judge scores them.

    `remaining` starts as `{id(edge) for edge in the trajectory's edges}`;
    a rejected edge sets `dead` so the endpoint is dropped outright, however
    many of its siblings turn out kept -- reverse.reverse_batch must never
    sample a response built on a transition the judge called degenerate.
    """

    key: str
    response: str
    remaining: set[int]
    dead: bool = False


def _completion_text(completion: object) -> str:
    """Response text from either a plain string or a TRL conversational
    completion (`[{"role": "assistant", "content": ...}]`)."""
    if isinstance(completion, list):
        return str(completion[-1]["content"])
    return str(completion)


def score_response(
    model: torch.nn.Module, tokenizer: object, prompt_text: str, response: str
) -> tuple[torch.Tensor, int]:
    """log pi(response | prompt_text) and its completion-token length.

    One forward pass on `model` as it stands: adapters enabled or disabled,
    and gradient tracking or not, are entirely the caller's responsibility
    (wrap the call in `torch.no_grad()` / `model.disable_adapter()` for a
    reference score; call plainly for a policy score that needs a gradient).
    Used identically for y_plus and y_minus.
    """
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=False,
        **config.POLICY_CHAT_TEMPLATE_KWARGS,
    )
    device = next(model.parameters()).device
    prompt_ids = tokenizer(rendered_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.cat([torch.zeros_like(prompt_ids), torch.ones_like(response_ids)], dim=1)
    logp = losses.sequence_logprobs(model, input_ids, attention_mask, completion_mask)[0]
    length = int(losses.completion_lengths(completion_mask)[0].item())
    return logp, length


class ConPOTrainer(GRPOTrainer):
    """GRPOTrainer subclass implementing ConPO.

    TRL folds `beta * KL(pi_theta || pi_ref)` into its own clean-objective
    loss (trl's `GRPOTrainer._compute_loss`), so `compute_loss` below never
    adds a separate KL term itself: the effective KL weight applied in
    practice is `LAMBDA_G * KL_BETA` (config.py), and the paper's objective
    `L = L_resp + lambda_P L_prompt + lambda_G L_clean + beta L_KL` is
    realised exactly only at `LAMBDA_G = 1`.

    pi_ref is the frozen initial policy: with a zero-initialised LoRA
    adapter (peft's default init), the adapter's output equals the base
    model's, so every `logp_ref_*` cached on an Edge is computed once, at
    admission, under `accelerator.unwrap_model(model).disable_adapter()`
    (see `score_response`), and never recomputed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args: object,
        train_dataset: object,
        processing_class: object,
        arm: ArmSpec,
        rows_by_key: Mapping[str, PoolRow],
        edge_buffer: EdgeBuffer,
        feasible_buffer: FeasibleBuffer,
        stats: FamilyStats,
        judge: object,
        engine_factory: Callable[["ConPOTrainer"], Engine],
        meter: TokenMeter,
        mechanism_log_path: str | Path,
        seed: int = 0,
        overrides: Overrides | None = None,
        probe_enabled: bool = True,
        veto_period_k: int = config.VETO_PERIOD_K,
        pending_queue_max: int = config.PENDING_QUEUE_MAX,
        **kwargs: object,
    ) -> None:
        self.arm = arm
        self.overrides = overrides if overrides is not None else Overrides()
        self.rows_by_key: dict[str, PoolRow] = dict(rows_by_key)
        self.edge_buffer = edge_buffer
        self.feasible_buffer = feasible_buffer
        self.stats = stats
        self.judge = judge
        self.meter = meter
        self.mechanism_log_path = Path(mechanism_log_path)
        self.mechanism_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.probe_enabled = probe_enabled
        self.last_loss_stats: dict[str, float] = {}
        # Guards the T_c probe to once per optimiser step: self.state.global_step
        # is constant across every micro-batch of one accumulation cycle, so a
        # value change (including the first call, None != 0) marks a new step.
        self._last_probe_step: int | None = None
        # Lifetime count of edges admitted into edge_buffer, independent of the
        # buffer's own FIFO/age-cap eviction, for the cumulative mechanism-log rate.
        self.cumulative_admitted_edges = 0

        # Batched veto (USER DECISIONS 2026-09-02): edges that pass exactness/
        # distance wait here, tagged with the VetoRequest that will decide
        # them, until flush_pending_veto scores the whole queue in one burst
        # -- every veto_period_k optimiser steps, when the queue reaches
        # pending_queue_max, or right before a checkpoint save (_save_checkpoint
        # below). scripts/train.py calls flush_pending_veto() once more after
        # trainer.train() returns, so nothing is left unscored at run end.
        self.pending_queue = PendingQueue(pending_queue_max)
        self.veto_period_k = veto_period_k
        self._steps_since_flush = 0
        # Round2 finding 1: forward feasible endpoints wait here (keyed by
        # id(edge)) until every edge their trajectory built is resolved by
        # flush_pending_veto -- see _PendingFeasible and _defer_feasible_endpoint.
        self._pending_feasible_by_edge_id: dict[int, _PendingFeasible] = {}
        self._sleep_warned = False
        self._veto_seen_by_direction: Counter[str] = Counter()
        self._veto_rejected_by_direction: Counter[str] = Counter()

        reward_func = self.reward_conjunctive if arm.reward == "conjunctive" else self.reward_additive
        super().__init__(
            model=model,
            reward_funcs=reward_func,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            **kwargs,
        )
        self.engine: Engine = engine_factory(self)

    # -----------------------------------------------------------------
    # Reward functions (the arm decides which one is passed to GRPOTrainer)
    # -----------------------------------------------------------------

    def reward_conjunctive(self, completions: Sequence[object], key: Sequence[str], **_: object) -> list[float]:
        """F(x, y) (ConPO.pdf's conjunctive reward): 1.0 iff every constraint on the row passes, else 0.0."""
        scores = []
        for completion, k in zip(completions, key):
            row = self.rows_by_key[k]
            text = _completion_text(completion)
            scores.append(1.0 if feasible(row.constraints, text) else 0.0)
        return scores

    def reward_additive(self, completions: Sequence[object], key: Sequence[str], **_: object) -> list[float]:
        """open-instruct IFEvalVerifier semantics: mean per-constraint 0/1 verdict; empty completion scores 0."""
        scores = []
        for completion, k in zip(completions, key):
            row = self.rows_by_key[k]
            text = _completion_text(completion)
            if not text.strip() or not row.constraints:
                scores.append(0.0)
                continue
            verdicts = verifier_vector(row.constraints, text)
            scores.append(sum(verdicts) / len(verdicts))
        return scores

    # -----------------------------------------------------------------
    # Counterfactual construction
    # -----------------------------------------------------------------

    def _generate_and_score_completions(self, inputs: list[dict[str, object]]) -> dict[str, object]:
        output = super()._generate_and_score_completions(inputs)

        tokens_before_step = self.meter.total
        keys = [str(row["key"]) for row in inputs]
        completions_text = [
            strip_think(t)
            for t in self.processing_class.batch_decode(output["completion_ids"], skip_special_tokens=True)
        ]
        self.meter.add(self.engine.count_tokens(completions_text))

        current_update = self.state.global_step
        n = self.num_generations if self.model.training else self.num_generations_eval
        advantages = output["advantages"]

        forward_reasons: Counter[str] = Counter()
        reverse_reasons: Counter[str] = Counter()
        new_edges: list[Edge] = []
        degenerate_groups = 0
        total_groups = 0

        for start in range(0, len(inputs), n):
            end = start + n
            key = keys[start]
            row = self.rows_by_key[key]
            group_completions = completions_text[start:end]
            group_advantages = advantages[start:end]
            total_groups += 1
            if torch.allclose(group_advantages, torch.zeros_like(group_advantages)):
                degenerate_groups += 1

            observations: dict[str, list[bool]] = {c.iid: [] for c in row.constraints}
            failed_responses: list[str] = []
            for text in group_completions:
                verdicts = verifier_vector(row.constraints, text)
                for c, ok in zip(row.constraints, verdicts):
                    observations[c.iid].append(ok)
                if verdicts and all(verdicts):
                    self.feasible_buffer.add(key, text)
                elif verdicts:
                    failed_responses.append(text)
            if observations:
                self.stats.update_pass_rates(observations)

            if self.arm.forward and failed_responses:
                veto_flag = not self.overrides.no_veto
                collector = _QueueingVetoCollector() if veto_flag else None
                result = forward.repair_failed_group(
                    row,
                    failed_responses,
                    self.engine,
                    collector if veto_flag else self.judge,
                    self.rng,
                    self.meter,
                    current_update,
                    veto=veto_flag,
                )
                if veto_flag:
                    # Register a tracker for every trajectory-derived endpoint
                    # BEFORE pushing its edges below (round2 finding 1): a
                    # push can trigger an early flush when pending_queue is
                    # already at pending_queue_max, and that flush must find
                    # the tracker already in place or the edge's verdict
                    # would resolve nothing. An endpoint with no source
                    # edges (repair_trajectory's already-feasible-input
                    # shape) needs no veto and goes straight in.
                    for endpoint, edges in zip(result.feasible_endpoints, result.endpoint_sources):
                        if edges:
                            self._defer_feasible_endpoint(key, endpoint, edges)
                        else:
                            self.feasible_buffer.add(key, endpoint)
                    # 1:1, in order: forward.py calls judge.veto() exactly once
                    # per admitted candidate, right before appending it (see
                    # _QueueingVetoCollector's docstring).
                    for edge, req in zip(result.edges, collector.captured):
                        self._push_pending(PendingPair(edge=edge, request=req, direction="forward"))
                else:
                    new_edges.extend(result.edges)
                    for endpoint in result.feasible_endpoints:
                        self.feasible_buffer.add(key, endpoint)
                forward_reasons.update(result.reasons)

        if self.arm.reverse and len(self.feasible_buffer) > 0:
            veto_flag = not self.overrides.no_veto
            collector = _QueueingVetoCollector() if veto_flag else None
            results = reverse.reverse_batch(
                self.rows_by_key,
                self.feasible_buffer,
                config.REVERSE_DRAWS_PER_STEP,
                self.stats,
                collector if veto_flag else self.judge,
                self.rng,
                current_update,
                selection=self.overrides.selection,
                exactness=not self.overrides.no_exactness,
                distance_gate=not self.overrides.no_distance,
                veto=veto_flag,
            )
            if veto_flag:
                captured_iter = iter(collector.captured)
                for r in results:
                    reverse_reasons[r.reason] += 1
                    if r.edge is not None:
                        self._push_pending(PendingPair(edge=r.edge, request=next(captured_iter), direction="reverse"))
            else:
                for r in results:
                    reverse_reasons[r.reason] += 1
                    if r.edge is not None:
                        new_edges.append(r.edge)

        # new_edges only ever holds edges admitted with the veto off
        # (Overrides.no_veto): under the batched veto (the default), every
        # veto-gated edge went into self.pending_queue above instead, and is
        # scored/pushed later by flush_pending_veto, not here. Round2 finding
        # 2: this means mean_edit_distance and edges_per_million_tokens_step
        # below are only ever real per-step statistics under the no-veto
        # ablation; under the default they are 0.0 every step, indistinguishable
        # in mechanism.jsonl from "measured zero" -- read burst_mean_edit_distance
        # on the veto_flush record instead for the batched-path equivalent.
        distances: list[float] = []
        admitted_by_direction: Counter[str] = Counter()
        unwrapped = self.accelerator.unwrap_model(self.model)
        with torch.no_grad(), unwrapped.disable_adapter():
            for edge in new_edges:
                logp_plus, len_plus = score_response(unwrapped, self.processing_class, edge.prompt, edge.y_plus)
                logp_minus, len_minus = score_response(unwrapped, self.processing_class, edge.prompt, edge.y_minus)
                scored_edge = dataclasses.replace(
                    edge,
                    logp_ref_plus=logp_plus.item(),
                    logp_ref_minus=logp_minus.item(),
                    len_plus=len_plus,
                    len_minus=len_minus,
                )
                self.edge_buffer.push(scored_edge)
                admitted_by_direction[scored_edge.direction] += 1
                distances.append(float(scored_edge.meta.get("distance", 0.0)))

        self.cumulative_admitted_edges += len(new_edges)
        tokens_this_step = self.meter.total - tokens_before_step
        record = {
            "step": current_update,
            "admitted_by_direction": dict(admitted_by_direction),
            "veto_discarded_by_direction": {
                "forward": forward_reasons.get("veto", 0),
                "reverse": reverse_reasons.get("veto", 0),
            },
            "reasons_by_direction": {"forward": dict(forward_reasons), "reverse": dict(reverse_reasons)},
            "mean_edit_distance": (sum(distances) / len(distances)) if distances else 0.0,
            "degenerate_group_rate": (degenerate_groups / total_groups) if total_groups else 0.0,
            # Per-step: this step's new edges over this step's generated tokens.
            # Cumulative: all edges ever admitted over the meter's lifetime
            # total -- distinct from edge_buffer_size, which shrinks under
            # FIFO/age-cap eviction and so undercounts the cumulative rate.
            "edges_per_million_tokens_step": (
                (len(new_edges) / tokens_this_step * 1e6) if tokens_this_step else 0.0
            ),
            "edges_per_million_tokens_cumulative": (
                (self.cumulative_admitted_edges / self.meter.total * 1e6) if self.meter.total else 0.0
            ),
            "edge_buffer_size": len(self.edge_buffer),
            "feasible_buffer_size": len(self.feasible_buffer),
            "family_stats": self.stats.snapshot(),
        }
        with open(self.mechanism_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # The periodic flush condition (USER DECISIONS 2026-09-02); the
        # queue-size condition is enforced per-push by _push_pending above
        # (PendingQueue.push raises past max_size, so a check only here would
        # be too late); the other two conditions are _save_checkpoint below
        # (before a checkpoint save) and scripts/train.py's explicit call
        # after trainer.train() (end of training) -- both outside this
        # per-step method.
        self._steps_since_flush += 1
        if self._steps_since_flush >= self.veto_period_k:
            self.flush_pending_veto()

        if self.meter.exhausted():
            self.control.should_training_stop = True

        return output

    # -----------------------------------------------------------------
    # Batched veto
    # -----------------------------------------------------------------

    def _defer_feasible_endpoint(self, key: str, response: str, edges: Sequence[Edge]) -> None:
        """Register `response` as withheld from feasible_buffer until every
        edge in `edges` is resolved by flush_pending_veto (round2 finding
        1). `edges` is never empty here -- callers route the empty case
        straight to feasible_buffer.add instead."""
        tracker = _PendingFeasible(key=key, response=response, remaining={id(e) for e in edges})
        for edge in edges:
            self._pending_feasible_by_edge_id[id(edge)] = tracker

    def _resolve_feasible_tracker(self, edge: Edge, kept: bool) -> None:
        """Called once per scored pair in flush_pending_veto: checks `edge`
        off its feasible-endpoint tracker, if it has one. A reject kills the
        tracker outright (its trajectory built on a transition the real
        judge called degenerate, so the endpoint is never trustworthy,
        however many of its other edges are kept); a keep only checks it
        off, and the endpoint reaches feasible_buffer only once every edge
        on its trajectory has been checked off with none dead."""
        tracker = self._pending_feasible_by_edge_id.pop(id(edge), None)
        if tracker is None:
            return
        if not kept:
            tracker.dead = True
            return
        tracker.remaining.discard(id(edge))
        if not tracker.remaining and not tracker.dead:
            self.feasible_buffer.add(tracker.key, tracker.response)

    def _push_pending(self, pair: PendingPair) -> None:
        """Push one pair, flushing first if the queue is already at
        pending_queue_max (PendingQueue.push raises past that size, since a
        caller is expected to flush before it fills -- this is that flush)."""
        if len(self.pending_queue) >= self.pending_queue.max_size:
            self.flush_pending_veto()
        self.pending_queue.push(pair)

    def flush_pending_veto(self) -> None:
        """Score every pending pair in one burst (USER DECISIONS 2026-09-02).

        No-op (nothing scored, no JSONL line) when the queue is empty, so a
        flush-condition check that finds nothing pending costs nothing and
        never adds an empty record to mechanism_log_path. Otherwise: confirm
        the colocated vLLM engine is asleep if configured to
        (_sleep_engine_if_enabled -- it never calls `.sleep()` itself, only
        reports the invariant TRL's own generate() already establishes),
        score the whole drained queue with one self.judge.veto_many call,
        wake the engine back up, then score reference log-probs and push
        every kept edge into edge_buffer exactly as the old inline path did
        -- rejected pairs are dropped, nothing more. Always resets the
        veto_period_k step counter, flushed or not.
        """
        self._steps_since_flush = 0
        pending_size_before = len(self.pending_queue)
        if pending_size_before == 0:
            return

        pairs = self.pending_queue.drain()
        requests = [p.request for p in pairs]

        t0 = time.perf_counter()
        slept = self._sleep_engine_if_enabled()
        verdicts = self.judge.veto_many(requests)
        if slept:
            self._wake_engine()
        burst_seconds = time.perf_counter() - t0

        kept: Counter[str] = Counter()
        rejected: Counter[str] = Counter()
        kept_distances: list[float] = []
        unwrapped = self.accelerator.unwrap_model(self.model)
        with torch.no_grad(), unwrapped.disable_adapter():
            for pair, verdict in zip(pairs, verdicts):
                # Round2 finding 1: check this edge off its feasible-endpoint
                # tracker (if any) whichever way the verdict fell, before the
                # early-continue below -- a reject must kill the tracker too.
                self._resolve_feasible_tracker(pair.edge, verdict.keep)
                if not verdict.keep:
                    rejected[pair.direction] += 1
                    continue
                kept[pair.direction] += 1
                edge = pair.edge
                kept_distances.append(float(edge.meta.get("distance", 0.0)))
                logp_plus, len_plus = score_response(unwrapped, self.processing_class, edge.prompt, edge.y_plus)
                logp_minus, len_minus = score_response(unwrapped, self.processing_class, edge.prompt, edge.y_minus)
                scored_edge = dataclasses.replace(
                    edge,
                    logp_ref_plus=logp_plus.item(),
                    logp_ref_minus=logp_minus.item(),
                    len_plus=len_plus,
                    len_minus=len_minus,
                )
                self.edge_buffer.push(scored_edge)
                self.cumulative_admitted_edges += 1

        for direction in ("forward", "reverse"):
            self._veto_seen_by_direction[direction] += kept[direction] + rejected[direction]
            self._veto_rejected_by_direction[direction] += rejected[direction]

        def veto_rate(direction: str) -> float:
            seen = self._veto_seen_by_direction[direction]
            return (self._veto_rejected_by_direction[direction] / seen) if seen else 0.0

        record = {
            "event": "veto_flush",
            "step": self.state.global_step,
            "pending_size_before": pending_size_before,
            "burst_seconds": burst_seconds,
            "burst_kept_forward": kept["forward"],
            "burst_kept_reverse": kept["reverse"],
            "burst_rejected_forward": rejected["forward"],
            "burst_rejected_reverse": rejected["reverse"],
            "veto_rate_forward": veto_rate("forward"),
            "veto_rate_reverse": veto_rate("reverse"),
            # Round2 finding 2: the per-step record's mean_edit_distance is
            # only ever real under the no-veto ablation (see the comment in
            # _generate_and_score_completions); this is its batched-path
            # equivalent, computed from the same edge.meta["distance"] the
            # per-step version reads, over this burst's kept edges alone.
            "burst_mean_edit_distance": (sum(kept_distances) / len(kept_distances)) if kept_distances else 0.0,
        }
        with open(self.mechanism_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _sleep_engine_if_enabled(self) -> bool:
        """True iff RUN_USE_VLLM=1, args.vllm_enable_sleep_mode is on, and
        the trainer has a colocated vLLM engine handle (USER DECISIONS
        2026-09-02): confirms the engine's KV cache is already free for the
        judge to use during the burst, so flush_pending_veto knows to call
        _wake_engine() afterwards -- it does NOT call `.sleep()` itself.

        Every flush only ever runs after this step's `super().
        _generate_and_score_completions()` call has already run
        (_generate_and_score_completions calls flush directly or indirectly
        for every trigger: the periodic check below, _save_checkpoint, and
        scripts/train.py's end-of-training call all fire after that point in
        the same step), and TRL's own `VLLMGeneration.generate()`
        (trl/generation/vllm_generation.py) already puts the colocated
        engine to `sleep(level=2)` unconditionally at the end of every call
        when `vllm_enable_sleep_mode` is on -- so by the time a burst starts
        the engine is already asleep. Calling `.sleep()` again here would be
        a second sleep with no intervening wake, which vllm's sleep/wake
        pair does not support (a state transition, not idempotent).

        UNTESTED: no GPU in this environment, verified only by reading
        trl/generation/vllm_generation.py's own wake_up(tags=...)/
        sleep(level=2) calls (conpo/engine.py's VLLMColocateEngine
        docstring names the same source). False with no warning for
        HFEngine/FakeEngine runs (RUN_USE_VLLM=0), since there is no
        colocated engine to protect; False with a once-only warning when
        RUN_USE_VLLM=1 but vllm_enable_sleep_mode is off, since the judge
        then contends with the awake engine for GPU memory during the burst.
        """
        if config.env("RUN_USE_VLLM", "0") != "1":
            return False
        if not getattr(self.args, "vllm_enable_sleep_mode", False):
            if not self._sleep_warned:
                warnings.warn(
                    "veto burst running with the colocated vLLM engine awake: set "
                    "vllm_enable_sleep_mode=True (GRPOConfig) to free GPU memory for "
                    "the judge during the burst.",
                    stacklevel=2,
                )
                self._sleep_warned = True
            return False
        llm = getattr(getattr(self.engine, "trainer", None), "vllm_generation", None)
        llm = getattr(llm, "llm", None)
        return llm is not None

    def _wake_engine(self) -> None:
        self.engine.trainer.vllm_generation.llm.wake_up()  # type: ignore[attr-defined]

    def _save_checkpoint(self, model: torch.nn.Module, trial: object) -> None:
        """Flush before every checkpoint save (USER DECISIONS 2026-09-02),
        so a saved edges.jsonl / resumed run never carries unscored pairs."""
        self.flush_pending_veto()
        super()._save_checkpoint(model, trial)

    # -----------------------------------------------------------------
    # Losses
    # -----------------------------------------------------------------

    def _score_edge_pairs(self, model: torch.nn.Module, edges: Sequence[Edge]) -> tuple[torch.Tensor, torch.Tensor]:
        """Delta_theta(e) (ConPO.pdf section 2.4) and age(e) for a batch of edges.

        One policy forward pass per response (2 per edge, gradient-tracked);
        logp_ref_*/len_* come from each edge's cache, filled once at
        admission by `_generate_and_score_completions`.

        ponytail: one edge at a time (2*len(edges) separate forward calls,
        batch size 1 each) instead of padding the batch into one call; add
        batched scoring if per-step throughput on the real GPU run measures
        as a bottleneck.
        """
        logp_plus, logp_minus, ref_plus, ref_minus, len_plus, len_minus, ages = [], [], [], [], [], [], []
        for edge in edges:
            lp_plus, _ = score_response(model, self.processing_class, edge.prompt, edge.y_plus)
            lp_minus, _ = score_response(model, self.processing_class, edge.prompt, edge.y_minus)
            logp_plus.append(lp_plus)
            logp_minus.append(lp_minus)
            ref_plus.append(edge.logp_ref_plus)
            ref_minus.append(edge.logp_ref_minus)
            len_plus.append(edge.len_plus)
            len_minus.append(edge.len_minus)
            ages.append(self.edge_buffer.age(edge, self.state.global_step))
        device = logp_plus[0].device
        q_plus = losses.q_theta(
            torch.stack(logp_plus),
            torch.tensor(ref_plus, dtype=torch.float32, device=device),
            torch.tensor(len_plus, dtype=torch.long, device=device),
        )
        q_minus = losses.q_theta(
            torch.stack(logp_minus),
            torch.tensor(ref_minus, dtype=torch.float32, device=device),
            torch.tensor(len_minus, dtype=torch.long, device=device),
        )
        ages_t = torch.tensor(ages, dtype=torch.long, device=device)
        return q_plus - q_minus, ages_t

    def _prompt_margins(self, model: torch.nn.Module, edges: Sequence[Edge]) -> tuple[torch.Tensor, torch.Tensor]:
        """m_plus = m_theta(x^+A), m_minus = m_theta(x^-A) (ConPO.pdf section 2.5) for single-family edges.

        x^+A is `edge.prompt` itself (counterfactual_prompts leaves it
        unchanged), so its reference log-probs are already cached on the
        edge; x^-A's reference log-probs are computed fresh here, under
        `disable_adapter()`, since no edge caches them.
        """
        unwrapped = self.accelerator.unwrap_model(model)
        m_plus_terms: list[torch.Tensor] = []
        m_minus_terms: list[torch.Tensor] = []
        for edge in edges:
            row = self.rows_by_key[edge.prompt_id]
            family = next(iter(edge.families))
            target_index = next(i for i, c in enumerate(row.constraints) if c.iid == family)
            x_plus, x_minus = prompts.counterfactual_prompts(row, target_index)

            lp_plus_xplus, _ = score_response(model, self.processing_class, x_plus, edge.y_plus)
            lp_minus_xplus, _ = score_response(model, self.processing_class, x_plus, edge.y_minus)
            ref_plus_xplus = torch.tensor(edge.logp_ref_plus, dtype=lp_plus_xplus.dtype, device=lp_plus_xplus.device)
            ref_minus_xplus = torch.tensor(
                edge.logp_ref_minus, dtype=lp_minus_xplus.dtype, device=lp_minus_xplus.device
            )
            q_plus_xplus = losses.q_theta(
                lp_plus_xplus, ref_plus_xplus, torch.tensor(edge.len_plus, device=lp_plus_xplus.device)
            )
            q_minus_xplus = losses.q_theta(
                lp_minus_xplus, ref_minus_xplus, torch.tensor(edge.len_minus, device=lp_minus_xplus.device)
            )
            m_plus_terms.append(q_plus_xplus - q_minus_xplus)

            lp_plus_xminus, len_p_xminus = score_response(model, self.processing_class, x_minus, edge.y_plus)
            lp_minus_xminus, len_m_xminus = score_response(model, self.processing_class, x_minus, edge.y_minus)
            with torch.no_grad(), unwrapped.disable_adapter():
                ref_plus_xminus, _ = score_response(unwrapped, self.processing_class, x_minus, edge.y_plus)
                ref_minus_xminus, _ = score_response(unwrapped, self.processing_class, x_minus, edge.y_minus)
            q_plus_xminus = losses.q_theta(
                lp_plus_xminus, ref_plus_xminus.detach(), torch.tensor(len_p_xminus, device=lp_plus_xminus.device)
            )
            q_minus_xminus = losses.q_theta(
                lp_minus_xminus, ref_minus_xminus.detach(), torch.tensor(len_m_xminus, device=lp_minus_xminus.device)
            )
            m_minus_terms.append(q_plus_xminus - q_minus_xminus)

        return torch.stack(m_plus_terms), torch.stack(m_minus_terms)

    def _run_probe(self, model: torch.nn.Module, clean: torch.Tensor, current_update: int) -> None:
        """T_c gradient-alignment probe (ConPO.pdf section 2.6): cosine between
        L_resp's gradient on a probe batch of the due family and L_clean's
        gradient on the current inputs, w.r.t. the LoRA parameters only.

        `torch.autograd.grad` (not `.backward()`) never touches `.grad`
        attributes, so this has no optimiser side effects; `retain_graph`
        keeps both graphs alive for the real `total.backward()` still to come.
        """
        families = sorted(self.edge_buffer.counts_by_family())
        if not families:
            return
        family = selection.next_probe_family(self.stats, families, current_update)
        if family is None or not selection.probe_due(self.stats, family, current_update, config.PROBE_PERIOD_S):
            return
        probe_edges = self.edge_buffer.sample_where(
            lambda e: family in e.families, config.PROBE_SIZE, current_update, self.rng
        )
        if not probe_edges:
            return
        lora_params = [p for name, p in model.named_parameters() if "lora_" in name and p.requires_grad]
        if not lora_params:
            return

        delta, ages = self._score_edge_pairs(model, probe_edges)
        probe_loss, _ = losses.resp_loss(delta, ages, tau=config.TAU, gamma=config.GAMMA, kappa=config.KAPPA)
        probe_grad = torch.autograd.grad(probe_loss, lora_params, retain_graph=True, allow_unused=True)
        clean_grad = torch.autograd.grad(clean, lora_params, retain_graph=True, allow_unused=True)
        probe_grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(probe_grad, lora_params)]
        clean_grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(clean_grad, lora_params)]
        cosine = selection.grad_cosine(probe_grad, clean_grad)
        self.stats.record_probe(family, cosine, current_update)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, object],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if return_outputs:
            raise ValueError("ConPOTrainer does not support return_outputs")

        clean = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
        total = config.LAMBDA_G * clean
        current_update = self.state.global_step
        stats: dict[str, float] = {"clean_loss": clean.item()}

        # config.EDGE_PAIRS_PER_STEP / PROMPT_PAIRS_PER_STEP are per-OPTIMISER-step
        # budgets (config.py), but compute_loss runs once per micro-batch under
        # gradient accumulation; split the budget so the per-optimiser-step total
        # sampled from edge_buffer stays at the configured value regardless of
        # accumulation, rather than being drawn in full on every micro-batch.
        grad_accum = self.args.gradient_accumulation_steps
        edge_n = math.ceil(config.EDGE_PAIRS_PER_STEP / grad_accum)
        prompt_n = math.ceil(config.PROMPT_PAIRS_PER_STEP / grad_accum)

        if self.arm.resp_loss and len(self.edge_buffer) > 0:
            resp_edges = self.edge_buffer.sample(edge_n, current_update, self.rng, balanced=True)
            if resp_edges:
                delta, ages = self._score_edge_pairs(model, resp_edges)
                loss_resp, resp_stats = losses.resp_loss(
                    delta, ages, tau=config.TAU, gamma=config.GAMMA, kappa=config.KAPPA
                )
                total = total + loss_resp
                stats["resp_loss"] = loss_resp.item()
                stats["resp_mean_age"] = ages.float().mean().item()
                stats.update({f"resp_{k}": v for k, v in resp_stats.items()})

                margins_by_direction_family: dict[tuple[str, str], list[float]] = {}
                for edge, margin in zip(resp_edges, delta.detach().tolist()):
                    family = next(iter(edge.families)) if len(edge.families) == 1 else "multi"
                    margins_by_direction_family.setdefault((edge.direction, family), []).append(margin)
                for (direction, family), values in margins_by_direction_family.items():
                    stats[f"resp_margin_{direction}_{family}"] = sum(values) / len(values)

        if self.arm.prompt_loss and self.overrides.lambda_p > 0 and len(self.edge_buffer) > 0:
            prompt_edges = self.edge_buffer.sample_where(
                lambda e: len(e.families) == 1, prompt_n, current_update, self.rng
            )
            if prompt_edges:
                m_plus, m_minus = self._prompt_margins(model, prompt_edges)
                loss_prompt, prompt_stats = losses.prompt_loss(
                    m_plus, m_minus, tau_p=config.TAU_P, lambda_0=config.LAMBDA_0
                )
                total = total + self.overrides.lambda_p * loss_prompt
                stats["prompt_loss"] = loss_prompt.item()
                stats.update({f"prompt_{k}": v for k, v in prompt_stats.items()})

        if self.probe_enabled and len(self.edge_buffer) > 0 and current_update != self._last_probe_step:
            self._run_probe(model, clean, current_update)
            self._last_probe_step = current_update

        self.last_loss_stats = stats
        mode = "train" if model.training else "eval"
        for name, value in stats.items():
            self._metrics[mode][name].append(value)

        return total

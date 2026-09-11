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

import contextlib
import dataclasses
import json
import math
import random
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path

import torch
from accelerate.utils import is_peft_model
from transformers import TrainerCallback
from trl import GRPOTrainer

from conpo import config, forward, losses, prompts, reverse, selection
from conpo.arms import ArmSpec, Overrides
from conpo.compositor import PairStats
from conpo.data import PoolRow, TokenMeter
from conpo.distance import token_distance
from conpo.edges import Edge, EdgeBuffer, FeasibleBuffer
from conpo.engine import Engine
from conpo.judge import PendingPair, PendingQueue, Verdict, VetoRequest, strip_think
from conpo.selection import FamilyStats
from conpo.verifiers import feasible, verifier_vector


# Design 2026-09-04, item 3: at most this many judged pairs per burst get a
# raw-reply sample written to RUN_OUT_DIR/judge_replies.jsonl -- a sample for
# eyeballing judge behaviour, not a full audit log of every verdict.
JUDGE_REPLY_DUMP_PER_BURST = 8


def _pair_edit_distance(pair: PendingPair) -> float:
    """token_distance(y_plus, y_minus) for one pending pair, at flush time
    (design 2026-09-04, item 4): forward.py and reverse.py always stamp
    `edge.meta["distance"]` at construction, so that cached value is used
    when present; a pair without one (defensive -- no current producer
    omits it) falls back to computing it fresh from the request text."""
    stored = pair.edge.meta.get("distance")
    if isinstance(stored, (int, float)):
        return float(stored)
    return token_distance(pair.request.y_plus, pair.request.y_minus)


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
        # A placeholder, not a parse of anything, so parse_ok is True
        # regardless (design 2026-09-04, same convention as StaticJudge).
        return Verdict(keep=True, reason="pending_veto", raw="", parse_ok=True)

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


class _StepTimingCallback(TrainerCallback):
    """Per-optimiser-step wall clock and peak CUDA memory (design choice,
    Ethan devbox perf pass, 2026-09-11): `on_step_end` fires only after
    `accelerator.backward` and the optimiser's own `.step()` have both run
    for every micro-batch of the accumulation cycle -- strictly later than
    `compute_loss`'s own mechanism.jsonl write (USER DECISIONS / design
    2026-09-04, see the class docstring) -- so this can never be folded into
    that record and is appended as its own "step_timing" JSONL line instead,
    the same convention `flush_pending_veto`'s "veto_flush" record already
    uses. `on_step_begin`/`on_step_end` each fire exactly once per optimiser
    step (HF Trainer's own accumulation-boundary contract), matching the
    granularity of `current_update`/`self.state.global_step` everywhere else
    in this module.

    optimiser_update_seconds is total_step_seconds minus every phase
    ConPOTrainer measured directly this step (`trainer._phase_seconds`:
    TRL generation, forward repair, reverse corruption, the periodic veto
    burst when it fires) -- a remainder, so it also folds in compute_loss's
    own forward/backward passes across every micro-batch of the
    accumulation cycle, not a direct measurement of optimizer.step() alone.
    # ponytail: remainder-based, not a dedicated compute_loss/backward
    # timer; split further only if that breakdown is ever needed.
    """

    def __init__(self, trainer: "ConPOTrainer") -> None:
        self.trainer = trainer

    def on_step_begin(self, args: object, state: object, control: object, **kwargs: object) -> None:
        self.trainer._step_t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_end(self, args: object, state: object, control: object, **kwargs: object) -> None:
        trainer = self.trainer
        if trainer._step_t0 is None:
            return
        total_seconds = time.perf_counter() - trainer._step_t0
        trainer._step_t0 = None
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        measured = sum(trainer._phase_seconds.values())
        record = {
            "event": "step_timing",
            # global_step has already advanced past this step's own
            # current_update by the time on_step_end fires (HF Trainer
            # increments it right before this callback), so subtract 1 to
            # tag the same step number the per-step generate/compute_loss
            # record used.
            "step": max(0, state.global_step - 1),  # type: ignore[attr-defined]
            "total_step_seconds": total_seconds,
            "optimiser_update_seconds": max(0.0, total_seconds - measured),
            "peak_memory_gb": peak_gb,
        }
        with open(trainer.mechanism_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


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
    (see `score_response`), and never recomputed. Under RUN_FULL_FT=1 there
    is no adapter to disable; `_reference_model` yields TRL's own
    `self.ref_model` instead (see its docstring).
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
        out_dir: str | Path | None = None,
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
        # design choice (builder, 2026-09-04): the per-step mechanism record is
        # built here but its file write is deferred to compute_loss (see the
        # merge there). _generate_and_score_completions runs before compute_loss
        # for the same step, so a record written here would only ever carry
        # last_loss_stats from the PREVIOUS step; stashing it and writing after
        # compute_loss lets the record carry this step's own prompt_gamma_theta.
        self._pending_mechanism_record: dict[str, object] | None = None
        # Guards the T_c probe to once per optimiser step: self.state.global_step
        # is constant across every micro-batch of one accumulation cycle, so a
        # value change (including the first call, None != 0) marks a new step.
        self._last_probe_step: int | None = None
        # Lifetime count of edges admitted into edge_buffer, independent of the
        # buffer's own FIFO/age-cap eviction, for the cumulative mechanism-log rate.
        self.cumulative_admitted_edges = 0

        # H6 part B (design 2026-09-04): a PairStats only exists at all when
        # overrides.multi is set (RUN_MULTI=1); reverse_batch's own
        # `pair_stats` parameter treats None as "never attempt a multi-family
        # draw", so every other arm/override combination is unaffected.
        # `out_dir` (RUN_OUT_DIR) is optional so tests that never touch
        # multi-family edges or the judge-reply dump need not pass one --
        # save_pair_stats and _dump_judge_replies are then no-ops instead of
        # raising.
        self.out_dir: Path | None = Path(out_dir) if out_dir is not None else None
        self.pair_stats: PairStats | None = PairStats() if self.overrides.multi else None
        self._pair_stats_path: Path | None = self.out_dir / "pair_stats.json" if self.out_dir is not None else None
        if self.pair_stats is not None and self._pair_stats_path is not None and self._pair_stats_path.exists():
            self.pair_stats.load(self._pair_stats_path)
        # Cumulative reverse-edge / multi-reverse-edge admission counts
        # (mirrors cumulative_admitted_edges above), for
        # multi_edge_share_cumulative in the per-step mechanism-log record.
        self._cumulative_reverse_admitted = 0
        self._cumulative_multi_reverse_admitted = 0

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

        # Per-step phase timing (design choice, Ethan devbox perf pass,
        # 2026-09-11): _generate_and_score_completions fills _phase_seconds
        # fresh each time it runs (TRL's own generation-batch
        # caching skips it on every micro-batch but the first of an
        # accumulation cycle -- see trl/trainer/grpo_trainer.py's
        # _prepare_inputs); _StepTimingCallback reads it at on_step_end to
        # compute optimiser_update_seconds and write the "step_timing"
        # mechanism.jsonl record.
        self._phase_seconds: dict[str, float] = {}
        self._step_t0: float | None = None

        reward_func = self.reward_conjunctive if arm.reward == "conjunctive" else self.reward_additive
        callbacks = list(kwargs.pop("callbacks", None) or [])
        callbacks.append(_StepTimingCallback(self))
        super().__init__(
            model=model,
            reward_funcs=reward_func,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            **kwargs,
        )
        # RUN_FULL_FT=1 (design choice, builder 2026-09-04): GRPOTrainer builds
        # self.ref_model via create_reference_model only when the policy isn't
        # PEFT and beta != 0.0 (trl 1.9.2 grpo_trainer.py); config.KL_BETA is
        # nonzero in every arm, so this should always hold for a full fine-tune
        # run -- checked here, at construction, so a misconfiguration fails
        # before training rather than with an AttributeError on the first step.
        if not is_peft_model(self.accelerator.unwrap_model(self.model)) and self.ref_model is None:
            raise ValueError(
                "Full fine-tune (RUN_FULL_FT=1) needs a reference model, but the policy "
                "is not PEFT and self.ref_model is None. This happens only if beta == 0.0 "
                "(config.KL_BETA); set it nonzero or run with a PEFT adapter instead."
            )
        self.engine: Engine = engine_factory(self)

    @contextlib.contextmanager
    def _reference_model(self, model: torch.nn.Module) -> Iterator[torch.nn.Module]:
        """The model to score pi_ref under, with gradients disabled.

        PEFT: the unwrapped policy with its adapter turned off reverts to the
        frozen initial policy (class docstring). Full fine-tune: TRL's own
        `self.ref_model` instead, guaranteed non-None by the __init__ check
        above.
        """
        unwrapped = self.accelerator.unwrap_model(model)
        with torch.no_grad():
            if is_peft_model(unwrapped):
                with unwrapped.disable_adapter():
                    yield unwrapped
            else:
                yield self.ref_model

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
        t0 = time.perf_counter()
        output = super()._generate_and_score_completions(inputs)
        # Reset here, not in on_step_begin: TRL's own generation-batch cache
        # means this method (and therefore this step's forward/reverse work)
        # runs only on the first micro-batch of an accumulation
        # cycle, so phase timing must start from that same call.
        self._phase_seconds = {
            "trl_generation": time.perf_counter() - t0,
            "forward_repair": 0.0,
            "reverse_corruption": 0.0,
            "veto_burst": 0.0,
        }

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
        multi_attempted = 0

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
            # H6 part B: every sample's verifier vector, in constraint order,
            # for PairStats.update below -- the same verdicts computed for
            # `observations` a few lines down, never re-verified.
            group_verdicts: list[tuple[bool, ...]] = []
            for text in group_completions:
                verdicts = verifier_vector(row.constraints, text)
                group_verdicts.append(verdicts)
                for c, ok in zip(row.constraints, verdicts):
                    observations[c.iid].append(ok)
                if verdicts and all(verdicts):
                    self.feasible_buffer.add(key, text)
                elif verdicts:
                    failed_responses.append(text)
            if observations:
                self.stats.update_pass_rates(observations)
            if self.pair_stats is not None and group_verdicts:
                self.pair_stats.update(row.constraints, tuple(group_verdicts))

            if self.arm.forward and failed_responses:
                veto_flag = not self.overrides.no_veto
                collector = _QueueingVetoCollector() if veto_flag else None
                _t_forward = time.perf_counter()
                result = forward.repair_failed_group(
                    row,
                    failed_responses,
                    self.engine,
                    collector if veto_flag else self.judge,
                    self.rng,
                    self.meter,
                    current_update,
                    veto=veto_flag,
                    distance_gate=not self.overrides.no_distance,
                )
                self._phase_seconds["forward_repair"] += time.perf_counter() - _t_forward
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
            _t_reverse = time.perf_counter()
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
                pair_stats=self.pair_stats,
            )
            self._phase_seconds["reverse_corruption"] += time.perf_counter() - _t_reverse
            # A comma-joined ReverseResult.family (reverse.py's own
            # multi-vs-single-family signal) marks a multi-family attempt,
            # whatever it was ultimately admitted, distance-gated or vetoed.
            multi_attempted += sum(1 for r in results if r.family is not None and "," in r.family)
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
        multi_admitted = 0
        with self._reference_model(self.model) as ref_model:
            for edge in new_edges:
                logp_plus, len_plus = score_response(ref_model, self.processing_class, edge.prompt, edge.y_plus)
                logp_minus, len_minus = score_response(ref_model, self.processing_class, edge.prompt, edge.y_minus)
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
                if scored_edge.direction == "reverse":
                    self._cumulative_reverse_admitted += 1
                    if len(scored_edge.families) > 1:
                        self._cumulative_multi_reverse_admitted += 1
                        multi_admitted += 1

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
            # Perf-pass instrumentation (Ethan devbox, 2026-09-11): phase_seconds
            # covers everything this method measured directly this step
            # ("veto_burst" is 0.0 unless the periodic flush below
            # fires this step); optimiser_update_seconds and peak_memory_gb
            # land in a separate "step_timing" record instead (see
            # _StepTimingCallback) since they aren't knowable until after
            # this step's backward pass and optimiser step have both run,
            # strictly later than this record's own write in compute_loss.
            # Not a copy: the periodic flush below (still inside this method
            # call) may still add to "veto_burst" after this dict is built,
            # and record/self._pending_mechanism_record must see that update
            # before compute_loss serialises it -- see the reassignment (not
            # in-place clear) at this method's top for why the next step's
            # reset never mutates this same dict object retroactively.
            "phase_seconds": self._phase_seconds,
            "tokens_generated_step": tokens_this_step,
            # Raw anchors behind the ratios above (edges_per_million_tokens_*):
            # both are monotone non-decreasing over a run's lifetime, unlike
            # edge_buffer_size, so a reader can recompute any windowed rate
            # from two records without needing the buffer's own eviction state.
            "tokens_cumulative": self.meter.total,
            "cumulative_admitted_edges": self.cumulative_admitted_edges,
            # H6 part B: multi_attempted counts every reverse draw that tried
            # a multi-family target this step (admitted, distance-gated or
            # vetoed alike); multi_admitted only counts an edge that reached
            # edge_buffer here -- like admitted_by_direction above, that is
            # only ever nonzero under the no-veto ablation, since a
            # batched-veto multi edge is still sitting in pending_queue at
            # this point in the step. multi_edge_share_cumulative reads
            # regardless of veto mode: flush_pending_veto below also updates
            # the same cumulative counters.
            "multi_attempted": multi_attempted,
            "multi_admitted": multi_admitted,
            "multi_edge_share_cumulative": (
                (self._cumulative_multi_reverse_admitted / self._cumulative_reverse_admitted)
                if self._cumulative_reverse_admitted
                else 0.0
            ),
        }
        if self.pair_stats is not None and current_update % config.PROBE_PERIOD_S == 0:
            record["pair_stats_top_pairs"] = [list(row) for row in self.pair_stats.top_pairs(5)]
        # Stashed, not written here -- see the __init__ comment on
        # _pending_mechanism_record. compute_loss merges in this step's
        # prompt_* diagnostics (prompt_gamma_theta included) and writes the
        # single record for this step.
        self._pending_mechanism_record = record

        # The periodic flush condition (USER DECISIONS 2026-09-02); the
        # queue-size condition is enforced per-push by _push_pending above
        # (PendingQueue.push raises past max_size, so a check only here would
        # be too late); the other two conditions are _save_checkpoint below
        # (before a checkpoint save) and scripts/train.py's explicit call
        # after trainer.train() (end of training) -- both outside this
        # per-step method.
        self._steps_since_flush += 1
        if self._steps_since_flush >= self.veto_period_k:
            self._phase_seconds["veto_burst"] += self.flush_pending_veto()

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

    def flush_pending_veto(self) -> float:
        """Score every pending pair in one burst (USER DECISIONS 2026-09-02).

        No-op (nothing scored, no JSONL line, returns 0.0) when the queue is
        empty, so a flush-condition check that finds nothing pending costs
        nothing and never adds an empty record to mechanism_log_path.
        Otherwise: confirm the colocated vLLM engine is asleep if configured
        to (_sleep_engine_if_enabled -- it never calls `.sleep()` itself,
        only reports the invariant TRL's own generate() already
        establishes), score the whole drained queue with one
        self.judge.veto_many call, wake the engine back up, then score
        reference log-probs and push every kept edge into edge_buffer
        exactly as the old inline path did -- rejected pairs are dropped,
        nothing more. Always resets the veto_period_k step counter, flushed
        or not. Returns the burst's wall-clock seconds (0.0 when a no-op),
        for _generate_and_score_completions's own phase_seconds["veto_burst"]
        tally (Ethan devbox perf pass, 2026-09-11).
        """
        self._steps_since_flush = 0
        pending_size_before = len(self.pending_queue)
        if pending_size_before == 0:
            return 0.0

        pairs = self.pending_queue.drain()
        requests = [p.request for p in pairs]
        # Item 4 fix (design 2026-09-04): every pair judged in this burst
        # gets an edit distance, kept or rejected -- see burst_mean_edit_distance
        # below for why averaging kept edges alone was wrong.
        distances = [_pair_edit_distance(p) for p in pairs]

        t0 = time.perf_counter()
        slept = self._sleep_engine_if_enabled()
        verdicts = self.judge.veto_many(requests)
        if slept:
            self._wake_engine()
        burst_seconds = time.perf_counter() - t0
        self._dump_judge_replies(pairs, verdicts)

        kept: Counter[str] = Counter()
        rejected: Counter[str] = Counter()
        # parse_failures (design 2026-09-04, item 2): counted separately from
        # rejected -- a parse failure always also counts as a rejected pair
        # (fail-closed), so burst_rejected_* keeps meaning "all rejections"
        # for any old reader, while these new counters isolate how many of
        # them were "the judge's reply never parsed" instead of "the judge
        # said this pair is degenerate".
        parse_failures: Counter[str] = Counter()
        with self._reference_model(self.model) as ref_model:
            for pair, verdict in zip(pairs, verdicts):
                # Round2 finding 1: check this edge off its feasible-endpoint
                # tracker (if any) whichever way the verdict fell, before the
                # early-continue below -- a reject must kill the tracker too.
                self._resolve_feasible_tracker(pair.edge, verdict.keep)
                if not verdict.parse_ok:
                    parse_failures[pair.direction] += 1
                if not verdict.keep:
                    rejected[pair.direction] += 1
                    continue
                kept[pair.direction] += 1
                edge = pair.edge
                logp_plus, len_plus = score_response(ref_model, self.processing_class, edge.prompt, edge.y_plus)
                logp_minus, len_minus = score_response(ref_model, self.processing_class, edge.prompt, edge.y_minus)
                scored_edge = dataclasses.replace(
                    edge,
                    logp_ref_plus=logp_plus.item(),
                    logp_ref_minus=logp_minus.item(),
                    len_plus=len_plus,
                    len_minus=len_minus,
                )
                self.edge_buffer.push(scored_edge)
                self.cumulative_admitted_edges += 1
                if scored_edge.direction == "reverse":
                    self._cumulative_reverse_admitted += 1
                    if len(scored_edge.families) > 1:
                        self._cumulative_multi_reverse_admitted += 1

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
            "burst_parse_failures_forward": parse_failures["forward"],
            "burst_parse_failures_reverse": parse_failures["reverse"],
            "veto_rate_forward": veto_rate("forward"),
            "veto_rate_reverse": veto_rate("reverse"),
            # Round2 finding 2: the per-step record's mean_edit_distance is
            # only ever real under the no-veto ablation (see the comment in
            # _generate_and_score_completions); this is its batched-path
            # equivalent. Item 4 fix (design 2026-09-04): averaged over every
            # pair judged this burst, instead of only the ones the judge
            # kept -- a burst that rejects everything still has nonzero
            # distances to report; averaging kept edges alone read 0.0 in
            # that case, indistinguishable from "measured zero".
            "burst_mean_edit_distance": (sum(distances) / len(distances)) if distances else 0.0,
        }
        with open(self.mechanism_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return burst_seconds

    def _dump_judge_replies(self, pairs: Sequence[PendingPair], verdicts: Sequence[Verdict]) -> None:
        """Append the first JUDGE_REPLY_DUMP_PER_BURST judged pairs of this
        burst to `<out_dir>/judge_replies.jsonl` (design 2026-09-04, item 3):
        a raw-reply sample for eyeballing judge behaviour on a small judge
        model, not a full audit log of every verdict. No-op when `out_dir`
        was never given to the constructor.
        """
        if self.out_dir is None:
            return
        path = self.out_dir / "judge_replies.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for pair, verdict in zip(pairs[:JUDGE_REPLY_DUMP_PER_BURST], verdicts[:JUDGE_REPLY_DUMP_PER_BURST]):
                record = {
                    "step": self.state.global_step,
                    "direction": pair.direction,
                    "families": list(pair.request.families),
                    "keep": verdict.keep,
                    "parse_ok": verdict.parse_ok,
                    "raw": verdict.raw,
                    "prompt_chars": len(pair.request.prompt),
                    "y_plus_chars": len(pair.request.y_plus),
                    "y_minus_chars": len(pair.request.y_minus),
                }
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
        so a saved edges.jsonl / resumed run never carries unscored pairs.
        Also persists pair_stats (H6 part B) so a resumed run's phi/count
        table survives the restart instead of starting cold."""
        self.flush_pending_veto()
        self.save_pair_stats()
        super()._save_checkpoint(model, trial)

    def save_pair_stats(self) -> None:
        """Persist `self.pair_stats` to `<out_dir>/pair_stats.json` (H6 part
        B). No-op when `overrides.multi` is off (`pair_stats` is None) or no
        `out_dir` was given to the constructor."""
        if self.pair_stats is None or self._pair_stats_path is None:
            return
        self._pair_stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.pair_stats.save(self._pair_stats_path)

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
        """m_plus = m_theta(x^+A), m_minus = m_theta(x^-A) (ConPO.pdf section 2.5).

        x^-A drops every family in `edge.families` (H6 part B:
        prompts.counterfactual_prompts_set generalises the single-family
        prompts.counterfactual_prompts this used to call directly --
        identical output for a single-family edge, per its own docstring --
        so a multi-family edge now joins the same L_prompt term instead of
        being excluded from it by compute_loss's own sampling predicate).
        x^+A is `edge.prompt` itself (unchanged), so its reference log-probs
        are already cached on the edge; x^-A's reference log-probs are
        computed fresh here, under `disable_adapter()`, since no edge caches
        them.
        """
        m_plus_terms: list[torch.Tensor] = []
        m_minus_terms: list[torch.Tensor] = []
        for edge in edges:
            row = self.rows_by_key[edge.prompt_id]
            # First occurrence of each family, same convention the old
            # single-family lookup used (`next(i for i, c in ... if c.iid ==
            # family)`), applied per family in edge.families.
            target_indices = [next(i for i, c in enumerate(row.constraints) if c.iid == family) for family in edge.families]
            x_plus, x_minus = prompts.counterfactual_prompts_set(row, target_indices)

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
            with self._reference_model(model) as ref_model:
                ref_plus_xminus, _ = score_response(ref_model, self.processing_class, x_minus, edge.y_plus)
                ref_minus_xminus, _ = score_response(ref_model, self.processing_class, x_minus, edge.y_minus)
            q_plus_xminus = losses.q_theta(
                lp_plus_xminus, ref_plus_xminus.detach(), torch.tensor(len_p_xminus, device=lp_plus_xminus.device)
            )
            q_minus_xminus = losses.q_theta(
                lp_minus_xminus, ref_minus_xminus.detach(), torch.tensor(len_m_xminus, device=lp_minus_xminus.device)
            )
            m_minus_terms.append(q_plus_xminus - q_minus_xminus)

        return torch.stack(m_plus_terms), torch.stack(m_minus_terms)

    def _resample_clean_loss(self, model: torch.nn.Module, inputs: dict[str, object]) -> torch.Tensor:
        """Fresh GRPO loss over a `config.PROBE_SIZE`-sequence (seeded off
        `self.rng`) subsample of this step's own `inputs` -- the T_c probe's
        clean-side ESTIMATE (task H design choice, see `_run_probe`'s
        docstring). Any `inputs` value that is a tensor batch-aligned with
        `completion_ids` is index-selected down to the subsample; everything
        else (None, scalars, unrelated shapes) passes through unchanged.

        `super().compute_loss` is TRL's own GRPOTrainer.compute_loss (the
        exact call the real `clean` already goes through), which also
        appends to `self._metrics[mode]` and, when adaptive entropy is on,
        mutates `self.entropy_coef` / `self._entropy_window_stats` /
        `self._last_world_entropy` -- real training state, not only a
        return value. This extra call is a probe-only side computation, so
        every one of those is snapshotted before and restored after: the
        subsample must contribute to T_c, never to logged metrics or the
        entropy controller.
        """
        batch_size = inputs["completion_ids"].shape[0]  # type: ignore[union-attr]
        k = min(config.PROBE_SIZE, batch_size)
        idx = torch.tensor(self.rng.sample(range(batch_size), k), device=inputs["completion_ids"].device)  # type: ignore[union-attr]
        sub_inputs = {
            key: value.index_select(0, idx)
            if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size
            else value
            for key, value in inputs.items()
        }

        mode = "train" if model.training else "eval"
        metrics_before = {k_: list(v) for k_, v in self._metrics[mode].items()}
        entropy_coef_before = self.entropy_coef
        entropy_window_before = self._entropy_window_stats
        last_world_entropy_before = self._last_world_entropy
        try:
            return super().compute_loss(model, sub_inputs, return_outputs=False)
        finally:
            self._metrics[mode].clear()
            self._metrics[mode].update(metrics_before)
            self.entropy_coef = entropy_coef_before
            self._entropy_window_stats = entropy_window_before
            self._last_world_entropy = last_world_entropy_before

    def _run_probe(
        self, model: torch.nn.Module, inputs: dict[str, object], clean: torch.Tensor, current_update: int
    ) -> None:
        """T_c gradient-alignment probe (ConPO.pdf section 2.6): cosine between
        L_resp's gradient on a probe batch of the due family and an estimate
        of L_clean's gradient, w.r.t. the LoRA parameters only.

        Memory bound (task H design choice, Ethan devbox, 2026-09-11): the
        clean side used to differentiate the full-batch `clean` tensor with
        `retain_graph=True`, so the full accumulation-batch graph, the
        probe-edge graph and both gradient sets were all alive at once --
        with long prompts that hit 96.5 GB on an 80 GB card
        (promptcap_smoke, task E2+H notes). Neither gradient here is needed
        after this method returns (`total.backward()` still differentiates
        the real, unmodified `clean`, which never has `retain_graph` called
        on it), so both are now taken with `retain_graph=False` from graphs
        that never duplicate the full batch: the probe-edge graph is its own
        small `PROBE_SIZE`-edge forward, and the clean side is a fresh,
        separate, PROBE_SIZE-sequence-or-fewer forward over a seeded
        subsample of `inputs` (`_resample_clean_loss`) rather than the exact
        full-batch `clean` -- an ESTIMATE, not the true full-batch gradient.
        T_c itself (cosine between two LoRA gradients) is unchanged.
        `clean` is still accepted (unused below) so callers do not have to
        special-case this signature; `torch.cuda.empty_cache()` after the
        probe returns whatever the two freed graphs held to the allocator
        before the real forward/backward pass runs.
        """
        del clean  # kept for signature stability; the clean side is resampled below
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

        try:
            delta, ages = self._score_edge_pairs(model, probe_edges)
            probe_loss, _ = losses.resp_loss(delta, ages, tau=config.TAU, gamma=config.GAMMA, kappa=config.KAPPA)
            probe_grad = torch.autograd.grad(probe_loss, lora_params, retain_graph=False, allow_unused=True)
            probe_grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(probe_grad, lora_params)]

            clean_sample = self._resample_clean_loss(model, inputs)
            clean_grad = torch.autograd.grad(clean_sample, lora_params, retain_graph=False, allow_unused=True)
            clean_grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(clean_grad, lora_params)]

            cosine = selection.grad_cosine(probe_grad, clean_grad)
            self.stats.record_probe(family, cosine, current_update)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

        # Design choice (orchestrator, 2026-09-04): L_resp stays UNGATED on
        # Overrides.multi. Forward multi-family repair edges are part of
        # every arm's edge stream and the hub arm "full" must not change;
        # H6 isolates the phi-selected reverse multi-family corruption edges
        # plus their prompt term (L_prompt below is the only term gated by
        # overrides.multi).
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

        if self.arm.prompt_loss and len(self.edge_buffer) > 0:
            # _prompt_margins handles a multi-family edge exactly like a
            # single-family one (H6 part B), but admitting `>= 1` families
            # only under overrides.multi: gating on the RUN_MULTI flag keeps
            # every non-H6 arm's L_prompt population (H1/H3/H4/H5's
            # full/forward_only/reverse_only runs) bit-identical to before
            # this change, even on the steps where forward.py's own
            # multi-constraint repair (independent of RUN_MULTI) admits a
            # multi-family edge into edge_buffer.
            family_predicate = (lambda e: len(e.families) >= 1) if self.overrides.multi else (lambda e: len(e.families) == 1)
            prompt_edges = self.edge_buffer.sample_where(family_predicate, prompt_n, current_update, self.rng)
            if prompt_edges:
                # The margins and prompt_gamma_theta diagnostic are computed
                # (and logged) whenever the arm carries L_prompt at all,
                # independent of overrides.lambda_p: the H5 ablation
                # (RUN_LAMBDA_P=0) needs prompt_gamma_theta on record to show
                # what L_prompt would have measured had it not been zeroed
                # out. Only the weighted contribution to `total` is gated on
                # lambda_p > 0 -- that is what "ablates L_prompt's
                # contribution" (arms.py's Overrides docstring) means.
                m_plus, m_minus = self._prompt_margins(model, prompt_edges)
                loss_prompt, prompt_stats = losses.prompt_loss(
                    m_plus, m_minus, tau_p=config.TAU_P, lambda_0=config.LAMBDA_0
                )
                if self.overrides.lambda_p > 0:
                    total = total + self.overrides.lambda_p * loss_prompt
                stats["prompt_loss"] = loss_prompt.item()
                stats.update({f"prompt_{k}": v for k, v in prompt_stats.items()})

        if self.probe_enabled and len(self.edge_buffer) > 0 and current_update != self._last_probe_step:
            self._run_probe(model, inputs, clean, current_update)
            self._last_probe_step = current_update

        if self._pending_mechanism_record is not None:
            self._pending_mechanism_record.update({k: v for k, v in stats.items() if k.startswith("prompt_")})
            with open(self.mechanism_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._pending_mechanism_record) + "\n")
            self._pending_mechanism_record = None

        self.last_loss_stats = stats
        mode = "train" if model.training else "eval"
        for name, value in stats.items():
            self._metrics[mode][name].append(value)

        return total

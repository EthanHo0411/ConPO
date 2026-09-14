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
import gc
import json
import random
import time
import warnings
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from accelerate.utils import is_peft_model
from transformers import TrainerCallback
from trl import GRPOTrainer
from trl.trainer.utils import nanmax, nanmin

from conpo import config, forward, losses, prompts, reverse, selection
from conpo.arms import ArmSpec, Overrides
from conpo.compositor import PairStats
from conpo.data import PoolRow, TokenMeter
from conpo.distance import token_distance
from conpo.edges import Edge, EdgeBuffer, FeasibleBuffer
from conpo.engine import Engine
from conpo.judge import PendingPair, PendingQueue, Verdict, VetoRequest, build_veto_prompt, strip_think, veto_plus_label
from conpo.selection import FamilyStats
from conpo.verifiers import feasible, verifier_vector


# B2 (audit 2026-09-14): mechanism.jsonl fields compute_loss populates from
# one micro-batch's own `stats` (resp_evicted, plus every "prompt_"-prefixed
# key -- prompt_loss, prompt_gamma_theta, prompt_mean_m_plus,
# prompt_mean_m_minus, prompt_clamp_frac -- and pair_scoring_seconds/
# pair_scoring_forwards) rather than from the whole accumulation cycle: a
# reader averaging these over several records is averaging single samples,
# not per-step means. Listed on every record as "sampled_fields" so a
# downstream analysis script does not have to know this by convention.
_SAMPLED_MECHANISM_FIELDS = (
    "resp_evicted",
    "pair_scoring_seconds",
    "pair_scoring_forwards",
    "prompt_loss",
    "prompt_gamma_theta",
    "prompt_mean_m_plus",
    "prompt_mean_m_minus",
    "prompt_clamp_frac",
)


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


def _pairs_this_microbatch(total: int, micro_batch_index: int, accumulation: int) -> int:
    """Exact share of a per-optimiser-step budget `total` to draw on
    micro-batch `micro_batch_index` (0-based) of an `accumulation`-step
    cycle, so the cycle's sum over every index in range(accumulation)
    equals `total` for any total/accumulation pair (B3, audit 2026-09-14;
    replaces the old math.ceil(total / accumulation) drawn on EVERY
    micro-batch, which overdrew whenever accumulation did not divide total
    evenly -- e.g. ceil(8 / 16) == 1 times 16 micro-batches == 16, double
    the configured 8). floor((i+1)*total/accumulation) -
    floor(i*total/accumulation) is the standard exact-remainder split: each
    index gets total // accumulation or that plus one, with the +1
    micro-batches spread evenly across the cycle instead of piled onto the
    first ceil(total/accumulation) of them."""
    return (total * (micro_batch_index + 1)) // accumulation - (total * micro_batch_index) // accumulation


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


def _pad_last_dim(tensor: torch.Tensor, width: int, side: str, value: float) -> torch.Tensor:
    """Right- or left-pad a 2D tensor's last (sequence) dimension to
    `width` with `value`; a no-op if it is already that wide. `side` is
    "left" or "right", matching TRL's own `pad()` padding-side convention
    (grpo_trainer.py: left for prompt tensors, right for completion-aligned
    ones) -- used by ConPOTrainer._splice_group_output (RUN_RESAMPLE_ZERO_STD)
    to reconcile two independently-padded _generate_and_score_completions
    outputs before splicing rows from one into the other."""
    extra = width - tensor.shape[-1]
    if extra <= 0:
        return tensor
    pad_spec = (extra, 0) if side == "left" else (0, extra)
    return torch.nn.functional.pad(tensor, pad_spec, value=value)


def _tokenize_prompt_response(
    tokenizer: object, prompt_text: str, response: str
) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """input_ids, attention_mask, completion_mask (CPU) for one (prompt, response) pair.

    Factored out of score_response so RUN_TOKEN_LOGRATIO_CLAMP's per-token
    policy/reference scoring (_token_clamped_q below) can tokenize once and
    score both sides on the exact same ids, guaranteeing the two per-token
    log-prob tensors line up token for token.
    """
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=False,
        **config.POLICY_CHAT_TEMPLATE_KWARGS,
    )
    prompt_ids = tokenizer(rendered_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"]
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.cat([torch.zeros_like(prompt_ids), torch.ones_like(response_ids)], dim=1)
    return input_ids, attention_mask, completion_mask


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
    input_ids, attention_mask, completion_mask = _tokenize_prompt_response(tokenizer, prompt_text, response)
    device = next(model.parameters()).device
    input_ids, attention_mask, completion_mask = (
        input_ids.to(device),
        attention_mask.to(device),
        completion_mask.to(device),
    )
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
        resume_from: str | Path | None = None,
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
        # B2 (audit 2026-09-14): resp_evicted/pair_scoring_seconds/
        # pair_scoring_forwards are single-micro-batch samples in the
        # mechanism record (unchanged, see sampled_fields below), but the
        # true per-optimiser-step totals need every micro-batch's
        # contribution, not just the first. Accumulated here across
        # compute_loss's calls for one cycle, reset at the top of
        # _generate_and_score_completions (the one call per cycle) once its
        # value has been captured into that new cycle's record as the
        # PREVIOUS cycle's totals -- the current cycle's own total is not
        # known yet when the first compute_loss call writes the record.
        self._resp_evicted_step_accum = 0
        self._pair_scoring_seconds_step_accum = 0.0
        self._pair_scoring_forwards_step_accum = 0
        # B3 (audit 2026-09-14): realised edge/prompt pairs actually drawn
        # this cycle (post edge_buffer.sample/sample_where, so a buffer
        # running dry still counts truthfully), accumulated and surfaced
        # the same way as the three counters above.
        self._edge_pairs_step_accum = 0
        self._prompt_pairs_step_accum = 0
        self._step_totals_initialized = False
        # B3: 0-based index of the next compute_loss call within its
        # accumulation cycle, feeding _pairs_this_microbatch's exact P/A
        # split; reset at the top of _generate_and_score_completions (the
        # one call per cycle), same convention as the accumulators above.
        self._microbatch_in_cycle = 0
        # Guards the T_c probe to once per optimiser step: self.state.global_step
        # is constant across every micro-batch of one accumulation cycle, so a
        # value change (including the first call, None != 0) marks a new step.
        self._last_probe_step: int | None = None
        # Lifetime count of edges admitted into edge_buffer, independent of the
        # buffer's own FIFO/age-cap eviction, for the cumulative mechanism-log rate.
        self.cumulative_admitted_edges = 0
        # RUN_REPAIR_BATCH_GENERATION (L2, default on): merge every failing
        # group's forward-repair calls for a step into one
        # forward.repair_all_groups call instead of one forward.
        # repair_failed_group call per group (_generate_and_score_completions
        # below). Read once here, not per step: nothing in a run changes it
        # after this trainer is built.
        self._repair_batch_generation = config.env("RUN_REPAIR_BATCH_GENERATION", "1") == "1"
        # RUN_CACHE_REF_LOGPROBS (L3, default on): _prompt_margins reads an
        # edge's x^-A reference log-probs from its cache once present
        # instead of re-running the reference model on every replay. Read
        # once here, same convention as _repair_batch_generation above.
        self._cache_ref_logprobs = config.env("RUN_CACHE_REF_LOGPROBS", "1") == "1"
        # RUN_KL_CLAMP (K4; decision 2026-09-14, default config.KL_CLAMP =
        # 40.0): clamps the KL-estimator diff (ref_per_token_logps -
        # per_token_logps) to +/- this value before the exp, in the
        # overridden _compute_loss below -- mirrors open-instruct's own KL
        # clamp (at 40). "none" (case-insensitive) or an explicitly empty
        # value disables the clamp (self._kl_clamp = None); a numeric "0" is
        # a real clamp, not the disable spelling (see config.py's KL_CLAMP
        # comment). Read once here, same convention as
        # _repair_batch_generation/_cache_ref_logprobs above.
        raw_kl_clamp = config.env("RUN_KL_CLAMP", str(config.KL_CLAMP))
        self._kl_clamp: float | None = (
            None if raw_kl_clamp.strip().lower() in ("", "none") else float(raw_kl_clamp)
        )
        # RUN_TOKEN_LOGRATIO_CLAMP (K6, default unset): clamps the per-token
        # policy-minus-reference log-ratio in _score_edge_pairs to +/- this
        # value before it is summed into Delta_theta(e)'s q_plus/q_minus, so
        # no single token can carry more than this many nats of margin. Read
        # once here, same convention as _kl_clamp above.
        # B5 (viii, audit 2026-09-14): acts on L_resp only (_score_edge_pairs
        # feeds resp_loss). L_prompt's m_theta margins (_prompt_margins,
        # losses.prompt_loss) are untouched and stay whole-sequence
        # regardless of this flag -- L_prompt's own on/off switch is
        # overrides.lambda_p / RUN_LAMBDA_P, unrelated to this one.
        raw_token_logratio_clamp = config.env("RUN_TOKEN_LOGRATIO_CLAMP", "")
        self._token_logratio_clamp: float | None = (
            float(raw_token_logratio_clamp) if raw_token_logratio_clamp else None
        )
        # RUN_RESAMPLE_ZERO_STD (K5, default unset): in
        # _generate_and_score_completions, a prompt group whose rewards all
        # have zero std (torch.allclose to zeros) carries no gradient signal
        # (GRPO advantages are all zero); when set, such a group is
        # replaced with a freshly drawn, freshly generated substitute prompt
        # (_resample_degenerate_groups) instead of being trained on as-is.
        # RUN_RESAMPLE_MAX_ROUNDS caps the number of extra generation rounds
        # spent trying (default 3, open-instruct has no direct equivalent);
        # once exhausted, the batch proceeds with whatever it still holds.
        # Read once here, same convention as _kl_clamp/_token_logratio_clamp
        # above.
        self._resample_zero_std = config.env("RUN_RESAMPLE_ZERO_STD", "0") == "1"
        self._resample_max_rounds = int(config.env("RUN_RESAMPLE_MAX_ROUNDS", "3"))
        # RUN_PAIR_OBJECTIVE (K8, default "sequence"): "sequence" keeps
        # _score_edge_pairs's existing whole-sequence margin (cached-scalar,
        # or K6-clamped) byte for byte; "span_local" replaces Delta_theta(e)
        # with a margin over ONLY the tokens where y_plus and y_minus differ
        # (conpo/losses.py's diff_span), so the tokens the pair shares carry
        # no gradient at all instead of diluting/cancelling it -- the fix
        # for the likelihood-displacement diagnosis (devbox 2026-09-13).
        # Validated eagerly, same convention as arms.py's resolve_arm, so a
        # typo fails at construction rather than silently training as
        # "sequence". Read once here, same convention as
        # _kl_clamp/_token_logratio_clamp above.
        # B5 (viii, audit 2026-09-14): like RUN_TOKEN_LOGRATIO_CLAMP above,
        # span_local only ever replaces _score_edge_pairs's L_resp margin;
        # L_prompt (overrides.lambda_p / RUN_LAMBDA_P) always stays the
        # whole-sequence m_theta margin from _prompt_margins, unaffected by
        # this flag.
        raw_pair_objective = config.env("RUN_PAIR_OBJECTIVE", "sequence")
        if raw_pair_objective not in ("sequence", "span_local"):
            raise ValueError(
                f"Unknown RUN_PAIR_OBJECTIVE: {raw_pair_objective!r}, expected 'sequence' or 'span_local'"
            )
        self._pair_objective = raw_pair_objective
        # RUN_POS_ANCHOR_WEIGHT (K8, default 0 = off): weight on span_local's
        # positive-anchor term (DPO-Positive: penalise a repaired token
        # whose policy log-prob falls below the reference's). Meaningless
        # under the "sequence" objective; read unconditionally anyway, same
        # convention as every other knob above.
        self._pos_anchor_weight = float(config.env("RUN_POS_ANCHOR_WEIGHT", "0"))
        # RUN_PAIR_SCALE_AUTO (fix devbox 2026-09-14, default on): compute_loss
        # adds loss_resp/loss_prompt (plain per-micro-batch means) to `total`
        # AFTER super().compute_loss() has already applied TRL's own
        # normalisation for loss_type; GRPOTrainer disables Trainer's usual
        # /gradient_accumulation_steps scaling of the whole returned loss for
        # every loss_type (it sets compute_loss_func so Trainer skips it,
        # trl/trainer/grpo_trainer.py), so backward() accumulates
        # gradient_accumulation_steps copies of that per-micro-batch pair-loss
        # mean per optimiser step -- inflating PAIR_LOSS_WEIGHT's effective
        # weight by ~gradient_accumulation_steps versus its intended meaning
        # (a batch-mean pair loss), independent of loss_type. "1" divides
        # both pair terms in compute_loss by
        # self.current_gradient_accumulation_steps (the same attribute TRL's
        # own _compute_loss divides the clean loss by) so PAIR_LOSS_WEIGHT
        # means one optimiser step sees PAIR_LOSS_WEIGHT times a batch-mean
        # pair loss, for grpo and dapo alike. "0" reproduces today's
        # byte-for-byte behaviour (existing runs' tuned PAIR_LOSS_WEIGHT
        # already implicitly absorbs the inflation). Read once here, same
        # convention as _kl_clamp/_token_logratio_clamp above.
        self._pair_scale_auto = config.env("RUN_PAIR_SCALE_AUTO", "1") == "1"
        # Side channel from _score_edge_pairs_span_local to compute_loss's
        # resp block (RUN_POS_ANCHOR_WEIGHT term): set immediately before
        # use and cleared immediately after, within the same compute_loss
        # call, so it never leaks across micro-batches or into the T_c
        # probe's own _score_edge_pairs call. Mirrors
        # self._pending_mechanism_record's existing across-call handoff.
        self._pending_span_anchor: torch.Tensor | None = None

        # Wall-clock accumulation across RUN_RESUME processes (scripts/train.py):
        # the node-local disk is wiped roughly every 48 hours, so one logical
        # run can span several process restarts. wall_seconds_before_resume is
        # whatever a prior process's conpo_state.json recorded (0.0 for a
        # fresh run); cumulative_wall_seconds() adds this process's own
        # elapsed time since construction on top, so run_metadata.json's
        # actual_h100h reflects the whole run, not only the last process.
        self._process_wall_start = time.monotonic()
        self.wall_seconds_before_resume: float = 0.0

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
        # Engine-time breakdown (Ethan devbox perf pass, 2026-09-11): set to
        # a zeroed dict only by _install_vllm_reload_guard,
        # only when a colocated vLLM engine exists -- stays None for
        # HFEngine/FakeEngine runs, so the per-step record's "engine_seconds"
        # key is absent (not zeroed) for them. See _install_vllm_reload_guard.
        self._engine_seconds: dict[str, float] | None = None
        # RUN_VLLM_PUSH_ONCE_PER_STEP (L1): always exists, unlike the
        # push-once bookkeeping _install_vllm_reload_guard itself owns
        # (_vllm_synced_this_step, _vllm_push_once, ...), because
        # _flush_deferred_vllm_sleep reads it unconditionally every step
        # (HFEngine/FakeEngine runs included, where it just stays False).
        self._vllm_sleep_pending: bool = False

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
        # Root-cause fix for the vLLM weight overwrite (builder 2026-09-11):
        # self.vllm_generation exists from here on if RUN_USE_VLLM=1 (trl
        # 1.9.2 grpo_trainer.py sets it inside super().__init__() above,
        # only when self.use_vllm), so this is the earliest point the guard
        # can inspect it.
        self._install_vllm_reload_guard()
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

        # RUN_RESUME (scripts/train.py): load ConPO's own state (everything
        # this class keeps outside TRL) from the checkpoint TRL itself will
        # resume from. Runs after every other piece of state above exists,
        # so load_state_dict/load calls below have somewhere to write.
        if resume_from is not None:
            self._load_conpo_state(Path(resume_from))

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
        # B2: this method's own top is exactly the accumulation-cycle
        # boundary (see its B1 docstring note below), so by the time it
        # runs, every compute_loss call of the PREVIOUS cycle has already
        # added its contribution to the three accumulators below -- capture
        # that finished total before resetting for the new cycle. Absent on
        # the very first call of the run (nothing preceded it).
        previous_step_totals: dict[str, object] | None = None
        if self._step_totals_initialized:
            previous_step_totals = {
                "step": self.state.global_step - 1,
                "resp_evicted_step": self._resp_evicted_step_accum,
                "pair_scoring_seconds_step": self._pair_scoring_seconds_step_accum,
                "pair_scoring_forwards_step": self._pair_scoring_forwards_step_accum,
                # B3: realised draws across the previous cycle's micro-batches.
                "edge_pairs_step": self._edge_pairs_step_accum,
                "prompt_pairs_step": self._prompt_pairs_step_accum,
            }
        self._resp_evicted_step_accum = 0
        self._pair_scoring_seconds_step_accum = 0.0
        self._pair_scoring_forwards_step_accum = 0
        self._edge_pairs_step_accum = 0
        self._prompt_pairs_step_accum = 0
        self._microbatch_in_cycle = 0
        self._step_totals_initialized = True
        # Reset before super()'s own call, not after (unlike _phase_seconds
        # below): TRL's rollout generation happens inside that call, through
        # the same wrapped vllm_generation.generate/llm.wake_up/sleep the
        # guard installed, so the accumulator must already read zero when
        # that call starts or the rollout's own timings would be lost.
        if self._engine_seconds is not None:
            self._engine_seconds = {
                "wake_reload_resync": 0.0,
                "generate_only": 0.0,
                "sleep": 0.0,
                "generate_calls": 0,
                "reload_calls": 0,
                "reload_calls_skipped": 0,
            }
            # RUN_VLLM_PUSH_ONCE_PER_STEP (L1): a step boundary is exactly
            # this method's own top, same reasoning as _engine_seconds above
            # -- so both push-once trackers reset here too.
            self._vllm_synced_this_step = False
            self._vllm_sleep_pending = False
        t0 = time.perf_counter()
        output = super()._generate_and_score_completions(inputs)
        # n moved up here (was computed further down, just above the
        # per-group loop): RUN_RESAMPLE_ZERO_STD needs the group size before
        # that loop runs, to find and replace degenerate groups in `inputs`/
        # `output` while there is still time for the rest of this method to
        # see the replacement instead of the original.
        n = self.num_generations if self.model.training else self.num_generations_eval
        resample_groups_replaced = 0
        resample_rounds = 0
        if self._resample_zero_std:
            resample_groups_replaced, resample_rounds = self._resample_degenerate_groups(inputs, output, n)
        # Reset here, not in on_step_begin: TRL's own generation-batch cache
        # means this method (and therefore this step's forward/reverse work)
        # runs only on the first micro-batch of an accumulation
        # cycle, so phase timing must start from that same call.
        self._phase_seconds = {
            # B5 (i, audit 2026-09-14): times the whole
            # super()._generate_and_score_completions() call above, not
            # rollout sampling alone -- TRL's own GRPOTrainer also runs its
            # reward function(s) and the two no-grad log-prob passes (old
            # policy, reference) inside that same call. Field name kept as
            # "trl_generation" (mechanism.jsonl / trainer metrics reader
            # contract); this is a documentation-only note.
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
        advantages = output["advantages"]

        forward_reasons: Counter[str] = Counter()
        reverse_reasons: Counter[str] = Counter()
        # B1 (audit 2026-09-14): every forward.RepairResult this step's own
        # candidate_calls/distinct_candidate_calls, summed across every
        # group -- see forward_distinct_candidate_frac in `record` below.
        forward_candidate_calls = 0
        forward_distinct_candidate_calls = 0
        new_edges: list[Edge] = []
        degenerate_groups = 0
        total_groups = 0
        multi_attempted = 0
        # RUN_REPAIR_BATCH_GENERATION (L2): every failing group's own
        # (key, row, failed_responses, judge_for_group) collected here
        # instead of repaired inline, so ONE forward.repair_all_groups call
        # below can merge every group's engine.generate calls for the whole
        # step. judge_for_group is exactly what the flag-off branch below
        # would have passed as repair_failed_group's own `judge` argument
        # (a fresh _QueueingVetoCollector under the batched veto, else
        # self.judge) -- one per group, never shared, so each group's
        # captured veto requests stay lined up with that group's own edges.
        pending_forward: list[tuple[str, PoolRow, list[str], object]] = []

        for start in range(0, len(inputs), n):
            end = start + n
            key = keys[start]
            row = self.rows_by_key[key]
            group_completions = completions_text[start:end]
            group_advantages = advantages[start:end]
            total_groups += 1
            if self._group_is_degenerate(group_advantages):
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
                judge_for_group = _QueueingVetoCollector() if veto_flag else self.judge
                if self._repair_batch_generation:
                    pending_forward.append((key, row, failed_responses, judge_for_group))
                else:
                    _t_forward = time.perf_counter()
                    result = forward.repair_failed_group(
                        row,
                        failed_responses,
                        self.engine,
                        judge_for_group,
                        self.rng,
                        self.meter,
                        current_update,
                        veto=veto_flag,
                        distance_gate=not self.overrides.no_distance,
                    )
                    self._phase_seconds["forward_repair"] += time.perf_counter() - _t_forward
                    self._consume_forward_result(key, result, veto_flag, judge_for_group, new_edges)
                    forward_reasons.update(result.reasons)
                    forward_candidate_calls += result.candidate_calls
                    forward_distinct_candidate_calls += result.distinct_candidate_calls

        if self._repair_batch_generation and pending_forward:
            # RUN_REPAIR_BATCH_GENERATION (L2): every group collected above
            # shares one forward.repair_all_groups call -- see that
            # function's docstring (conpo/forward.py) for the batching
            # contract and the rng-seed ordering caveat this merge carries.
            veto_flag = not self.overrides.no_veto
            _t_forward = time.perf_counter()
            results = forward.repair_all_groups(
                [(row, failed_responses) for _, row, failed_responses, _ in pending_forward],
                self.engine,
                [judge_for_group for _, _, _, judge_for_group in pending_forward],
                self.rng,
                self.meter,
                current_update,
                veto=veto_flag,
                distance_gate=not self.overrides.no_distance,
            )
            self._phase_seconds["forward_repair"] += time.perf_counter() - _t_forward
            for (key, _, _, judge_for_group), result in zip(pending_forward, results):
                self._consume_forward_result(key, result, veto_flag, judge_for_group, new_edges)
                forward_reasons.update(result.reasons)
                forward_candidate_calls += result.candidate_calls
                forward_distinct_candidate_calls += result.distinct_candidate_calls

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
            # B1 (audit 2026-09-14): fraction of this step's forward-repair
            # engine.generate calls whose max_candidates proposals were
            # pairwise distinct (a max_candidates=1 call is trivially
            # distinct). Vacuously 1.0 when the step made no repair calls
            # (arm.forward off, or every group's responses were already
            # feasible). Against VLLMColocateEngine this is the regression
            # test for the seeded-duplicate bug the fix above closes: it
            # collapsed toward 0.0 whenever max_candidates > 1.
            "forward_distinct_candidate_frac": (
                (forward_distinct_candidate_calls / forward_candidate_calls) if forward_candidate_calls else 1.0
            ),
            # B2: fixed list, not step-specific -- see _SAMPLED_MECHANISM_FIELDS.
            "sampled_fields": list(_SAMPLED_MECHANISM_FIELDS),
        }
        # B2: the PREVIOUS cycle's step totals, captured at this method's
        # top before this cycle's accumulators were reset; absent on the
        # run's first cycle (previous_step_totals is None, see the top).
        if previous_step_totals is not None:
            record["previous_step_totals"] = previous_step_totals
        # Not a copy, same reasoning as "phase_seconds" above: a same-step
        # flush_pending_veto's _wake_engine() call may still touch the
        # wrapped llm.wake_up after this dict is built.
        if self._engine_seconds is not None:
            record["engine_seconds"] = self._engine_seconds
        # RUN_RESAMPLE_ZERO_STD (K5): only present when the flag is on, same
        # convention as engine_seconds/pair_stats_top_pairs above -- absent,
        # not zero, is how a flag-off run reports "not attempted" here.
        if self._resample_zero_std:
            record["resample_groups_replaced"] = resample_groups_replaced
            record["resample_rounds"] = resample_rounds
        if self.pair_stats is not None and current_update % config.PROBE_PERIOD_S == 0:
            record["pair_stats_top_pairs"] = [list(row) for row in self.pair_stats.top_pairs(5)]
        # Stashed, not written here -- see the __init__ comment on
        # _pending_mechanism_record. compute_loss merges in this step's
        # prompt_* diagnostics (prompt_gamma_theta included) and writes the
        # single record for this step.
        self._pending_mechanism_record = record
        # B1: this method runs exactly once per optimiser step (TRL's own
        # generation-batch cache, see this method's top), so one append here
        # already gives TRL's log() a per-step value, not a 16-micro-batch
        # mean the way compute_loss's own stats loop would.
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["forward_distinct_candidate_frac"].append(record["forward_distinct_candidate_frac"])

        # RUN_VLLM_PUSH_ONCE_PER_STEP (L1): run this step's deferred
        # llm.sleep() for real, if one is pending, before any of the three
        # flush_pending_veto triggers below can run -- _sleep_engine_if_enabled
        # trusts that the colocated engine is already asleep by the time a
        # burst starts (periodic check right below, _save_checkpoint, and
        # scripts/train.py's end-of-training call all fire after this method
        # has already returned for the step, so this one call site covers
        # all three).
        self._flush_deferred_vllm_sleep()

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

    @staticmethod
    def _group_is_degenerate(group_advantages: torch.Tensor) -> bool:
        """True iff every advantage in this prompt group is (numerically)
        zero -- the group's reward std was zero, so GRPO's own advantage
        normalisation leaves nothing for the policy gradient to learn from
        it. Shared by the per-group loop above and
        _resample_degenerate_groups below (RUN_RESAMPLE_ZERO_STD)."""
        return bool(torch.allclose(group_advantages, torch.zeros_like(group_advantages)))

    def _splice_group_output(
        self, output: dict[str, object], sub_output: dict[str, object], start: int, end: int
    ) -> None:
        """Overwrite output[key][start:end] with sub_output[key] for every
        per-row key a _generate_and_score_completions call may return,
        padding both to a common sequence width first: two separate calls
        each pad prompt/completion to their own max length (TRL's
        grpo_trainer.py), so a shorter side must be grown before an
        in-place slice assignment can work. Padding side mirrors TRL's own
        `pad()` calls -- left for prompt_ids/prompt_mask, right for
        everything aligned to the completion axis. `advantages` has no
        sequence dimension (one scalar per row) so it is spliced directly.
        `num_items_in_batch` (a single batch-wide scalar, not per-row) has no
        per-row slice to overwrite, so it is instead adjusted by the change
        in valid completion tokens this splice makes: TRL computes it once,
        over the whole pre-resampling batch (grpo_trainer.py's
        `_generate_and_score_completions`, ~line 2427), before this method
        ever runs, and ConPOTrainer._compute_loss uses it as the loss
        normalizer for loss_type in {dapo, cispo, vespo} -- left untouched
        here, a replaced group's different completion length would silently
        miscalibrate that normalizer for every such step (fixer review
        2026-09-13)."""
        discarded_valid = output["completion_mask"][start:end].sum()
        replacement_valid = sub_output["completion_mask"].sum()
        pad_token_id = self.processing_class.pad_token_id
        per_key_pad: dict[str, tuple[str, int | float]] = {
            "prompt_ids": ("left", pad_token_id),
            "prompt_mask": ("left", 0),
            "completion_ids": ("right", pad_token_id),
            "completion_mask": ("right", 0),
            "old_per_token_logps": ("right", 0.0),
            "ref_per_token_logps": ("right", 0.0),
            "sampling_per_token_logps": ("right", 0.0),
            "importance_sampling_ratio": ("right", 0.0),
        }
        for key, (side, pad_value) in per_key_pad.items():
            if key not in output or key not in sub_output:
                continue
            width = max(output[key].shape[-1], sub_output[key].shape[-1])
            output[key] = _pad_last_dim(output[key], width, side, pad_value)
            sub_output[key] = _pad_last_dim(sub_output[key], width, side, pad_value)
            output[key][start:end] = sub_output[key]
        output["advantages"][start:end] = sub_output["advantages"]
        output["num_items_in_batch"] = output["num_items_in_batch"] + (replacement_valid - discarded_valid)

    def _resample_degenerate_groups(
        self, inputs: list[dict[str, object]], output: dict[str, object], n: int
    ) -> tuple[int, int]:
        """RUN_RESAMPLE_ZERO_STD (K5): replace every degenerate prompt group
        in this batch with a freshly drawn, freshly generated substitute
        prompt, for up to self._resample_max_rounds extra generation
        rounds; after that cap the batch proceeds with whatever it still
        holds. `inputs` and `output` are mutated in place so the rest of
        _generate_and_score_completions (decode, advantages, the per-group
        loop's own degenerate_group_rate) sees the replacement rather than
        the original. Returns (groups_replaced, rounds_run) for the
        mechanism-log record.
        """
        groups_replaced = 0
        rounds_run = 0
        for _ in range(self._resample_max_rounds):
            degenerate_starts = [
                start
                for start in range(0, len(inputs), n)
                if self._group_is_degenerate(output["advantages"][start : start + n])
            ]
            if not degenerate_starts:
                break
            rounds_run += 1
            for start in degenerate_starts:
                end = start + n
                current_keys = {str(row["key"]) for row in inputs}
                candidates = [key for key in self.rows_by_key if key not in current_keys]
                if not candidates:
                    continue
                new_row = self.rows_by_key[self.rng.choice(candidates)]
                sub_inputs = [{"prompt": [{"role": "user", "content": new_row.prompt}], "key": new_row.key}] * n
                sub_output = super()._generate_and_score_completions(sub_inputs)
                self._splice_group_output(output, sub_output, start, end)
                inputs[start:end] = sub_inputs
                groups_replaced += 1
        return groups_replaced, rounds_run

    def _consume_forward_result(
        self,
        key: str,
        result: forward.RepairResult,
        veto_flag: bool,
        judge_for_group: object,
        new_edges: list[Edge],
    ) -> None:
        """Route one group's forward.RepairResult into feasible_buffer, the
        pending veto queue, or new_edges -- exactly what
        _generate_and_score_completions's per-group loop always did inline,
        factored out (RUN_REPAIR_BATCH_GENERATION, L2) so both the per-group
        forward.repair_failed_group call (flag off) and the merged
        forward.repair_all_groups call (the default) consume their
        RepairResult identically. `judge_for_group` is the same
        `_QueueingVetoCollector` this group's repair call was given when
        `veto_flag` is True (unused, since collector-only, when it is
        False)."""
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
            # _QueueingVetoCollector's docstring). judge_for_group is a
            # _QueueingVetoCollector here: veto_flag is only True when the
            # call site built it as one (see pending_forward's own comment).
            captured = judge_for_group.captured  # type: ignore[attr-defined]
            for edge, req in zip(result.edges, captured):
                self._push_pending(PendingPair(edge=edge, request=req, direction="forward"))
        else:
            new_edges.extend(result.edges)
            for endpoint in result.feasible_endpoints:
                self.feasible_buffer.add(key, endpoint)

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
        # B5 (iii, audit 2026-09-14): _generate_and_score_completions's own
        # call site for this already flushes any deferred vLLM sleep before
        # its three triggers (periodic, _save_checkpoint, end of training),
        # but _push_pending's queue-overflow trigger above can call this
        # method mid-step, before that call site is reached -- flushing here
        # too makes every trigger safe regardless of call order, not only
        # the three that go through _generate_and_score_completions. No-op
        # when nothing is pending (_flush_deferred_vllm_sleep's own guard).
        self._flush_deferred_vllm_sleep()
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
        if torch.cuda.is_available():
            # Release the caching allocator's reserved-but-free blocks so the
            # judge child (a separate CUDA context) sees them as free: the
            # end-of-training flush found only 39 of 79 GiB free next to the
            # finished trainer and its judge died at vLLM start-up
            # (2026-09-11); VLLMBatchedJudge._fit_fraction adapts to what is
            # left after this.
            gc.collect()
            torch.cuda.empty_cache()
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
        """Append every judged pair of this burst to `<out_dir>/judge_replies.jsonl`.

        Extended 2026-09-11 from design 2026-09-04 item 3's 8-pair-per-burst
        sample to a full per-burst audit log: a rejected pair never reaches
        edge_buffer, so its texts existed nowhere else once dropped from
        pending_queue, leaving no fixed corpus to A/B test a judge-prompt
        change against. Every record now also carries the full texts the
        judge saw -- `prompt`, `y_plus`, `y_minus`, and the exact rendered
        `judge_request` -- plus `y_plus_label` ("A" or "B"): build_veto_prompt
        swaps which letter carries y_plus per request (judge.py's
        build_veto_prompt docstring), so a reader of `raw` otherwise cannot
        tell which letter the judge's verdict was about. `judge_request` is
        reproduced via build_veto_prompt(pair.request) rather than stored on
        PendingPair: build_veto_prompt is a pure function of `request`'s own
        fields (deterministic per pair, same swap decision every call -- see
        judge.py's _prompt_seed), so this is exactly the string the judge
        was sent. `step` (already present below) doubles as the burst index:
        one flush writes one contiguous run of records at the same step.
        No-op when `out_dir` was never given to the constructor.
        """
        if self.out_dir is None:
            return
        path = self.out_dir / "judge_replies.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for pair, verdict in zip(pairs, verdicts):
                judge_request = build_veto_prompt(pair.request)
                y_plus_label = veto_plus_label(pair.request)
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
                    "prompt": pair.request.prompt,
                    "y_plus": pair.request.y_plus,
                    "y_minus": pair.request.y_minus,
                    "judge_request": judge_request,
                    "y_plus_label": y_plus_label,
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
        RUN_VLLM_PUSH_ONCE_PER_STEP (L1, default on) defers that per-call
        sleep instead of dropping it: `_generate_and_score_completions`
        calls `_flush_deferred_vllm_sleep()` before every trigger below, so
        the engine is still genuinely asleep by the time this runs.

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

    def _install_vllm_reload_guard(self) -> None:
        """Wrap `self.vllm_generation.llm.collective_rpc` so the policy is
        the last thing written into the vLLM engine before every
        `generate()` call, whatever TRL does around it.

        Root cause (verified 2026-09-11, all line numbers against the
        installed trl==1.9.2 / vllm==0.26.0): `VLLMGeneration.generate`
        (trl/generation/vllm_generation.py lines 540-548) wakes the engine's
        weights and, with sleep mode on, calls
        `self.llm.collective_rpc("reload_weights")` on every single call
        ("work around for vllm issue 29341") before sampling, and ends with
        `self.llm.sleep(level=2)` (line 702). `reload_weights` with no
        arguments (`vllm/v1/worker/gpu_worker.py` line 444 ->
        `vllm/v1/worker/gpu_model_runner.py` line 5480) reloads the
        *original* model from disk, unconditionally. `sync_weights`
        (vllm_generation.py lines 444-484) had already merged the PEFT
        adapter and pushed every parameter earlier in the same optimiser
        step (trl/trainer/grpo_trainer.py lines 1791-1793, gated on
        `_last_loaded_step`), so every `generate()` call was silently
        replacing that pushed policy with the base model before sampling
        from it -- GRPO rollouts, ConPO's forward repair, and its reverse
        corruption all sample through this same path
        (`conpo/engine.py`'s `VLLMColocateEngine.generate`), several times
        per step.

        The fix: after TRL's own `collective_rpc("reload_weights")` call
        runs (unchanged -- this does not remove TRL's workaround for vllm
        issue 29341, nor change any sleep level), also call
        `sync_weights()`, which wakes weights idempotently, re-merges the
        adapter, pushes every parameter, and unmerges. Cost: one extra
        merge/push/unmerge per `generate()` call (a few seconds), several
        times per optimiser step. Cheaper alternative not taken: sleep
        level 1 (keeps weights resident, only frees the KV cache) plus
        skipping TRL's `reload_weights` call entirely -- rejected because
        it means patching around TRL's own upstream workaround rather than
        living with it, and level-1 sleep frees less GPU memory during the
        veto burst that `_sleep_engine_if_enabled` relies on.

        Installs only once (idempotent: checked via a marker attribute on
        `llm` itself, not on `self`, since the object being wrapped is what
        must not be double-wrapped) and only when a colocated vLLM engine
        with a callable `collective_rpc` exists: a no-op for
        RUN_USE_VLLM=0 (no `self.vllm_generation` at all -- trl only sets
        it when `self.use_vllm`), for `vllm_mode="server"` (no `self.llm`,
        only `self.vllm_client`), and for HFEngine/FakeEngine runs.

        Re-entrant by construction, not by special-casing: the wrapper only
        reacts to the method name `"reload_weights"`, so any other call
        `collective_rpc` receives (including one `sync_weights()` itself
        might make, e.g. `"load_weights"`) is forwarded to the original
        `collective_rpc` unchanged and does not re-trigger a resync.

        Perf-pass instrumentation (Ethan devbox, 2026-09-11), logging only --
        no change to what TRL/vLLM do: also wraps
        `vllm_generation.generate` and `llm.wake_up`/`llm.sleep` (installed
        together with the reload guard above, same marker, same early
        returns) to fill `self._engine_seconds`, so the per-step mechanism
        record can show where a step's time goes inside this path.
        `VLLMGeneration.generate`'s colocate-mode body (vllm_generation.py
        lines 536-548, 650-702) always runs, in order: `llm.wake_up(tags=
        ["weights"])`, `collective_rpc("reload_weights")` (the guard above),
        then, only under TP > 1, a distributed gather; `llm.wake_up(tags=
        ["kv_cache"])`; the actual `llm.generate(...)` sampling call (not
        wrapped -- vLLM's own `LLM.generate`, a different method from
        `vllm_generation.generate`); `llm.sleep(level=2)`. So the call
        (`tags=["kv_cache"])`'s wake_up marks the boundary between
        "everything before sampling" and "sampling": `wake_reload_resync`
        accumulates wall time from `vllm_generation.generate`'s own entry
        through that call's return (covering the weights wake, the reload
        rpc, the guard's resync above, and the kv_cache wake_up call
        itself); `generate_only` accumulates from that same return to
        `llm.sleep`'s start; `sleep` accumulates time spent inside
        `llm.sleep` itself. `generate_calls` counts every
        `vllm_generation.generate` call (TRL's own rollout and every
        `VLLMColocateEngine.generate` call alike, conpo/engine.py -- the
        same object either way); `reload_calls` counts every
        `"reload_weights"` the guard above resynced after.
        `enable_sleep_mode=False` (vllm_enable_sleep_mode off) means TRL
        never calls wake_up/collective_rpc/sleep at all: no kv_cache wake
        ever fires that call, so the whole call is attributed to
        `generate_only` instead (the fallback branch in `_timed_generate`
        below) rather than left unmeasured.
        # ponytail: an exception raised between the kv_cache wake_up and
        # llm.sleep (mid-sampling) leaves that slice of wall time
        # unattributed -- accept the undercount for a logging-only counter
        # rather than a try/finally that would also have to guess which
        # bucket an interrupted call belongs in.
        `wake_up`/`sleep`/`generate` are wrapped defensively (`getattr` +
        `callable`, independently of each other): a caller-supplied fake
        `llm`/`vllm_generation` (tests/test_vllm_reload_guard.py's minimal
        `_FakeLLM`, which predates this change and defines only
        `collective_rpc`) still gets the reload guard with no
        AttributeError, only no engine_seconds timing.
        """
        vllm_generation = getattr(self, "vllm_generation", None)
        llm = getattr(vllm_generation, "llm", None)
        original_collective_rpc = getattr(llm, "collective_rpc", None)
        if llm is None or not callable(original_collective_rpc):
            return
        if getattr(llm, "_conpo_reload_guard_installed", False):
            return

        self._engine_seconds = {
            "wake_reload_resync": 0.0,
            "generate_only": 0.0,
            "sleep": 0.0,
            "generate_calls": 0,
            "reload_calls": 0,
            "reload_calls_skipped": 0,
        }
        # Per-call markers the wake_up/sleep/generate wrappers below hand off
        # through: generate() calls never overlap (one training process, no
        # concurrent sampling), so instance attributes are enough state.
        self._engine_gen_t0: float | None = None
        self._engine_kv_wake_done_t: float | None = None
        self._engine_kv_wake_seen = False
        # RUN_VLLM_PUSH_ONCE_PER_STEP (L1, default on): once this step's
        # policy is pushed into vLLM, every later generate() call in the
        # same step skips TRL's own reload_weights RPC and our resync
        # (_reload_then_resync below), and defers llm.sleep() instead of
        # running it (_timed_sleep below) so weights and the KV cache stay
        # resident -- there is nothing to reload from. Read once here, not
        # per call: nothing in a run changes it after this trainer is built.
        self._vllm_push_once = config.env("RUN_VLLM_PUSH_ONCE_PER_STEP", "1") == "1"
        self._vllm_synced_this_step = False
        self._vllm_sleep_pending = False
        # True only while _flush_deferred_vllm_sleep below is running the
        # deferred sleep for real, so _timed_sleep's push-once branch
        # doesn't defer that same call back onto itself.
        self._vllm_flushing_deferred_sleep = False

        def _reload_then_resync(method: str, *args: Any, **kwargs: Any) -> Any:
            if method == "reload_weights" and self._vllm_push_once and self._vllm_synced_this_step:
                self._engine_seconds["reload_calls_skipped"] += 1  # type: ignore[index]
                return None
            result = original_collective_rpc(method, *args, **kwargs)
            if method == "reload_weights":
                vllm_generation.sync_weights()
                self._engine_seconds["reload_calls"] += 1  # type: ignore[index]
                self._vllm_synced_this_step = True
            return result

        llm.collective_rpc = _reload_then_resync
        llm._conpo_reload_guard_installed = True

        original_wake_up = getattr(llm, "wake_up", None)
        original_sleep = getattr(llm, "sleep", None)
        original_generate = getattr(vllm_generation, "generate", None)

        if callable(original_wake_up):

            def _timed_wake_up(*args: Any, **kwargs: Any) -> Any:
                tags = kwargs.get("tags") or (args[0] if args else None)
                is_kv_cache_wake = self._engine_gen_t0 is not None and bool(tags) and "kv_cache" in tags
                result = original_wake_up(*args, **kwargs)
                if is_kv_cache_wake:
                    now = time.perf_counter()
                    self._engine_seconds["wake_reload_resync"] += (  # type: ignore[index]
                        now - self._engine_gen_t0
                    )
                    self._engine_kv_wake_done_t = now
                    self._engine_kv_wake_seen = True
                return result

            llm.wake_up = _timed_wake_up

        if callable(original_sleep):

            def _timed_sleep(*args: Any, **kwargs: Any) -> Any:
                t0 = time.perf_counter()
                if self._engine_kv_wake_done_t is not None:
                    self._engine_seconds["generate_only"] += t0 - self._engine_kv_wake_done_t  # type: ignore[index]
                    self._engine_kv_wake_done_t = None
                if self._vllm_push_once and not self._vllm_flushing_deferred_sleep:
                    # Defer: weights and the KV cache stay resident so the
                    # next generate() call in this step has nothing to
                    # reload. _flush_deferred_vllm_sleep runs the real sleep
                    # (through this same wrapper) before anything that needs
                    # the engine actually asleep.
                    self._vllm_sleep_pending = True
                    return None
                result = original_sleep(*args, **kwargs)
                self._engine_seconds["sleep"] += time.perf_counter() - t0  # type: ignore[index]
                return result

            llm.sleep = _timed_sleep

        if callable(original_generate):

            def _timed_generate(*args: Any, **kwargs: Any) -> Any:
                self._engine_seconds["generate_calls"] += 1  # type: ignore[index]
                self._engine_gen_t0 = time.perf_counter()
                self._engine_kv_wake_seen = False
                self._engine_kv_wake_done_t = None
                try:
                    return original_generate(*args, **kwargs)
                finally:
                    if not self._engine_kv_wake_seen:
                        # Sleep mode disabled this call: no wake/reload/sleep
                        # boundary fired, so the whole call is generation work.
                        self._engine_seconds["generate_only"] += (  # type: ignore[index]
                            time.perf_counter() - self._engine_gen_t0
                        )
                    self._engine_gen_t0 = None

            vllm_generation.generate = _timed_generate

    def _flush_deferred_vllm_sleep(self) -> None:
        """RUN_VLLM_PUSH_ONCE_PER_STEP (L1): run this step's deferred
        `llm.sleep(level=2)` for real, if `_timed_sleep` above deferred one.

        Goes through the same wrapped `llm.sleep` (`_vllm_flushing_deferred_sleep`
        set while it runs, so `_timed_sleep` executes the real call instead of
        deferring again), so `sleep`/`generate_only` timing stays accounted
        for exactly as it is without push-once. No-op with the flag off or
        with no colocated vLLM engine (`vllm_generation` never set, or
        nothing pending). Must run before anything that trusts the engine is
        already asleep -- see the call site in `_generate_and_score_completions`
        and `_sleep_engine_if_enabled`'s own docstring.
        """
        if not self._vllm_sleep_pending:
            return
        self._vllm_flushing_deferred_sleep = True
        try:
            self.vllm_generation.llm.sleep(level=2)  # type: ignore[attr-defined]
        finally:
            self._vllm_flushing_deferred_sleep = False
        self._vllm_sleep_pending = False

    def _save_checkpoint(self, model: torch.nn.Module, trial: object) -> None:
        """Flush before every checkpoint save (USER DECISIONS 2026-09-02),
        so a saved edges.jsonl / resumed run never carries unscored pairs.
        Also persists pair_stats (H6 part B) so a resumed run's phi/count
        table survives the restart instead of starting cold, and, after
        TRL's own save, the rest of ConPO's state (RUN_RESUME) into the same
        checkpoint directory super()._save_checkpoint created."""
        self.flush_pending_veto()
        self.save_pair_stats()
        super()._save_checkpoint(model, trial)
        checkpoint_dir = Path(self._get_output_dir(trial)) / f"checkpoint-{self.state.global_step}"
        self._save_conpo_state(checkpoint_dir)

    _CONPO_STATE_FILENAME = "conpo_state.json"
    _CONPO_EDGES_FILENAME = "conpo_edges.jsonl"
    _CONPO_PAIR_STATS_FILENAME = "conpo_pair_stats.json"

    def _save_conpo_state(self, checkpoint_dir: Path) -> None:
        """Persist every piece of ConPO's own mutable state (outside TRL)
        into `checkpoint_dir`, alongside TRL's own checkpoint files
        (RUN_RESUME, scripts/train.py): the edge buffer, feasible buffer,
        family/selection stats, PairStats (RUN_MULTI only), the TokenMeter's
        totals, cumulative counters, the rng state, and cumulative wall time.

        Called from _save_checkpoint, strictly after flush_pending_veto has
        already drained pending_queue and resolved every pending-feasible
        tracker (see flush_pending_veto's own docstring): pending_queue and
        _pending_feasible_by_edge_id are therefore always empty here and
        need no snapshot, and _steps_since_flush is always 0 right after a
        flush -- none of the three need persisting.
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.edge_buffer.save(checkpoint_dir / self._CONPO_EDGES_FILENAME)
        if self.pair_stats is not None:
            self.pair_stats.save(checkpoint_dir / self._CONPO_PAIR_STATS_FILENAME)
        version, internal_state, gauss_next = self.rng.getstate()
        state: dict[str, object] = {
            "meter_budget": self.meter.budget,
            "meter_total": self.meter.total,
            "family_stats": self.stats.state_dict(),
            "feasible_buffer": self.feasible_buffer.state_dict(),
            "cumulative_admitted_edges": self.cumulative_admitted_edges,
            "cumulative_reverse_admitted": self._cumulative_reverse_admitted,
            "cumulative_multi_reverse_admitted": self._cumulative_multi_reverse_admitted,
            "veto_seen_by_direction": dict(self._veto_seen_by_direction),
            "veto_rejected_by_direction": dict(self._veto_rejected_by_direction),
            "rng_state": [version, list(internal_state), gauss_next],
            "wall_seconds": self.cumulative_wall_seconds(),
        }
        (checkpoint_dir / self._CONPO_STATE_FILENAME).write_text(json.dumps(state), encoding="utf-8")

    def _load_conpo_state(self, checkpoint_dir: Path) -> None:
        """Inverse of _save_conpo_state, called once from __init__ when
        `resume_from` is given. Fails loud when conpo_state.json is missing:
        TRL's own resume would otherwise silently restore model/optimiser/
        scheduler/rng/global_step and restart ConPO's own mechanism cold,
        with an empty edge buffer and reset selection stats.
        """
        state_path = checkpoint_dir / self._CONPO_STATE_FILENAME
        if not state_path.exists():
            raise FileNotFoundError(
                f"RUN_RESUME=1 but {state_path} is missing: this checkpoint has no ConPO state to "
                "resume from (a TRL-only resume would silently restart the edge buffer, selection "
                "stats and token budget empty)."
            )
        state = json.loads(state_path.read_text(encoding="utf-8"))
        edges_path = checkpoint_dir / self._CONPO_EDGES_FILENAME
        if edges_path.exists():
            self.edge_buffer.load(edges_path)
        self.meter.budget = state["meter_budget"]
        self.meter.total = state["meter_total"]
        self.stats.load_state_dict(state["family_stats"])
        self.feasible_buffer.load_state_dict(state["feasible_buffer"])
        self.cumulative_admitted_edges = state["cumulative_admitted_edges"]
        self._cumulative_reverse_admitted = state["cumulative_reverse_admitted"]
        self._cumulative_multi_reverse_admitted = state["cumulative_multi_reverse_admitted"]
        self._veto_seen_by_direction = Counter(state["veto_seen_by_direction"])
        self._veto_rejected_by_direction = Counter(state["veto_rejected_by_direction"])
        version, internal_state, gauss_next = state["rng_state"]
        self.rng.setstate((version, tuple(internal_state), gauss_next))
        self.wall_seconds_before_resume = state["wall_seconds"]
        pair_stats_path = checkpoint_dir / self._CONPO_PAIR_STATS_FILENAME
        if self.pair_stats is not None and pair_stats_path.exists():
            self.pair_stats.load(pair_stats_path)

        trainer_state_path = checkpoint_dir / "trainer_state.json"
        resumed_step = None
        if trainer_state_path.exists():
            resumed_step = json.loads(trainer_state_path.read_text(encoding="utf-8"))["global_step"]
        record = {
            "event": "resume",
            "checkpoint": str(checkpoint_dir),
            "resumed_from_step": resumed_step,
            "edge_buffer_size": len(self.edge_buffer),
            "tokens_cumulative": self.meter.total,
        }
        with open(self.mechanism_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def cumulative_wall_seconds(self) -> float:
        """Wall-clock seconds spent training this logical run, across every
        RUN_RESUME process: whatever a prior process's conpo_state.json
        recorded, plus this process's own elapsed time since construction."""
        return self.wall_seconds_before_resume + (time.monotonic() - self._process_wall_start)

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

    def _score_edge_pairs(
        self, model: torch.nn.Module, edges: Sequence[Edge]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Delta_theta(e) (ConPO.pdf section 2.4) and age(e) for a batch of edges.

        One policy forward pass per response (2 per edge, gradient-tracked);
        logp_ref_*/len_* come from each edge's cache, filled once at
        admission by `_generate_and_score_completions`. The third return
        value is `{}` here; it exists so the RUN_TOKEN_LOGRATIO_CLAMP branch
        below can report resp_token_clamp_frac through the same signature.

        ponytail: one edge at a time (2*len(edges) separate forward calls,
        batch size 1 each) instead of padding the batch into one call; add
        batched scoring if per-step throughput on the real GPU run measures
        as a bottleneck.
        """
        if self._pair_objective == "span_local":
            return self._score_edge_pairs_span_local(model, edges)
        if self._token_logratio_clamp is not None:
            return self._score_edge_pairs_clamped(model, edges, self._token_logratio_clamp)

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
        return q_plus - q_minus, ages_t, {}

    def _score_edge_pairs_clamped(
        self, model: torch.nn.Module, edges: Sequence[Edge], clamp: float
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """_score_edge_pairs under RUN_TOKEN_LOGRATIO_CLAMP (K6).

        Per response: a gradient-tracked policy forward and a fresh no-grad
        reference forward over the identical tokenized ids (_token_clamped_q
        below), so the per-token policy-minus-reference log-ratio lines up
        token for token before losses.clamp_token_logratio clamps it. No
        cache reuse -- edge.logp_ref_plus/minus and len_plus/minus are
        unused on this path, unlike the cached path above -- doubling the
        per-edge reference-model forward cost; acceptable since it only
        runs when the flag is set.
        """
        q_plus_list, q_minus_list, ages, clamp_hits = [], [], [], []
        for edge in edges:
            q_plus, hits_plus = self._token_clamped_q(model, edge.prompt, edge.y_plus, clamp)
            q_minus, hits_minus = self._token_clamped_q(model, edge.prompt, edge.y_minus, clamp)
            q_plus_list.append(q_plus)
            q_minus_list.append(q_minus)
            clamp_hits.append(hits_plus)
            clamp_hits.append(hits_minus)
            ages.append(self.edge_buffer.age(edge, self.state.global_step))
        device = q_plus_list[0].device
        ages_t = torch.tensor(ages, dtype=torch.long, device=device)
        all_hits = torch.cat(clamp_hits)
        resp_stats = {"token_clamp_frac": all_hits.float().mean().item() if all_hits.numel() else 0.0}
        return torch.stack(q_plus_list) - torch.stack(q_minus_list), ages_t, resp_stats

    def _token_clamped_q(
        self, model: torch.nn.Module, prompt_text: str, response: str, clamp: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q_theta(x, y) with its per-token policy-minus-reference log-ratio
        clamped to +/-`clamp` (RUN_TOKEN_LOGRATIO_CLAMP, K6), plus which
        completion tokens hit the clamp (pooled by the caller into
        resp_token_clamp_frac). Policy and reference are scored on the same
        tokenized ids (_tokenize_prompt_response), so their per-token
        log-probs line up position for position.
        """
        input_ids, attention_mask, completion_mask = _tokenize_prompt_response(
            self.processing_class, prompt_text, response
        )
        device = next(model.parameters()).device
        input_ids, attention_mask, completion_mask = (
            input_ids.to(device),
            attention_mask.to(device),
            completion_mask.to(device),
        )
        _, policy_tok = losses.sequence_logprobs_with_tokens(model, input_ids, attention_mask, completion_mask)
        with self._reference_model(model) as ref_model:
            _, ref_tok = losses.sequence_logprobs_with_tokens(ref_model, input_ids, attention_mask, completion_mask)
        mask = completion_mask[:, 1:].bool()
        diff = (policy_tok - ref_tok)[mask]
        clamped_sum, hits = losses.clamp_token_logratio(diff, clamp)
        length = losses.completion_lengths(completion_mask)
        q = losses.q_theta_from_logratio(clamped_sum.unsqueeze(0), length)[0]
        return q, hits

    def _response_logratios(
        self, model: torch.nn.Module, prompt_text: str, response: str
    ) -> tuple[torch.Tensor, list[int]]:
        """One response's per-token policy-minus-reference log-ratio, plus
        its own token ids in the same left-to-right order (RUN_PAIR_OBJECTIVE
        span_local, K8): `losses.diff_span` slices both consistently since
        they come from masking the identical axis of the identical
        tokenized ids. Same tokenize-once/policy-and-reference-forward-twice
        shape as `_token_clamped_q` above -- span_local always needs the
        reference model's PER-TOKEN log-probs (not the cached sequence-level
        scalar on the edge), so this always pays that method's documented
        doubled reference-forward cost, unconditionally whenever span_local
        is the active objective (not only when RUN_TOKEN_LOGRATIO_CLAMP is
        also set).

        ponytail: one response at a time, same shape as _token_clamped_q and
        _score_edge_pairs' own "one edge at a time" note; share a
        tokenize+forward helper across all three if a fourth caller needs
        this shape.
        """
        input_ids, attention_mask, completion_mask = _tokenize_prompt_response(
            self.processing_class, prompt_text, response
        )
        device = next(model.parameters()).device
        input_ids, attention_mask, completion_mask = (
            input_ids.to(device),
            attention_mask.to(device),
            completion_mask.to(device),
        )
        _, policy_tok = losses.sequence_logprobs_with_tokens(model, input_ids, attention_mask, completion_mask)
        with self._reference_model(model) as ref_model:
            _, ref_tok = losses.sequence_logprobs_with_tokens(ref_model, input_ids, attention_mask, completion_mask)
        mask = completion_mask[:, 1:].bool()
        diff = (policy_tok - ref_tok)[mask]
        response_ids = input_ids[0][completion_mask[0].bool()].tolist()
        return diff, response_ids

    def _score_edge_pairs_span_local(
        self, model: torch.nn.Module, edges: Sequence[Edge]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """_score_edge_pairs under RUN_PAIR_OBJECTIVE=span_local (K8): per
        edge, the margin is the span-normalised mean over ONLY the tokens
        where y_plus and y_minus differ (`losses.diff_span`), instead of the
        whole-sequence length-normalised sum every other path uses -- the
        tokens a near-identical counterfactual pair shares carry no
        gradient at all rather than a diluted or cancelling one (yesterday's
        likelihood-displacement diagnosis). RUN_TOKEN_LOGRATIO_CLAMP is
        reused as the span-token clamp when set.

        Sets `self._pending_span_anchor` as a side channel to compute_loss's
        resp block (RUN_POS_ANCHOR_WEIGHT term) instead of returning it,
        mirroring `self._pending_mechanism_record`'s existing across-call
        handoff in this class: every `_score_edge_pairs*` branch and both of
        its callers (`compute_loss`, `_run_probe`) keep relying on the same
        3-tuple contract.

        ponytail: one edge at a time, same as the sequence/clamped paths
        above; always pays 2 reference-model forwards per response (4 per
        edge total) regardless of RUN_TOKEN_LOGRATIO_CLAMP, since span_local
        needs per-token reference log-probs unconditionally -- same cost
        class `_token_clamped_q` already documents, unconditional here.
        """
        margins, ages, anchors, hits = [], [], [], []
        span_lens_plus, span_lens_minus, empty_flags = [], [], []
        for edge in edges:
            diff_plus, ids_plus = self._response_logratios(model, edge.prompt, edge.y_plus)
            diff_minus, ids_minus = self._response_logratios(model, edge.prompt, edge.y_minus)
            (start_plus, end_plus), (start_minus, end_minus) = losses.diff_span(ids_plus, ids_minus)
            margin, anchor, span_hits = losses.span_local_margin_and_anchor(
                diff_plus[start_plus:end_plus], diff_minus[start_minus:end_minus], self._token_logratio_clamp
            )
            margins.append(margin)
            anchors.append(anchor)
            hits.append(span_hits)
            span_lens_plus.append(end_plus - start_plus)
            span_lens_minus.append(end_minus - start_minus)
            empty_flags.append(end_plus == start_plus and end_minus == start_minus)
            ages.append(self.edge_buffer.age(edge, self.state.global_step))
        device = margins[0].device
        ages_t = torch.tensor(ages, dtype=torch.long, device=device)
        self._pending_span_anchor = torch.stack(anchors)
        all_hits = torch.cat(hits)
        resp_stats = {
            "token_clamp_frac": all_hits.float().mean().item() if all_hits.numel() else 0.0,
            "span_len_plus_mean": sum(span_lens_plus) / len(span_lens_plus),
            "span_len_minus_mean": sum(span_lens_minus) / len(span_lens_minus),
            "empty_span_frac": sum(empty_flags) / len(empty_flags),
        }
        return torch.stack(margins), ages_t, resp_stats

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
        replay-invariant too (x^-A depends only on the static `row` and
        `edge.families`, both fixed for the edge's whole lifetime), so
        RUN_CACHE_REF_LOGPROBS (L3, default on) computes them once under
        `disable_adapter()` and caches them onto the edge for every later
        draw; "0" recomputes them fresh on every call, as before.
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
            if self._cache_ref_logprobs and edge.logp_ref_plus_xminus is not None:
                ref_plus_xminus = torch.tensor(
                    edge.logp_ref_plus_xminus, dtype=lp_plus_xminus.dtype, device=lp_plus_xminus.device
                )
                ref_minus_xminus = torch.tensor(
                    edge.logp_ref_minus_xminus, dtype=lp_minus_xminus.dtype, device=lp_minus_xminus.device
                )
            else:
                with self._reference_model(model) as ref_model:
                    ref_plus_xminus, _ = score_response(ref_model, self.processing_class, x_minus, edge.y_plus)
                    ref_minus_xminus, _ = score_response(ref_model, self.processing_class, x_minus, edge.y_minus)
                if self._cache_ref_logprobs:
                    object.__setattr__(edge, "logp_ref_plus_xminus", ref_plus_xminus.item())
                    object.__setattr__(edge, "logp_ref_minus_xminus", ref_minus_xminus.item())
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
            delta, ages, _ = self._score_edge_pairs(model, probe_edges)
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
        # RUN_PAIR_SCALE_AUTO (see __init__): current_gradient_accumulation_steps
        # is set by transformers.Trainer before every training compute_loss
        # call (Trainer._inner_training_loop); the static config value is only
        # a fallback for a compute_loss call made outside that loop (e.g. a
        # unit test constructing ConPOTrainer directly).
        pair_scale_factor = (
            1.0 / getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)
            if self._pair_scale_auto
            else 1.0
        )
        stats["resp_pair_scale"] = pair_scale_factor
        # Perf-pass instrumentation (Ethan devbox, 2026-09-11): wall time and
        # forward-pass count for this micro-batch's own pair-loss scoring
        # below (_score_edge_pairs, _prompt_margins), one model(...) forward
        # per score_response call at batch size 1 each (conpo/losses.py's
        # sequence_logprobs). Local to this micro-batch: "stats" below still
        # only reaches the mechanism record's own resp_evicted/
        # pair_scoring_seconds/pair_scoring_forwards keys (sampled_fields)
        # from the one micro-batch that finds _pending_mechanism_record set
        # (see the write below) -- B2 (audit 2026-09-14) adds the real
        # per-optimiser-step totals alongside them via the
        # self._*_step_accum accumulators below, surfaced a cycle later as
        # previous_step_totals (_generate_and_score_completions).
        pair_scoring_seconds = 0.0
        pair_scoring_forwards = 0

        # config.EDGE_PAIRS_PER_STEP / PROMPT_PAIRS_PER_STEP are per-OPTIMISER-step
        # budgets (config.py), but compute_loss runs once per micro-batch under
        # gradient accumulation; split the budget so the per-optimiser-step total
        # sampled from edge_buffer stays at the configured value regardless of
        # accumulation, rather than being drawn in full on every micro-batch.
        # B3 (audit 2026-09-14): the old math.ceil(total / grad_accum) drawn on
        # EVERY micro-batch overdrew whenever grad_accum did not divide total
        # evenly -- PROMPT_PAIRS_PER_STEP=8 over grad_accum=16 drew
        # ceil(8/16)=1 per micro-batch, 16 per step, double the configured 8
        # (EDGE_PAIRS_PER_STEP=48/16=3 stayed exact only because 16 divides
        # 48). _pairs_this_microbatch's floor((i+1)*T/A)-floor(i*T/A) split
        # sums to exactly T over one A-micro-batch cycle for any T/A.
        grad_accum = self.args.gradient_accumulation_steps
        edge_n = _pairs_this_microbatch(config.EDGE_PAIRS_PER_STEP, self._microbatch_in_cycle, grad_accum)
        prompt_n = _pairs_this_microbatch(config.PROMPT_PAIRS_PER_STEP, self._microbatch_in_cycle, grad_accum)
        self._microbatch_in_cycle += 1
        edge_pairs_drawn = 0
        prompt_pairs_drawn = 0

        # Design choice (orchestrator, 2026-09-04): L_resp stays UNGATED on
        # Overrides.multi. Forward multi-family repair edges are part of
        # every arm's edge stream and the hub arm "full" must not change;
        # H6 isolates the phi-selected reverse multi-family corruption edges
        # plus their prompt term (L_prompt below is the only term gated by
        # overrides.multi).
        if self.arm.resp_loss and len(self.edge_buffer) > 0:
            resp_edges = self.edge_buffer.sample(edge_n, current_update, self.rng, balanced=True)
            edge_pairs_drawn = len(resp_edges)
            if resp_edges:
                _t_score = time.perf_counter()
                delta, ages, token_clamp_stats = self._score_edge_pairs(model, resp_edges)
                # Saturated-edge eviction: an edge whose margin already clears
                # kappa*tau carries weight 0 in resp_loss below (same test)
                # forever after, since delta only grows as training proceeds,
                # so it wastes a sample() draw and a score_edge_pairs forward
                # every future step until age_cap retires it. Drop it now.
                # B5 (ix, audit 2026-09-14): ConPO.pdf's Lemma 7 only states
                # that w(e) retires to 0 once Delta_theta(e) > kappa*tau;
                # physically discarding the edge from edge_buffer here
                # (rather than leaving it at weight 0 until age_cap) is this
                # implementation's own decision, not something the paper
                # specifies.
                saturated_edges = [
                    edge
                    for edge, margin in zip(resp_edges, delta.detach().tolist())
                    if margin > config.KAPPA * config.TAU
                ]
                stats["resp_evicted"] = self.edge_buffer.discard(saturated_edges)
                self._resp_evicted_step_accum += stats["resp_evicted"]
                pair_scoring_seconds += time.perf_counter() - _t_score
                # score_response(y_plus), score_response(y_minus); doubled by
                # RUN_TOKEN_LOGRATIO_CLAMP's fresh per-edge reference forward
                # (_score_edge_pairs_clamped) or by RUN_PAIR_OBJECTIVE=span_local
                # (_score_edge_pairs_span_local, K8), which the cached path
                # above skips and which span_local pays unconditionally.
                pair_scoring_forwards += (
                    4 if self._token_logratio_clamp is not None or self._pair_objective == "span_local" else 2
                ) * len(resp_edges)
                if self._pair_objective == "span_local":
                    loss_resp, resp_stats = losses.span_local_resp_loss(
                        delta,
                        ages,
                        self._pending_span_anchor,
                        tau=config.TAU,
                        gamma=config.GAMMA,
                        kappa=config.KAPPA,
                        pos_anchor_weight=self._pos_anchor_weight,
                    )
                    self._pending_span_anchor = None
                else:
                    loss_resp, resp_stats = losses.resp_loss(
                        delta, ages, tau=config.TAU, gamma=config.GAMMA, kappa=config.KAPPA
                    )
                total = total + pair_scale_factor * config.PAIR_LOSS_WEIGHT * loss_resp
                stats["resp_loss"] = loss_resp.item()
                stats["resp_mean_age"] = ages.float().mean().item()
                stats.update({f"resp_{k}": v for k, v in resp_stats.items()})
                stats.update({f"resp_{k}": v for k, v in token_clamp_stats.items()})

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
            prompt_pairs_drawn = len(prompt_edges)
            if prompt_edges:
                # The margins and prompt_gamma_theta diagnostic are computed
                # (and logged) whenever the arm carries L_prompt at all,
                # independent of overrides.lambda_p: the H5 ablation
                # (RUN_LAMBDA_P=0) needs prompt_gamma_theta on record to show
                # what L_prompt would have measured had it not been zeroed
                # out. Only the weighted contribution to `total` is gated on
                # lambda_p > 0 -- that is what "ablates L_prompt's
                # contribution" (arms.py's Overrides docstring) means.
                _t_score = time.perf_counter()
                # RUN_CACHE_REF_LOGPROBS (L3): a cache hit skips the two
                # ref_model forwards below, so count misses (cache absent,
                # or the flag off) before the call mutates the cache.
                ref_xminus_misses = (
                    sum(1 for edge in prompt_edges if edge.logp_ref_plus_xminus is None)
                    if self._cache_ref_logprobs
                    else len(prompt_edges)
                )
                m_plus, m_minus = self._prompt_margins(model, prompt_edges)
                pair_scoring_seconds += time.perf_counter() - _t_score
                # Per edge: score_response(x+,y+), score_response(x+,y-),
                # score_response(x-,y+), score_response(x-,y-) always, plus
                # the two ref_model score_response calls for (x-,y+)/(x-,y-)
                # only on a cache miss -- _prompt_margins' x+ side reuses the
                # edge's cached logp_ref_*.
                pair_scoring_forwards += 4 * len(prompt_edges) + 2 * ref_xminus_misses
                loss_prompt, prompt_stats = losses.prompt_loss(
                    m_plus,
                    m_minus,
                    tau_p=config.TAU_P,
                    lambda_0=config.LAMBDA_0,
                    clamp=config.MARGIN_CLAMP_MULT * config.KAPPA * config.TAU_P,
                )
                if self.overrides.lambda_p > 0:
                    total = total + pair_scale_factor * config.PAIR_LOSS_WEIGHT * self.overrides.lambda_p * loss_prompt
                stats["prompt_loss"] = loss_prompt.item()
                stats.update({f"prompt_{k}": v for k, v in prompt_stats.items()})

        # B2: this micro-batch's full local totals (both blocks above have
        # already added everything they will), folded into the running
        # per-optimiser-step totals _generate_and_score_completions surfaces
        # as previous_step_totals one cycle later.
        self._pair_scoring_seconds_step_accum += pair_scoring_seconds
        self._pair_scoring_forwards_step_accum += pair_scoring_forwards
        # B3: realised (post-sample) edge/prompt draws this micro-batch,
        # same accumulate-then-surface-next-cycle convention as above.
        self._edge_pairs_step_accum += edge_pairs_drawn
        self._prompt_pairs_step_accum += prompt_pairs_drawn

        if self.probe_enabled and len(self.edge_buffer) > 0 and current_update != self._last_probe_step:
            self._run_probe(model, inputs, clean, current_update)
            self._last_probe_step = current_update

        if self._pending_mechanism_record is not None:
            self._pending_mechanism_record.update({k: v for k, v in stats.items() if k.startswith("prompt_")})
            if "resp_evicted" in stats:
                self._pending_mechanism_record["resp_evicted"] = stats["resp_evicted"]
            self._pending_mechanism_record["pair_scoring_seconds"] = pair_scoring_seconds
            self._pending_mechanism_record["pair_scoring_forwards"] = pair_scoring_forwards
            with open(self.mechanism_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(self._pending_mechanism_record) + "\n")
            self._pending_mechanism_record = None

        self.last_loss_stats = stats
        mode = "train" if model.training else "eval"
        for name, value in stats.items():
            self._metrics[mode][name].append(value)

        return total

    # RUN_KL_CLAMP (K4, devbox 2026-09-13): a full copy of TRL 1.9.2's
    # GRPOTrainer._compute_loss (grpo_trainer.py:2991-3263), not a small
    # override -- TRL does not factor the KL block into its own overridable
    # method (unlike get_high_entropy_mask/get_off_policy_mask/
    # get_gamma_weights, which are), and per_token_logps (needed for the
    # diff) is computed earlier in the SAME method, so the clamp cannot be
    # applied from outside it. GRPOTrainer.compute_loss (public) dispatches
    # to self._compute_loss(model, inputs) via ordinary virtual dispatch, and
    # ConPOTrainer.compute_loss above already calls super().compute_loss(...)
    # -- that public method -- so no other plumbing is needed for this
    # override to take effect.
    #
    # The only change from TRL's own body is the "Compute the KL divergence"
    # block: the diff (ref_per_token_logps - per_token_logps) is clamped to
    # +/- self._kl_clamp before the exp when self._kl_clamp is set (mirrors
    # open-instruct's own KL clamp, at 40, config.KL_CLAMP's default);
    # self._kl_clamp is None (RUN_KL_CLAMP="none", the disable spelling)
    # leaves every line byte-identical to TRL's.
    #
    # PINNED TO trl==1.9.2: a TRL point release that changes _compute_loss
    # silently desyncs this copy. Recheck this method against the installed
    # trl/trainer/grpo_trainer.py on any TRL version bump (see SERVER.md).
    def _compute_loss(self, model: torch.nn.Module, inputs: dict[str, object]) -> torch.Tensor:
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies, aux_loss = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            compute_aux_loss=self.aux_loss_enabled,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            spatial_shapes=inputs.get("spatial_shapes"),
            num_tiles=inputs.get("num_tiles"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
            image_position_ids=inputs.get("image_position_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            # OPSM should use inference-time logprobs to detect both sources of off-policyness:
            # 1. Drift from gradient updates (always present)
            # 2. Drift from training-inference mismatch (when using vLLM)
            # When using vLLM, prioritize sampling_per_token_logps, otherwise use old_per_token_logps
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)

            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            diff = ref_per_token_logps - per_token_logps
            # RUN_KL_CLAMP (K4): clamp the diff to +/- self._kl_clamp before
            # the exp when set; None (disabled) is a no-op, keeping this
            # branch byte-identical to TRL's own unmodified computation.
            if self._kl_clamp is not None:
                diff = diff.clamp(min=-self._kl_clamp, max=self._kl_clamp)
            per_token_kl = torch.exp(diff) - diff - 1
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        elif self.loss_type == "vespo":
            phi_seq = self.get_gamma_weights(
                advantages=advantages,
                log_ratio_per_token=log_ratio,
                mask=mask,
                importance_sampling_ratio=inputs.get("importance_sampling_ratio"),
                k_pos=self.args.vespo_k_pos,
                lambda_pos=self.args.vespo_lambda_pos,
                k_neg=self.args.vespo_k_neg,
                lambda_neg=self.args.vespo_lambda_neg,
            )
            per_token_loss = -phi_seq * advantages * per_token_logps
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction and self.loss_type != "vespo":
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            policy_loss = loss.detach()
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo", "vespo"]:
            # `num_items_in_batch` spans the generation batch, so rescale it to one accumulation window
            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
            if mode == "train":  # in eval, the batch is neither split across steps nor accumulated
                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
            loss = (per_token_loss * mask).sum() / normalizer
            policy_loss = loss.detach()
        elif self.loss_type == "luspo":
            # Unless importance_sampling_level="token" (not recommended here), per_token_loss is expected to be (B, 1)
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            policy_loss = loss.detach()
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Entropy bonus: add entropy regularization to encourage exploration. _entropy_bonus_enabled is set
        # whenever a non-zero static coef is set OR adaptive mode is enabled (adaptive stays enabled even when
        # entropy_coef has been decremented to entropy_coef_min so it can recover once entropy drops again).
        if self._entropy_bonus_enabled:
            # When top_entropy_quantile < 1.0, entropy_mask restricts policy gradients to high-entropy
            # tokens. Use the same effective mask for the entropy bonus so it acts on the same tokens.
            effective_mask = mask if entropy_mask is None else mask * entropy_mask
            # Entropy bonus = mean per-token entropy H (the documented objective L = L_policy - coef * H), so
            # H does not depend on how each loss type normalizes its policy term. The term is computed so that
            # it accumulates to H over the optimizer step for every loss type and matches world_entropy below.
            # The only wrinkle is the normalizer: most loss types divide by the gradient accumulation step
            # count, but cispo/dapo/vespo divide by a global token count.
            if self.loss_type in ["cispo", "dapo", "vespo"]:
                # normalizer is a global token count, so summing the entropies (instead of averaging them
                # again) makes the term accumulate over the optimizer step to the global mean per-token
                # entropy, like the other loss types.
                entropy_loss = (entropies * effective_mask).sum() / normalizer
            else:
                # Mean per-token entropy of active tokens, scaled for gradient accumulation.
                entropy_loss = (entropies * effective_mask).sum() / effective_mask.sum().clamp(min=1.0) / normalizer

            # Apply the coefficient and gating from the end of the previous optimizer step, so that every
            # micro-batch in the current accumulation window applies the same entropy bonus. The adaptive
            # update below only takes effect on the next step.
            if self.use_adaptive_entropy:
                apply_coef = self.entropy_coef if self._last_world_entropy <= self.args.entropy_target else 0.0
            else:
                apply_coef = self.entropy_coef

            loss = loss - apply_coef * entropy_loss

            self._metrics[mode]["policy_loss"].append(self.accelerator.gather(policy_loss).nanmean().item())

            # Adaptive update. Gated on train mode so evaluation cannot mutate the entropy controller state.
            if self.use_adaptive_entropy and mode == "train":
                # Accumulate the entropy sum and active-token count of every micro-batch into a running window
                # buffer, so the controller measures the exact window-global entropy rather than just the last
                # micro-batch (which would be a 1 / gradient_accumulation_steps subsample).
                stats = torch.stack([(entropies * effective_mask).sum(), effective_mask.sum()]).detach()
                if self._entropy_window_stats is None:
                    self._entropy_window_stats = stats
                else:
                    self._entropy_window_stats = self._entropy_window_stats + stats
                # At the optimizer-step boundary, reduce the window totals across ranks (sum and token count
                # jointly, for a true global mean unbiased when ranks have different completion lengths),
                # update the coefficient for the next step, then reset the window buffer.
                if self.accelerator.sync_gradients:
                    window_stats = self.accelerator.reduce(self._entropy_window_stats, reduction="sum")
                    world_entropy = (window_stats[0] / window_stats[1].clamp(min=1.0)).item()
                    if world_entropy <= self.args.entropy_target:
                        self.entropy_coef = min(
                            self.entropy_coef + self.args.entropy_coef_delta, self.args.entropy_coef_max
                        )
                    else:
                        self.entropy_coef = max(
                            self.entropy_coef - self.args.entropy_coef_delta, self.args.entropy_coef_min
                        )
                    self._last_world_entropy = world_entropy
                    self._entropy_window_stats = None

            # Log entropy_coef on train optimizer-step boundaries (constant for static control; updated just
            # above for adaptive control). sync_gradients is always True in eval (no accumulation context).
            if mode == "train" and self.accelerator.sync_gradients:
                self._metrics[mode]["entropy_coef"].append(self.entropy_coef)

        # The policy loss above is scaled for gradient accumulation (HF auto-scaling is off here), so scale aux too
        if self.aux_loss_enabled:
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss + self.router_aux_loss_coef * aux_loss / normalizer
            self._metrics[mode]["aux_loss"].append(self.accelerator.gather_for_metrics(aux_loss).mean().item())

        # Log the metrics
        def masked_seq_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence": already one value per sequence
                return x.squeeze(1)
            return (x * mask).sum(-1) / mask.sum(-1)

        def global_masked_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence": one value per sequence
                local_sum, local_count = x.sum(), torch.tensor(float(x.shape[0]), device=x.device)
            else:
                local_sum, local_count = (x * mask).sum(), mask.sum().float()
            totals = self.accelerator.reduce(torch.stack([local_sum, local_count]), reduction="sum")
            return (totals[0] / totals[1].clamp(min=1.0)).item()

        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(global_masked_mean(per_token_kl))

        self._metrics[mode]["entropy"].append(global_masked_mean(entropies))

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            self._metrics[mode]["clip_ratio/low_mean"].append(global_masked_mean(is_low_clipped.float()))
            self._metrics[mode]["clip_ratio/high_mean"].append(global_masked_mean(is_high_clipped.float()))
            self._metrics[mode]["clip_ratio/region_mean"].append(global_masked_mean(is_region_clipped.float()))
            gathered_low_clip = self.accelerator.gather(masked_seq_mean(is_low_clipped.float()))
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(masked_seq_mean(is_high_clipped.float()))
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            self._metrics[mode]["cispo_clip_ratio"].append(global_masked_mean(is_cispo_clipped.float()))
        elif self.loss_type == "vespo":
            self._metrics[mode]["vespo/phi_seq_mean"].append(global_masked_mean(phi_seq))

        return loss

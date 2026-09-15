"""ConPO loss functions: response-level and prompt-side contrastive objectives.

Section references are to ConPO.pdf. Section 2.4 defines q_theta^(alpha) and
L_resp; section 2.5 defines m_theta, Gamma_theta, and L_prompt. Pure
functions only; the trainer owns model calls, buffering, and active
selection.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from conpo import config


def sequence_logprobs_with_tokens(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    completion_mask: torch.LongTensor,
    chunk_size: int = 512,
    temperature: float = 1.0,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """sequence_logprobs, plus the per-token values it sums and discards.

    One forward pass; log-probabilities are then computed in chunks along
    the sequence axis so no (B, L, vocab) fp32 tensor beyond one chunk is
    ever materialised. Standard causal shift: logits at position t predict
    the token at position t + 1, so completion_mask[:, 1:] gates which
    predicted tokens count. The second return value is that per-token
    log pi_theta(y_t | x, y_<t), masked to 0 outside the completion span
    (shape (B, L - 1)) -- RUN_TOKEN_LOGRATIO_CLAMP (K6, conpo/trainer.py)
    needs it to clamp a policy-minus-reference log-ratio token by token
    before summing, which the scalar sum below cannot support.

    `temperature` (RUN_PAIR_TEMPERATURE, audit F4, conpo/trainer.py's
    score_response) divides the logits before the log-softmax, the same
    place TRL divides by config.TEMPERATURE for the GRPO/KL terms
    (trl grpo_trainer.py's `logits.div_(self.temperature)`) -- this pair
    path computed q_theta from raw (temperature 1.0) logits otherwise.
    Default 1.0 is a no-op, byte-identical to before this parameter existed.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, L, V)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = completion_mask[:, 1:].to(shift_logits.dtype)

    seq_len = shift_labels.shape[1]
    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_logits = shift_logits[:, start:end, :]
        if temperature != 1.0:
            chunk_logits = chunk_logits / temperature
        chunk_logp = F.log_softmax(chunk_logits, dim=-1)
        token_logp = chunk_logp.gather(-1, shift_labels[:, start:end].unsqueeze(-1)).squeeze(-1)
        chunks.append(token_logp * shift_mask[:, start:end])
    token_logp = torch.cat(chunks, dim=1)
    return token_logp.sum(dim=1), token_logp


def sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    completion_mask: torch.LongTensor,
    chunk_size: int = 512,
    temperature: float = 1.0,
) -> torch.FloatTensor:
    """Sum log pi_theta(y_t | x, y_<t) over tokens where completion_mask is 1.

    One forward pass; log-probabilities are then computed in chunks along
    the sequence axis so no (B, L, vocab) fp32 tensor beyond one chunk is
    ever materialised. Standard causal shift: logits at position t predict
    the token at position t + 1, so completion_mask[:, 1:] gates which
    predicted tokens count. `temperature`: see sequence_logprobs_with_tokens.
    """
    return sequence_logprobs_with_tokens(model, input_ids, attention_mask, completion_mask, chunk_size, temperature)[
        0
    ]


def completion_lengths(completion_mask: torch.LongTensor) -> torch.LongTensor:
    """Number of completion tokens per sequence."""
    return completion_mask.sum(dim=1).long()


def q_theta_from_logratio(
    raw_logratio: torch.Tensor,
    lengths: torch.LongTensor,
    alpha: int = config.ALPHA,
) -> torch.Tensor:
    """q_theta^(alpha) from an already-formed raw log-ratio (section 2.4).

    q_theta normally forms `raw_logratio` as `logp_policy - logp_ref`, two
    scalar sequence-level sums; RUN_TOKEN_LOGRATIO_CLAMP (K6,
    conpo/trainer.py) instead forms it by clamping and summing a per-token
    log-ratio, so the length normalisation below is factored out to be
    shared by both.
    """
    length_norm = lengths.clamp(min=1).to(raw_logratio.dtype) ** alpha
    return raw_logratio / length_norm


def q_theta(
    logp_policy: torch.Tensor,
    logp_ref: torch.Tensor,
    lengths: torch.LongTensor,
    alpha: int = config.ALPHA,
) -> torch.Tensor:
    """q_theta^(alpha)(x, y) (section 2.4): length-normalised, reference-adjusted score."""
    return q_theta_from_logratio(logp_policy - logp_ref, lengths, alpha)


def clamp_token_logratio(diff: torch.Tensor, clamp: float) -> tuple[torch.Tensor, torch.Tensor]:
    """RUN_TOKEN_LOGRATIO_CLAMP (K6): elementwise-clamp a per-token
    policy-minus-reference log-ratio to +/-`clamp` before it is summed into
    the raw log-ratio `q_theta_from_logratio` normalises -- so no single
    token can carry more than `clamp` nats of margin.

    `diff` holds one response's completion-token log-ratios only (the
    caller, conpo/trainer.py's `_token_clamped_q`, selects those out of the
    masked (B, L-1) tensors `sequence_logprobs_with_tokens` returns before
    calling this). Returns (sum of the clamped values, a same-shape boolean
    of which entries the raw value exceeded -- pooled across every scored
    token by the caller into the `resp_token_clamp_frac` stat, mirroring
    prompt_loss's own clamp_frac above).
    """
    hits = diff.detach().abs() > clamp
    clamped = diff.clamp(min=-clamp, max=clamp)
    return clamped.sum(), hits


def resp_loss(
    delta: torch.Tensor,
    ages: torch.LongTensor,
    tau: float = config.TAU,
    gamma: float = config.GAMMA,
    kappa: float = config.KAPPA,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_resp (section 2.4) over a minibatch of N admitted edges.

    w(e) = gamma^age(e) * 1[Delta_theta(e) <= kappa * tau]; the active-edge
    indicator (True below the kappa*tau retirement bar; B5 (v, audit
    2026-09-14): renamed from the misleading `saturated`, which was True for
    edges that are NOT saturated) reads delta.detach() so a retired edge
    carries exactly zero gradient rather than a discontinuous one at the
    kappa*tau boundary.

    The mean below normalises by the full minibatch size N (matching the
    paper's E_{e~B}[...] over the buffer), so retired edges (w = 0) shrink
    the loss towards zero instead of being dropped from the denominator.
    Dividing by w.sum() instead is a valid alternative: it keeps surviving
    edges at a constant average contribution as more of the batch retires,
    at the cost of the loss no longer reflecting how much of the batch is
    still active. Not used here.
    """
    active = delta.detach() <= (kappa * tau)
    weights = (gamma ** ages.to(delta.dtype)) * active.to(delta.dtype)
    loss = -(weights * F.logsigmoid(delta / tau)).mean()
    stats = {
        "mean_margin": delta.mean().item(),
        "frac_retired": (~active).float().mean().item(),
        "mean_weight": weights.mean().item(),
    }
    return loss, stats


def diff_span(ids_a: Sequence[int], ids_b: Sequence[int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """The differing span between two token-id sequences (RUN_PAIR_OBJECTIVE
    span_local): the maximal common prefix and common suffix bound a span
    on each side that needs no edit metadata to find. Returns
    `((start_a, end_a), (start_b, end_b))`; identical sequences give both
    spans empty (`start == end`), a pure insertion or deletion gives one
    side empty, and two sequences with nothing in common give the whole of
    each.
    """
    n = min(len(ids_a), len(ids_b))
    prefix = 0
    while prefix < n and ids_a[prefix] == ids_b[prefix]:
        prefix += 1
    max_suffix = n - prefix
    suffix = 0
    while suffix < max_suffix and ids_a[len(ids_a) - 1 - suffix] == ids_b[len(ids_b) - 1 - suffix]:
        suffix += 1
    return (prefix, len(ids_a) - suffix), (prefix, len(ids_b) - suffix)


def clamp_span_logratio(diff: torch.Tensor, clamp: float | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise-clamp one side's span-token log-ratios to +/-`clamp`
    (RUN_TOKEN_LOGRATIO_CLAMP, reused here), returning the clamped tensor
    itself (not a sum -- `span_local_margin_and_anchor` below means it,
    unlike `clamp_token_logratio`'s whole-sequence sum) alongside a
    same-shape boolean of which entries the raw value exceeded. `clamp=None`
    passes `diff` through unchanged with an all-False hits tensor.
    """
    if clamp is None:
        return diff, torch.zeros_like(diff, dtype=torch.bool)
    hits = diff.detach().abs() > clamp
    return diff.clamp(min=-clamp, max=clamp), hits


def _safe_mean(x: torch.Tensor) -> torch.Tensor:
    """`x.mean()`, or a 0.0 scalar (not NaN) when `x` is empty."""
    return x.mean() if x.numel() else x.new_zeros(())


def span_local_margin_and_anchor(
    r_plus_span: torch.Tensor,
    r_minus_span: torch.Tensor,
    clamp: float | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One edge's span-local margin and positive-anchor term
    (RUN_PAIR_OBJECTIVE=span_local): `r_plus_span`/`r_minus_span` are the
    per-token policy-minus-reference log-ratios over each side's differing
    span only (`diff_span`) -- shared tokens never reach this function, so
    they carry no gradient at all. The margin is
    `mean(r_plus_span) - mean(r_minus_span)`, each clamped first
    (`clamp_span_logratio`); an empty span contributes exactly 0 to its
    side (`_safe_mean`), so a pair with both spans empty has margin 0. The
    anchor is `mean(max(0, -r_t))` over the clamped y_plus span only (the
    DPO-Positive penalty for a repaired token whose policy log-prob falls
    below the reference's).

    Returns `(margin, anchor, hits)`, `hits` the two sides' clamp-hit
    booleans concatenated (y_plus first) for the caller to pool across
    edges into `resp_token_clamp_frac`.
    """
    r_plus_c, hits_plus = clamp_span_logratio(r_plus_span, clamp)
    r_minus_c, hits_minus = clamp_span_logratio(r_minus_span, clamp)
    margin = _safe_mean(r_plus_c) - _safe_mean(r_minus_c)
    anchor = _safe_mean(torch.clamp(-r_plus_c, min=0.0))
    return margin, anchor, torch.cat([hits_plus, hits_minus])


def span_local_resp_loss(
    m: torch.Tensor,
    ages: torch.LongTensor,
    anchor_terms: torch.Tensor,
    tau: float = config.TAU,
    gamma: float = config.GAMMA,
    kappa: float = config.KAPPA,
    pos_anchor_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_resp under RUN_PAIR_OBJECTIVE=span_local: `resp_loss` unchanged
    (its retirement rule only ever saw a generic margin tensor, so it
    applies to `m` exactly as it does to the sequence-level Delta_theta),
    plus `pos_anchor_weight * mean(anchor_terms)` (RUN_POS_ANCHOR_WEIGHT,
    0 = off). `stats["pos_anchor"]` is always recorded, even when the
    weight is 0, mirroring how `prompt_stats["gamma_theta"]` is logged
    regardless of `overrides.lambda_p` (conpo/trainer.py's compute_loss).
    """
    loss, stats = resp_loss(m, ages, tau=tau, gamma=gamma, kappa=kappa)
    anchor_mean = _safe_mean(anchor_terms)
    if pos_anchor_weight:
        loss = loss + pos_anchor_weight * anchor_mean
    stats["pos_anchor"] = anchor_mean.item()
    return loss, stats


def prompt_loss(
    m_plus: torch.Tensor,
    m_minus: torch.Tensor,
    tau_p: float = config.TAU_P,
    lambda_0: float = config.LAMBDA_0,
    clamp: float = float("inf"),
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_prompt (section 2.5): one-sided prompt-side contrastive objective.

    Gamma_theta = mean(m_plus - m_minus) is reported for logging only; it
    is not part of the loss (section 2.5 explains why: a raw
    difference-in-differences loss would reward degrading the response
    when the constraint is absent).

    `clamp` bounds m_plus and m_minus to [-clamp, clamp] before m_plus
    enters logsigmoid and m_minus is squared: unlike Delta_theta, m_theta
    has no retirement rule of its own, so one outlier margin would otherwise
    dominate both terms' mean unchecked. Defaults to no clamping (+-inf) so
    an unspecified call is unaffected. The returned stats (gamma_theta,
    mean_m_plus, mean_m_minus) are computed on the clamped values;
    clamp_frac is the fraction of entries, pooled across both tensors,
    whose magnitude exceeded clamp.
    """
    m_plus_c = torch.clamp(m_plus, -clamp, clamp)
    m_minus_c = torch.clamp(m_minus, -clamp, clamp)
    loss = -F.logsigmoid(m_plus_c / tau_p).mean() + lambda_0 * (m_minus_c**2).mean()
    hit = torch.cat([m_plus.detach().abs() > clamp, m_minus.detach().abs() > clamp])
    stats = {
        "gamma_theta": (m_plus_c - m_minus_c).mean().item(),
        "mean_m_plus": m_plus_c.mean().item(),
        "mean_m_minus": m_minus_c.mean().item(),
        "clamp_frac": hit.float().mean().item(),
    }
    return loss, stats

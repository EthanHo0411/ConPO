"""ConPO loss functions: response-level and prompt-side contrastive objectives.

Section references are to ConPO.pdf. Section 2.4 defines q_theta^(alpha) and
L_resp; section 2.5 defines m_theta, Gamma_theta, and L_prompt. Pure
functions only; the trainer owns model calls, buffering, and active
selection.
"""

import torch
import torch.nn.functional as F

from conpo import config


def sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    completion_mask: torch.LongTensor,
    chunk_size: int = 512,
) -> torch.FloatTensor:
    """Sum log pi_theta(y_t | x, y_<t) over tokens where completion_mask is 1.

    One forward pass; log-probabilities are then computed in chunks along
    the sequence axis so no (B, L, vocab) fp32 tensor beyond one chunk is
    ever materialised. Standard causal shift: logits at position t predict
    the token at position t + 1, so completion_mask[:, 1:] gates which
    predicted tokens count.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits  # (B, L, V)

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = completion_mask[:, 1:].to(shift_logits.dtype)

    batch, seq_len = shift_labels.shape
    total = torch.zeros(batch, dtype=shift_logits.dtype, device=shift_logits.device)
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_logp = F.log_softmax(shift_logits[:, start:end, :], dim=-1)
        token_logp = chunk_logp.gather(-1, shift_labels[:, start:end].unsqueeze(-1)).squeeze(-1)
        total = total + (token_logp * shift_mask[:, start:end]).sum(dim=1)
    return total


def completion_lengths(completion_mask: torch.LongTensor) -> torch.LongTensor:
    """Number of completion tokens per sequence."""
    return completion_mask.sum(dim=1).long()


def q_theta(
    logp_policy: torch.Tensor,
    logp_ref: torch.Tensor,
    lengths: torch.LongTensor,
    alpha: int = config.ALPHA,
) -> torch.Tensor:
    """q_theta^(alpha)(x, y) (section 2.4): length-normalised, reference-adjusted score."""
    length_norm = lengths.clamp(min=1).to(logp_policy.dtype) ** alpha
    return (logp_policy - logp_ref) / length_norm


def resp_loss(
    delta: torch.Tensor,
    ages: torch.LongTensor,
    tau: float = config.TAU,
    gamma: float = config.GAMMA,
    kappa: float = config.KAPPA,
) -> tuple[torch.Tensor, dict[str, float]]:
    """L_resp (section 2.4) over a minibatch of N admitted edges.

    w(e) = gamma^age(e) * 1[Delta_theta(e) <= kappa * tau]; the saturation
    indicator reads delta.detach() so a retired edge carries exactly zero
    gradient rather than a discontinuous one at the kappa*tau boundary.

    The mean below normalises by the full minibatch size N (matching the
    paper's E_{e~B}[...] over the buffer), so retired edges (w = 0) shrink
    the loss towards zero instead of being dropped from the denominator.
    Dividing by w.sum() instead is a valid alternative: it keeps surviving
    edges at a constant average contribution as more of the batch retires,
    at the cost of the loss no longer reflecting how much of the batch is
    still active. Not used here.
    """
    saturated = delta.detach() <= (kappa * tau)
    weights = (gamma ** ages.to(delta.dtype)) * saturated.to(delta.dtype)
    loss = -(weights * F.logsigmoid(delta / tau)).mean()
    stats = {
        "mean_margin": delta.mean().item(),
        "frac_retired": (~saturated).float().mean().item(),
        "mean_weight": weights.mean().item(),
    }
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

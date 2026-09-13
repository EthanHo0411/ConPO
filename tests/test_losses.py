"""Tests for conpo.losses: response-level and prompt-side contrastive objectives.

The tiny causal LM below is only needed for test (a) (sequence_logprobs
reads real model logits); the rest of the loss functions are pure tensor
math tested directly against the formulas in ConPO.pdf section 2.4/2.5.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from conpo import config
from conpo.losses import completion_lengths, prompt_loss, q_theta, resp_loss, sequence_logprobs


def _tiny_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=512,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model


def _naive_sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    completion_mask: torch.LongTensor,
) -> torch.Tensor:
    """Reference implementation: one full (B, L, V) log_softmax, no chunking."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = completion_mask[:, 1:].to(shift_logits.dtype)
    logp = F.log_softmax(shift_logits, dim=-1)
    token_logp = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * shift_mask).sum(dim=1)


def _ragged_completion_mask(batch: int, seq_len: int) -> torch.LongTensor:
    """Row b's completion span starts at a distinct offset, so mask lengths ragged."""
    torch.manual_seed(1)
    mask = torch.zeros(batch, seq_len, dtype=torch.long)
    for b in range(batch):
        start = int(torch.randint(1, seq_len - 2, (1,)).item())
        mask[b, start:] = 1
    return mask


# --- (a) sequence_logprobs matches naive full log_softmax and is chunk-size invariant ---


def test_sequence_logprobs_matches_naive_and_is_chunk_size_invariant() -> None:
    model = _tiny_model()
    batch, seq_len = 3, 20
    torch.manual_seed(2)
    input_ids = torch.randint(0, 512, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
    completion_mask = _ragged_completion_mask(batch, seq_len)

    with torch.no_grad():
        naive = _naive_sequence_logprobs(model, input_ids, attention_mask, completion_mask)
        chunk_4 = sequence_logprobs(model, input_ids, attention_mask, completion_mask, chunk_size=4)
        chunk_512 = sequence_logprobs(model, input_ids, attention_mask, completion_mask, chunk_size=512)

    assert torch.allclose(chunk_4, naive, atol=1e-5)
    assert torch.allclose(chunk_512, naive, atol=1e-5)
    assert torch.allclose(chunk_4, chunk_512, atol=1e-5)

    lengths = completion_lengths(completion_mask)
    assert lengths.tolist() == completion_mask.sum(dim=1).tolist()


# --- (b) margins and loss are exactly zero-delta values at theta = theta_ref ---


def test_q_theta_zero_when_policy_equals_ref() -> None:
    logp = torch.tensor([-3.2, -10.0, -0.5])
    lengths = torch.tensor([4, 12, 1])
    margin = q_theta(logp, logp, lengths)
    assert torch.equal(margin, torch.zeros_like(margin))


def test_resp_loss_zero_deltas_equals_log2_weighted_by_age() -> None:
    delta = torch.zeros(5)
    ages = torch.tensor([0, 1, 2, 3, 500])
    loss, stats = resp_loss(delta, ages)

    expected_weights = config.GAMMA ** ages.to(torch.float32)
    expected_loss = expected_weights.mean() * math.log(2.0)

    assert torch.allclose(loss, expected_loss, atol=1e-6)
    assert stats["frac_retired"] == 0.0
    assert stats["mean_weight"] == pytest.approx(expected_weights.mean().item())


# --- (c) saturation: a margin above kappa * tau gets weight 0 -------------------


def test_resp_loss_saturated_edge_gets_zero_weight() -> None:
    tau, kappa, gamma = config.TAU, config.KAPPA, config.GAMMA
    delta = torch.tensor([kappa * tau + 0.05, kappa * tau - 0.05])
    ages = torch.tensor([2, 3])
    loss, stats = resp_loss(delta, ages, tau=tau, kappa=kappa, gamma=gamma)

    expected_weights = torch.tensor([0.0, gamma**3])
    expected_loss = -(expected_weights * F.logsigmoid(delta / tau)).mean()

    assert torch.allclose(loss, expected_loss, atol=1e-6)
    assert stats["frac_retired"] == 0.5
    assert stats["mean_weight"] == pytest.approx(expected_weights.mean().item())


# --- (d) gradient of lambda_P * L_prompt + L_resp scales linearly with lambda_P ---


def test_composite_gradient_scales_linearly_with_lambda_p() -> None:
    z = torch.tensor([0.4, -0.2, 0.9], requires_grad=True)
    delta = torch.tensor([0.01, 0.02, -0.01])  # independent of z: resp term contributes no grad
    ages = torch.zeros(3, dtype=torch.long)

    def composite(lambda_p: float) -> torch.Tensor:
        p_loss, _ = prompt_loss(z, -z)
        r_loss, _ = resp_loss(delta, ages)
        return lambda_p * p_loss + r_loss

    grads = {}
    for lambda_p in (1.0, 3.0):
        z.grad = None
        composite(lambda_p).backward()
        grads[lambda_p] = z.grad.clone()

    assert not torch.allclose(grads[1.0], torch.zeros_like(grads[1.0]))
    assert torch.allclose(grads[3.0], 3.0 * grads[1.0], atol=1e-6)


# --- (e) prompt_loss Gamma diagnostic ------------------------------------------


def test_prompt_loss_gamma_diagnostic_matches_mean_difference() -> None:
    m_plus = torch.tensor([0.5, -0.2, 1.1])
    m_minus = torch.tensor([0.1, 0.3, -0.4])
    _, stats = prompt_loss(m_plus, m_minus)

    assert stats["gamma_theta"] == pytest.approx((m_plus - m_minus).mean().item())
    assert stats["mean_m_plus"] == pytest.approx(m_plus.mean().item())
    assert stats["mean_m_minus"] == pytest.approx(m_minus.mean().item())
    assert stats["clamp_frac"] == 0.0  # default clamp is +-inf: nothing hit it


# --- (f) prompt_loss margin clamp -----------------------------------------


def test_prompt_loss_clamp_matches_loss_at_clamped_values_and_reports_clamp_frac() -> None:
    clamp = 1.0
    m_plus = torch.tensor([5.0, 0.3, -0.2])  # entry 0 exceeds +clamp
    m_minus = torch.tensor([-5.0, 0.1, 2.0])  # entries 0 and 2 exceed the clamp
    loss, stats = prompt_loss(m_plus, m_minus, clamp=clamp)

    clamped_plus = torch.clamp(m_plus, -clamp, clamp)
    clamped_minus = torch.clamp(m_minus, -clamp, clamp)
    expected_loss = -F.logsigmoid(clamped_plus / config.TAU_P).mean() + config.LAMBDA_0 * (clamped_minus**2).mean()

    assert torch.allclose(loss, expected_loss, atol=1e-6)
    assert stats["mean_m_plus"] == pytest.approx(clamped_plus.mean().item())
    assert stats["mean_m_minus"] == pytest.approx(clamped_minus.mean().item())
    assert stats["gamma_theta"] == pytest.approx((clamped_plus - clamped_minus).mean().item())
    # 3 of the 6 pooled entries (m_plus[0], m_minus[0], m_minus[2]) exceed the clamp.
    assert stats["clamp_frac"] == pytest.approx(3 / 6)

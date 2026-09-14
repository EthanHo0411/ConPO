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
from conpo.losses import (
    clamp_span_logratio,
    clamp_token_logratio,
    completion_lengths,
    diff_span,
    prompt_loss,
    q_theta,
    resp_loss,
    sequence_logprobs,
    span_local_margin_and_anchor,
    span_local_resp_loss,
)


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


# --- (g) K6 per-token log-ratio clamp -------------------------------------


def test_clamp_token_logratio_clamps_and_reports_hits() -> None:
    diff = torch.tensor([0.5, -3.0])  # token 0 within [-1, 1], token 1 exceeds it
    clamped_sum, hits = clamp_token_logratio(diff, clamp=1.0)

    assert torch.allclose(clamped_sum, torch.tensor(0.5 - 1.0))
    assert hits.tolist() == [False, True]  # 1 of 2 tokens clamped


# --- (h) K8 span_local: diff_span (common-prefix/suffix span finder) -----------


def test_diff_span_identical_sequences_gives_both_spans_empty() -> None:
    a, b = [1, 2, 3, 4], [1, 2, 3, 4]
    span_a, span_b = diff_span(a, b)
    assert span_a == (4, 4)
    assert span_b == (4, 4)


def test_diff_span_pure_insertion_gives_one_side_empty() -> None:
    a, b = [1, 2, 4], [1, 2, 3, 4]  # b inserts a 3 before the trailing 4
    span_a, span_b = diff_span(a, b)
    assert span_a == (2, 2)  # nothing missing from a's own tokens
    assert span_b == (2, 3)  # the inserted token
    assert b[span_b[0] : span_b[1]] == [3]


def test_diff_span_pure_deletion_gives_one_side_empty() -> None:
    a, b = [1, 2, 3, 4], [1, 2, 4]  # mirror of the insertion case
    span_a, span_b = diff_span(a, b)
    assert span_a == (2, 3)
    assert a[span_a[0] : span_a[1]] == [3]
    assert span_b == (2, 2)


def test_diff_span_single_token_substitution_in_the_middle() -> None:
    a, b = [1, 2, 3, 4], [1, 2, 9, 4]
    span_a, span_b = diff_span(a, b)
    assert a[span_a[0] : span_a[1]] == [3]
    assert b[span_b[0] : span_b[1]] == [9]


def test_diff_span_disjoint_sequences_spans_the_whole_of_each() -> None:
    a, b = [1, 2, 3], [4, 5, 6]
    span_a, span_b = diff_span(a, b)
    assert span_a == (0, 3)
    assert span_b == (0, 3)


def test_diff_span_common_prefix_only() -> None:
    a, b = [1, 2, 3, 4], [1, 2, 8, 9]
    span_a, span_b = diff_span(a, b)
    assert a[span_a[0] : span_a[1]] == [3, 4]
    assert b[span_b[0] : span_b[1]] == [8, 9]


def test_diff_span_common_suffix_only() -> None:
    a, b = [1, 2, 3, 4], [8, 9, 3, 4]
    span_a, span_b = diff_span(a, b)
    assert a[span_a[0] : span_a[1]] == [1, 2]
    assert b[span_b[0] : span_b[1]] == [8, 9]


def test_diff_span_degenerate_empty_sequences() -> None:
    assert diff_span([], []) == ((0, 0), (0, 0))
    span_a, span_b = diff_span([], [1, 2, 3])
    assert span_a == (0, 0)
    assert span_b == (0, 3)


# --- (i) K8 span_local: clamp_span_logratio (elementwise, not a reduction) -----


def test_clamp_span_logratio_clamps_elementwise_and_reports_hits() -> None:
    diff = torch.tensor([0.5, -3.0, 2.0])
    clamped, hits = clamp_span_logratio(diff, clamp=1.0)

    assert torch.allclose(clamped, torch.tensor([0.5, -1.0, 1.0]))
    assert hits.tolist() == [False, True, True]


def test_clamp_span_logratio_none_clamp_is_a_passthrough() -> None:
    diff = torch.tensor([0.5, -30.0])
    clamped, hits = clamp_span_logratio(diff, clamp=None)

    assert torch.equal(clamped, diff)
    assert hits.tolist() == [False, False]


def test_clamp_span_logratio_empty_tensor() -> None:
    diff = torch.empty(0)
    clamped, hits = clamp_span_logratio(diff, clamp=1.0)
    assert clamped.numel() == 0
    assert hits.numel() == 0


# --- (j) K8 span_local: span_local_margin_and_anchor ---------------------------


def test_span_local_margin_matches_mean_difference_of_clamped_spans() -> None:
    r_plus = torch.tensor([0.2, 0.4, 0.6])
    r_minus = torch.tensor([-0.1, 0.3])
    margin, anchor, hits = span_local_margin_and_anchor(r_plus, r_minus, clamp=None)

    assert margin == pytest.approx(r_plus.mean().item() - r_minus.mean().item())
    assert hits.numel() == 5  # pooled across both sides


def test_span_local_margin_empty_span_on_one_side_contributes_zero() -> None:
    r_plus = torch.empty(0)
    r_minus = torch.tensor([-0.5, 0.5])
    margin, anchor, hits = span_local_margin_and_anchor(r_plus, r_minus, clamp=None)

    assert margin == pytest.approx(0.0 - r_minus.mean().item())
    assert anchor == pytest.approx(0.0)  # anchor only looks at the (empty) y_plus span


def test_span_local_margin_both_spans_empty_is_exactly_zero() -> None:
    margin, anchor, hits = span_local_margin_and_anchor(torch.empty(0), torch.empty(0), clamp=None)
    assert margin.item() == 0.0
    assert anchor.item() == 0.0
    assert hits.numel() == 0


def test_span_local_anchor_only_penalises_negative_span_tokens() -> None:
    r_plus = torch.tensor([0.3, -0.2, -0.5, 0.1])  # two tokens below the reference
    _, anchor, _ = span_local_margin_and_anchor(r_plus, torch.tensor([0.0]), clamp=None)

    expected = torch.clamp(-r_plus, min=0.0).mean()
    assert anchor == pytest.approx(expected.item())
    assert anchor.item() > 0.0


def test_span_local_margin_and_anchor_report_clamp_hits() -> None:
    r_plus = torch.tensor([5.0, 0.1])  # entry 0 exceeds the clamp
    r_minus = torch.tensor([-5.0])  # exceeds the clamp
    _, _, hits = span_local_margin_and_anchor(r_plus, r_minus, clamp=1.0)
    assert hits.tolist() == [True, False, True]


def test_span_local_gradient_touches_only_the_differing_span() -> None:
    """Shared tokens (outside the diff_span slice) get exactly zero gradient
    -- not merely a diluted one -- because they never enter the computation
    graph the margin is built from."""
    ids_plus = [1, 2, 3, 4, 5]
    ids_minus = [1, 2, 9, 4, 5]
    (sp_start, sp_end), (sm_start, sm_end) = diff_span(ids_plus, ids_minus)

    r_plus_full = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], requires_grad=True)
    r_minus_full = torch.tensor([0.1, 0.2, -0.9, 0.4, 0.5], requires_grad=True)
    margin, anchor, _ = span_local_margin_and_anchor(
        r_plus_full[sp_start:sp_end], r_minus_full[sm_start:sm_end], clamp=None
    )
    (margin + anchor).backward()

    in_span_plus = torch.zeros_like(r_plus_full, dtype=torch.bool)
    in_span_plus[sp_start:sp_end] = True
    in_span_minus = torch.zeros_like(r_minus_full, dtype=torch.bool)
    in_span_minus[sm_start:sm_end] = True

    assert torch.all(r_plus_full.grad[~in_span_plus] == 0.0)
    assert torch.all(r_minus_full.grad[~in_span_minus] == 0.0)
    assert torch.any(r_plus_full.grad[in_span_plus] != 0.0)
    assert torch.any(r_minus_full.grad[in_span_minus] != 0.0)


# --- (k) K8 span_local: span_local_resp_loss ------------------------------------


def test_span_local_resp_loss_matches_resp_loss_when_anchor_weight_is_zero() -> None:
    m = torch.tensor([0.05, -0.02, 0.2])
    ages = torch.tensor([0, 1, 2])
    anchor_terms = torch.tensor([0.3, 0.1, 0.0])

    plain_loss, plain_stats = resp_loss(m, ages)
    span_loss, span_stats = span_local_resp_loss(m, ages, anchor_terms, pos_anchor_weight=0.0)

    assert torch.allclose(span_loss, plain_loss)
    for key in plain_stats:
        assert span_stats[key] == pytest.approx(plain_stats[key])
    assert span_stats["pos_anchor"] == pytest.approx(anchor_terms.mean().item())


def test_span_local_resp_loss_adds_weighted_anchor_mean() -> None:
    m = torch.tensor([0.05, -0.02, 0.2])
    ages = torch.tensor([0, 1, 2])
    anchor_terms = torch.tensor([0.3, 0.1, 0.0])

    plain_loss, _ = resp_loss(m, ages)
    weighted_loss, stats = span_local_resp_loss(m, ages, anchor_terms, pos_anchor_weight=2.0)

    assert torch.allclose(weighted_loss, plain_loss + 2.0 * anchor_terms.mean())
    assert stats["pos_anchor"] == pytest.approx(anchor_terms.mean().item())


def test_span_local_resp_loss_retirement_matches_resp_loss_on_same_delta_ages() -> None:
    tau, kappa, gamma = config.TAU, config.KAPPA, config.GAMMA
    m = torch.tensor([kappa * tau + 0.05, kappa * tau - 0.05])
    ages = torch.tensor([2, 3])
    anchor_terms = torch.zeros(2)

    plain_loss, plain_stats = resp_loss(m, ages, tau=tau, kappa=kappa, gamma=gamma)
    span_loss, span_stats = span_local_resp_loss(
        m, ages, anchor_terms, tau=tau, kappa=kappa, gamma=gamma, pos_anchor_weight=0.0
    )

    assert torch.allclose(span_loss, plain_loss)
    assert span_stats["frac_retired"] == plain_stats["frac_retired"] == 0.5

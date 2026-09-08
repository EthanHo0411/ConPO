"""ConPO constants. No logic here, only pinned values and their provenance.

Section references are to ConPO.pdf (Counterfactual Contrastive Policy
Optimisation for Verifiable Constraints in LLMs).
"""

import os

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

POLICY_MODEL = "Qwen/Qwen3-4B"
POLICY_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# later ablations only; no script uses these now
LATER_ABLATION_MODELS = {
    "qwen3_8b": "Qwen/Qwen3-8B",
    "tulu3_8b_sft": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "olmo3_7b_sft": "allenai/Olmo-3-7B-Instruct-SFT",
}

JUDGE_MODEL = "Qwen/Qwen3-14B"

# ---------------------------------------------------------------------------
# LoRA (section 4.1: "rank 64, scaling factor 128, and dropout 0.05, applied
# to attention and feed-forward projections")
# ---------------------------------------------------------------------------

LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# ---------------------------------------------------------------------------
# Sampling (section 4.1: "G = 8 responses per prompt with temperature 0.8,
# top-p 0.95, and a maximum of 1,024 tokens")
# ---------------------------------------------------------------------------

NUM_GENERATIONS = 8
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_COMPLETION_TOKENS = 1024
MAX_PROMPT_TOKENS = 2048
# Explicit because Qwen3 declares a huge context and vLLM sizes the KV cache
# for all of it; MAX_PROMPT_TOKENS + MAX_COMPLETION_TOKENS fits with margin.
MAX_MODEL_LEN = 3584

# ---------------------------------------------------------------------------
# Optimiser (section 4.1: "AdamW with learning rate 1e-5, 3% warm-up, cosine
# decay, gradient clipping at 1.0, and effective batch size 128")
# ---------------------------------------------------------------------------

LR = 1e-5
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
MAX_GRAD_NORM = 1.0
# The LoRA-without-regret precondition is under 32; batch sensitivity check
# pending.
EFFECTIVE_BATCH = 128

# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------

# TRL 1.9.2 defaults to "dapo", so explicit.
LOSS_TYPE = "grpo"
# Section 2: standard GRPO clip ratio; not renamed or redefined in this
# paper, so the conventional 0.2 is kept. DEV_SET.
EPSILON = 0.2  # DEV_SET
# beta in L_ConPO = L_resp + lambda_P L_prompt + lambda_G L_clean + beta L_KL
# (section 2.7): "the explicit KL term L_KL(theta), computed against the
# frozen initial policy, is the sole mechanism regularising absolute
# likelihoods, and beta is ablated accordingly." No numeric default is given
# in the main text (beta is one of the values "selected once on the internal
# development set", section 4.1). DEV_SET.
KL_BETA = 0.001  # DEV_SET

# ---------------------------------------------------------------------------
# Repair operator (section 4.1: "The repair operator proposes at most four
# candidates for each of at most three repair steps.")
# ---------------------------------------------------------------------------

MAX_CANDIDATES = 4
MAX_STEPS = 3

# ---------------------------------------------------------------------------
# Counterfactual-edge buffer (section 4.1: "The buffer holds at most 50,000
# edges ... with a hard age cap of 500 optimiser updates")
# ---------------------------------------------------------------------------

BUFFER_SIZE = 50000
BUFFER_AGE_CAP = 500

# ---------------------------------------------------------------------------
# ConPO loss / selection hyperparameters (section 2 defines the symbols,
# section 4.1: "Loss weights, temperatures, delta, gamma, kappa, and the
# probe-batch period S are selected once on the internal development set."
# The paper gives no numeric defaults for these; every one marked DEV_SET
# below is a sensible placeholder pending that tuning pass.
# ---------------------------------------------------------------------------

# delta (section 2.2/2.3): fidelity bound d_Y(y, y') <= delta gating both
# repair-proposal admission and corruption admission; d_Y is token-level
# edit distance normalised by response length (section 2.1), so delta lives
# in [0, 1]. DEV_SET.
DELTA = 0.3  # DEV_SET

# tau (section 2.4): response-level contrastive temperature in
# L_resp = -E[w(e) log sigma(Delta_theta(e) / tau)]. DEV_SET.
TAU = 0.1  # DEV_SET

# tau_p (section 2.5): prompt-side temperature in
# L_prompt = -E[log sigma(m_theta(x^+A) / tau_p)] + lambda_0 E[m_theta(x^-A)^2].
# DEV_SET.
TAU_P = 0.1  # DEV_SET

# gamma (section 2.4): per-update age decay in w(e) = gamma^age(e) * [...],
# gamma in (0, 1]. DEV_SET.
GAMMA = 0.99  # DEV_SET

# kappa (section 2.4): saturation multiplier; w(e) retires an edge once
# Delta_theta(e) > kappa * tau (Lemma 7). DEV_SET.
KAPPA = 5.0  # DEV_SET

# lambda_0 (section 2.5): coefficient on the indifference penalty
# E[m_theta(x^-A)^2] in L_prompt. DEV_SET.
LAMBDA_0 = 0.1  # DEV_SET

# lambda_P, lambda_G (section 2.7): weights in
# L_ConPO = L_resp + lambda_P L_prompt + lambda_G L_clean + beta L_KL. DEV_SET.
LAMBDA_P = 1.0  # DEV_SET
LAMBDA_G = 1.0  # DEV_SET

# lambda_T, lambda_U, lambda_C (section 2.6): acquisition-score weights in
# a_c = lambda_T T_c + lambda_U U_c + lambda_C C_c, "calibrated on the
# development set jointly ... so that the three terms have comparable
# dynamic ranges" -- started equal pending that calibration. DEV_SET.
LAMBDA_T = 1.0  # DEV_SET
LAMBDA_U = 1.0  # DEV_SET
LAMBDA_C = 1.0  # DEV_SET

# eta (section 2.6): softmax temperature in
# mu(c | x) = (1 - eps_a) * exp(a_c / eta) / sum_k exp(a_k / eta) + eps_a / |C(x)|.
# DEV_SET.
ETA = 1.0  # DEV_SET

# eps_a (section 2.6): exploration floor, "eps_a > 0 prevents permanent
# starvation of a constraint family." DEV_SET.
EPS_A = 0.1  # DEV_SET

# Probe size (section 2.6: "small probe batches of at most 16 edges per
# family"). Given explicitly in the paper.
PROBE_SIZE = 16

# Probe-batch period S (section 2.6: "refreshed every S", section 4.1: S is
# one of the values selected on the development set). DEV_SET.
PROBE_PERIOD_S = 50  # DEV_SET

# EMA decay for T_c (section 2.6, "Cost and noise": "the exponential moving
# average (decay 0.9) enters the acquisition score"). Given explicitly for
# T_c; p_c's own EMA (section 2.6: "an exponential moving average of
# v_c(x, y) over fresh on-policy samples") has no separately stated decay in
# the paper text, so it is set to match T_c's 0.9 by convention.
EMA_DECAY_P_C = 0.9
EMA_DECAY_T_C = 0.9

# alpha (section 2.4): length-normalisation exponent in
# q_theta^(alpha)(x, y) = |y|^-alpha [log pi_theta(y|x) - log pi_ref(y|x)].
# "we write q_theta = q_theta^(1) unless stated otherwise" -> default 1;
# ablated over {0, 1}.
ALPHA = 1
ALPHA_ABLATION_VALUES = (0, 1)

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

POOL_DATASET = "allenai/IF_multi_constraints_upto5"
IFEVAL_DATASET = "google/IFEval"
IFBENCH_TEST_DATASET = "allenai/IFBench_test"

# Retention suite (design choice, builder, 2026-09-04): GSM8K, MMLU, BBH, the
# full test splits (user decision), CPU-verified against the Hub with
# datasets.load_dataset -- row counts matched RETENTION_SIZES exactly.
GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "test"

MMLU_DATASET = "cais/mmlu"
MMLU_CONFIG = "all"
MMLU_SPLIT = "test"

BBH_DATASET = "lukaemon/bbh"
BBH_TASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)

RETENTION_MAX_TOKENS = 512
# Fixed n per benchmark (full test splits, user decision), for wave 2's
# Wilson intervals over scripts/eval_retention.py's "correct"/"n" counts.
RETENTION_SIZES = {"gsm8k": 1319, "mmlu": 14042, "bbh": 6511}

# ---------------------------------------------------------------------------
# Compute notes (section 4.1 / cluster allocation; not paper-derived)
# ---------------------------------------------------------------------------

# Two H100 80 GB GPUs, one training run per card (user decision 2026-09-03);
# the full-FT point-check takes both. Measured 4B LoRA co-locate footprint:
# 63-65 GB. Judge (Qwen3-14B) footprint: 27.5 GB.

# ---------------------------------------------------------------------------
# RUN_* environment contract
# ---------------------------------------------------------------------------


def env(name: str, default: str) -> str:
    """Read a RUN_* environment variable, falling back to `default`."""
    return os.environ.get(name, default)


# RUN_ARM: one of "full", "solve_only", "forward_only", "reverse_only".
# RUN_SEED: integer seed for the run.
# RUN_STEPS: integer optimiser-step budget.
# RUN_TOKEN_BUDGET: integer generated-token budget (section 4.1: "The
#   primary comparison matches total generated tokens").
# RUN_USE_VLLM: "1" to sample through vLLM, "0" for HF generate.
# RUN_JUDGE_ENDPOINT: URL of the judge model's inference endpoint (RUN_JUDGE_MODE=http only).
# RUN_JUDGE_MODE: "batched" (default, JUDGE_MODE_DEFAULT) or "http".
# RUN_JUDGE_MODEL: overrides JUDGE_MODEL for the judge conpo.judge.build_judge constructs.
# RUN_VETO_PERIOD: overrides VETO_PERIOD_K.
# RUN_MODEL: overrides POLICY_MODEL for the policy checkpoint scripts/train.py loads.
# RUN_LIMIT: caps training rows after RUN_SUBSET is applied, "0" (default) = no cap.
# RUN_OUT_DIR: directory to write run artefacts (checkpoints, logs, metrics).
# RUN_MULTI: "1" to turn on multi-constraint reverse edges (Overrides.multi, H6 part B).

# ---------------------------------------------------------------------------
# Trainer step sizes (section 2.4-2.6: sampled once per optimiser step from
# the counterfactual-edge buffer / feasible buffer; not paper-numbered, no
# default given). DEV_SET.
# ---------------------------------------------------------------------------

# Reverse-construction draws attempted per optimiser step (conpo/trainer.py's
# _generate_and_score_completions -> conpo/reverse.py:reverse_batch). DEV_SET.
REVERSE_DRAWS_PER_STEP = 8  # DEV_SET

# Edges sampled from the buffer for L_resp per OPTIMISER step, not per
# compute_loss call: conpo/trainer.py's compute_loss divides this by
# self.args.gradient_accumulation_steps (rounding up) so every micro-batch
# draws its share and the per-optimiser-step total stays at
# EDGE_PAIRS_PER_STEP regardless of accumulation
# (conpo/edges.py:EdgeBuffer.sample). DEV_SET.
EDGE_PAIRS_PER_STEP = 48  # DEV_SET

# Single-family edges sampled from the buffer for L_prompt per OPTIMISER
# step, divided across compute_loss calls the same way as
# EDGE_PAIRS_PER_STEP above (conpo/trainer.py's compute_loss). DEV_SET.
PROMPT_PAIRS_PER_STEP = 8  # DEV_SET

# ---------------------------------------------------------------------------
# Batched veto (user decision 2026-09-02(d): the judge scores a burst of
# pending pairs every VETO_PERIOD_K optimiser steps instead of one HTTP call
# per candidate, so it never has to sit resident beside training on the
# two-H100 estate -- conpo/judge.py's PendingQueue/VLLMBatchedJudge/
# HFBatchedJudge/build_judge, conpo/trainer.py's flush_pending_veto).
# ---------------------------------------------------------------------------

# Optimiser steps between veto bursts (RUN_VETO_PERIOD overrides). DEV_SET.
VETO_PERIOD_K = 25  # DEV_SET
# Forces an early burst once the pending queue grows this large. DEV_SET.
PENDING_QUEUE_MAX = 2000  # DEV_SET
# RUN_JUDGE_MODE default: "batched" (VLLMBatchedJudge/HFBatchedJudge, no
# resident judge server) or "http" (HttpJudge against local/judge_server.sh).
JUDGE_MODE_DEFAULT = "batched"

# ---------------------------------------------------------------------------
# H6 multi-constraint compositor (design 2026-09-03/2026-09-04): phi-
# coefficient pair statistics (conpo/compositor.py:PairStats) pick which
# constraint-family pairs are worth a joint (multi-family) counterfactual
# edge; conpo/reverse.py:reverse_batch draws one with probability
# MULTI_EDGE_SHARE when Overrides.multi (RUN_MULTI=1) is set. Moved here
# from conpo/compositor.py (part B; compositor.py now imports these names
# from this module instead of defining them).
# ---------------------------------------------------------------------------

# Phi coefficient floor a family pair must clear to be sample_family_set-
# eligible. DEV_SET.
PHI_MIN = 0.1  # DEV_SET
# Minimum contingency-table sample count (PairStats.count) before a pair is
# trusted at all. DEV_SET.
MIN_COUNT = 20  # DEV_SET
# Softmax temperature over eligible pairs' phi/eta scores in
# sample_family_set (mirrors selection.mu's ETA). DEV_SET.
ETA_PHI = 0.2  # DEV_SET
# sample_family_set's default target-set size; only size=2 is implemented
# (compositor.py's own NotImplementedError for anything else).
MAX_EDGE_FAMILIES = 2
# Share of reverse_batch draws that attempt a multi-family edge when a
# qualifying pair exists (Overrides.multi only; an empty sample_family_set
# draw always falls back to the ordinary single-family draw). DEV_SET.
MULTI_EDGE_SHARE = 0.5  # DEV_SET

# ---------------------------------------------------------------------------
# DPO baselines (locked decision 2026-09-02(c): vf_dpo, relax_dpo, edge_dpo
# arms; conpo/pairs.py builds the pairs, scripts/train_dpo.py trains on them)
# ---------------------------------------------------------------------------

# beta in the DPO loss -log sigmoid(beta * (logratio_chosen - logratio_rejected)),
# scripts/train_dpo.py's DPOConfig(beta=...). IFBench's own published DPO
# baseline setting is beta=5, learning rate 5e-7; this constant follows that
# beta (scripts/train_dpo.py keeps learning_rate=config.LR, matching every
# other arm, rather than adopting 5e-7 too). DEV_SET: not separately tuned on
# ConPO's own development set.
DPO_BETA = 5.0  # DEV_SET

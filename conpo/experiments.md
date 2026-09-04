# Experiment graph

This is the run plan for the locked 2026-09-02 scope: `Qwen/Qwen3-4B`
non-thinking, LoRA r=64 for every run, and IF-RLVR (GRPO with
open-instruct's additive per-constraint `IFEvalVerifier` reward, compute-
matched to ConPO) as the headline baseline in place of the paper's
solve-only comparator. Every arm below maps to `RUN_ARM` plus, for the
ablations, one env-override on top of `RUN_ARM=full`. The RUN_* names,
their defaults, the five arms and the five override knobs are the contract
in `conpo/arms.py` (`ArmSpec`, `Overrides`, `resolve_arm`) and
`README.md`; `config.py`'s own RUN_* docstring covers only `RUN_ARM`
through `RUN_OUT_DIR` and does not yet list `ifrlvr` or the override names.

## Arms

| Arm | How to run it | Seeds | What it is |
|---|---|---|---|
| full | `RUN_ARM=full` | 42, 43, 44 | Full ConPO: forward repair edges + reverse corruption edges, transfer-aware active selection, `L_prompt`, `L_clean`, quality veto on |
| ifrlvr | `RUN_ARM=ifrlvr` | 42, 43, 44 | Headline baseline: GRPO with the additive per-constraint IFEvalVerifier reward (mean of per-constraint 0/1 verdicts, empty response scores 0), same G, token cap, LoRA, LR as `full` |
| solve_only | `RUN_ARM=solve_only` | 42 | ConPO with edges off: `L_clean` only (GRPO on the binary full-compliance reward `F(x,y)`). Out of the headline table; kept runnable for attribution |
| forward_only | `RUN_ARM=forward_only` | 42 | `full` with reverse construction off: forward repair edges + `L_clean` only |
| reverse_only | `RUN_ARM=reverse_only` | 42 | `full` with forward construction off: reverse corruption edges + `L_clean` only |
| RUN_SELECTION=uniform | `RUN_ARM=full RUN_SELECTION=uniform` | 42 | Active-selection ablation: `mu(c\|x)` replaced by uniform sampling over `C(x)` |
| RUN_SELECTION=half | `RUN_ARM=full RUN_SELECTION=half` | 42 | Active-selection ablation: `mu(c\|x)` replaced by 50%-success-rate sampling |
| RUN_NO_VETO | `RUN_ARM=full RUN_NO_VETO=1` | 42 | Quality veto off: `scripts/train.py:build_judge` returns `StaticJudge(keep=True)` instead of the batched judge. The batched veto is the default judge placement (USER DECISIONS 2026-09-02): every other arm builds a `VLLMBatchedJudge`/`HFBatchedJudge` (`RUN_JUDGE_MODE=batched`, the default) that loads/unloads on the same card in periodic bursts (`ConPOTrainer.flush_pending_veto`), needing no judge server; `RUN_JUDGE_MODE=http` is a separate fallback placement against `slurm/judge_server.sbatch`, which `slurm/train.sbatch` waits for only in that mode |
| RUN_NO_EXACTNESS | `RUN_ARM=full RUN_NO_EXACTNESS=1` | 42 | Corruption exactness admission off: reverse edges no longer required to satisfy `V(x, y_A^-) = A` exactly |
| RUN_NO_DISTANCE | `RUN_ARM=full RUN_NO_DISTANCE=1` | 42 | Distance bound `delta` off on both directions: minimality (`d_Y <= delta`) no longer gates admission |
| RUN_LAMBDA_P=0 | `RUN_ARM=full RUN_LAMBDA_P=0` | 42 | Prompt counterfactuals off: the `L_prompt` term is zeroed |

`RUN_SELECTION`, `RUN_NO_VETO`, `RUN_NO_EXACTNESS`, `RUN_NO_DISTANCE` and
`RUN_LAMBDA_P` are overrides on top of `RUN_ARM=full`, not separate arm
names; `slurm/train.sbatch` passes them through unmodified via
`--export=ALL,<name>` (see README.md's cluster launch sequence).

## Hypotheses

`ConPO.pdf` section 4.1 states six hypotheses (H1-H6). H1 is restated below
against the locked headline baseline; H2-H5 are unchanged from the paper.

| H | Claim | Arms compared | Deciding metric |
|---|---|---|---|
| H1 | Verifier-dominating counterfactual pairs improve first-pass compliance beyond the RLVR baseline | `full` vs `ifrlvr` (base, untrained, and `solve_only` as reference floors) | IFBench prompt-level loose accuracy (external, the 58 unseen `IFBench_test` families) |
| H2 | Exact, minimal pairs outperform independent or non-minimal preference pairs | `reverse_only` vs `RUN_NO_EXACTNESS`, `RUN_NO_DISTANCE` | Internal PAcc splits |
| H3 | Forward repair and reverse corruption provide complementary relevance and coverage | `full` vs `forward_only` vs `reverse_only` | Internal PAcc splits |
| H4 | Transfer-aware selection is more compute-efficient than uniform or 50%-success sampling | `full` vs `RUN_SELECTION=uniform` vs `RUN_SELECTION=half` | Internal PAcc splits, normalised by generated tokens and by admitted edges |
| H5 | Prompt counterfactuals improve constraint-conditional preferences | `full` vs `RUN_LAMBDA_P=0` | Internal PAcc splits, alongside the `Gamma_theta` diagnostic |
| H6 | Interaction-selected multi-constraint examples improve compositional generalisation | none yet | Deferred: needs the phi-coefficient multi-constraint compositor (M4+, `ConPO-Build-Plan.md` section 10), not built in this scope |

`RUN_NO_VETO` does not decide a single H. It is a robustness check on H2/H3:
whether the quality veto (discarding degraded pairs before admission) is
load-bearing for the coverage and exactness gains those hypotheses claim.
Read it against the mechanism metrics (veto rates, blinded pairwise judge
quality vs the initial policy), not a single PAcc number.

Internal PAcc splits are the four sets in `conpo/data.py`: (i) unseen
prompts x familiar constraints, (ii) unseen family combinations, (iii)
higher simultaneous constraint counts, (iv) natural errors and held-out
operator variants.

### Implementation status

`conpo/arms.py:HYPOTHESES` is the executable pre-registration for a subset
of the edges above: `H1` (`full` vs `ifrlvr`) plus two attribution edges
(`full` vs `solve_only`, `solve_only` vs `ifrlvr`), `H3_forward` /
`H3_reverse` (`full` vs each single-direction arm), `H4_uniform` /
`H4_half` (`full` vs each selection override), and `H5_lambda_p_zero`
(`full` vs the `lambda_p=0` override). Two differences from the table
above, both open items rather than contradictions: every entry there is
currently keyed to one metric name, `ifbench_prompt_loose`, since the
automated harness wiring the internal-PAcc comparisons in `eval_internal.py`
(H2-H5's deciding metric per this doc) into that registry has not landed
yet; and `H2` (exactness/minimality) and `H6` are not pre-registered there
at all -- `H2` has no dedicated arm of its own (it reads `reverse_only`
against the `RUN_NO_EXACTNESS`/`RUN_NO_DISTANCE` overrides, both of which
exist as knobs but not yet as a registered comparison), and `H6` waits on
the multi-constraint compositor as noted above.

## Seeds

`full` and `ifrlvr` run 3 seeds each (42, 43, 44), matched to the locked H1
comparison. Every other arm -- the three ConPO-family ablations
(`solve_only`, `forward_only`, `reverse_only`) and the six env-override
ablations -- runs 1 seed (42), since H2-H5 read a direction of effect
against `full`, not a confidence interval.

## Run count

| Bucket | Runs | Notes |
|---|---|---|
| Hypothesis graph | 16 | 11 arms (above) at the seed schedule = 15 trained runs, plus the untrained base model evaluated once (no training, the H1 floor) |
| Baselines outside the graph | 2 | (1) inference-time verify-and-repair: the initial policy plus the same repair operator and judge loop applied at test time only, a system-level reference with a different inference budget (`ConPO.pdf` section 4.1); (2) IF-RLVR run with its own paper's literal settings (16 samples/prompt, 2048 tokens, lr 5e-7, full fine-tune) rather than token-matched to ConPO, a labelled replication (`ConPO-Build-Plan.md` section 7) |
| Model ablations | 2 | `Qwen/Qwen3-8B` and `allenai/Olmo-3-7B-Instruct-SFT`, each `RUN_ARM=full` only, 1 seed; `config.LATER_ABLATION_MODELS` also names `allenai/Llama-3.1-Tulu-3-8B-SFT` but it is not part of this run count |
| Full fine-tune point-check | 1 | `Qwen/Qwen3-4B`, `RUN_ARM=full`, full fine-tune instead of LoRA, on both H100s (`--gres=gpu:h100:2`), 1 seed |
| **Total** | **21** | |

"""Training arms and their ablation overrides (ConPO.pdf section 4.1).

An ArmSpec is a static recipe: which reward function trains the policy and
which counterfactual-construction mechanisms are switched on. Overrides layer
the RUN_* ablation knobs (selection rule, veto/exactness/distance-gate
disables, lambda_P override) on top of whichever arm is running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from conpo import config


@dataclass(frozen=True)
class ArmSpec:
    """One training arm's mechanism switches.

    reward: "conjunctive" (verifiers.feasible, the paper's F(x, y)) or
    "additive" (open-instruct's IFEvalVerifier mean-per-constraint reward,
    the IF-RLVR baseline). forward/reverse: whether that counterfactual
    direction runs at all. prompt_loss/resp_loss: whether L_prompt/L_resp
    are added to the training objective. trainer: "grpo" (conpo.trainer.
    ConPOTrainer, scripts/train.py) or "dpo" (trl.DPOTrainer on
    conpo.pairs-built pairs, scripts/train_dpo.py) -- which script and
    trainer class the arm runs through. For "dpo" arms,
    reward/forward/reverse/prompt_loss/resp_loss describe no mechanism
    scripts/train_dpo.py reads (that script trains on a fixed pairs file
    instead); they are set to solve_only's values (no counterfactual
    construction, conjunctive reward as the closest no-op default) purely
    so every ArmSpec stays fully populated.
    """

    reward: str
    forward: bool
    reverse: bool
    prompt_loss: bool
    resp_loss: bool
    trainer: str = "grpo"


ARMS: dict[str, ArmSpec] = {
    # ConPO in full: both construction directions, both counterfactual losses.
    "full": ArmSpec(reward="conjunctive", forward=True, reverse=True, prompt_loss=True, resp_loss=True, trainer="grpo"),
    # IF-RLVR baseline (locked decision 2026-09-02): compute-matched GRPO
    # with open-instruct's additive per-constraint reward, no counterfactual
    # construction at all.
    "ifrlvr": ArmSpec(
        reward="additive", forward=False, reverse=False, prompt_loss=False, resp_loss=False, trainer="grpo"
    ),
    # Solve-only ablation: conjunctive reward, no counterfactual construction.
    "solve_only": ArmSpec(
        reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=False, trainer="grpo"
    ),
    # H3 ablations: one construction direction at a time.
    "forward_only": ArmSpec(
        reward="conjunctive", forward=True, reverse=False, prompt_loss=True, resp_loss=True, trainer="grpo"
    ),
    "reverse_only": ArmSpec(
        reward="conjunctive", forward=False, reverse=True, prompt_loss=True, resp_loss=True, trainer="grpo"
    ),
    # DPO baselines (locked decision 2026-09-02(c), conpo/pairs.py +
    # scripts/train_dpo.py): three fixed pair sources, one plain DPO
    # objective (trl.DPOTrainer), no ConPOTrainer/GRPO mechanism involved.
    # vf_dpo: verifier-filtered DPO on independently sampled pairs from the
    # initial policy (the AutoIF / Tulu 3 shape, conpo.pairs.build_vf_pairs).
    "vf_dpo": ArmSpec(
        reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=False, trainer="dpo"
    ),
    # relax_dpo: relaxation-pair DPO in the Tulu 3 If-Persona-Pref shape
    # (conpo.pairs.build_relax_pairs, conpo.prompts.counterfactual_prompts).
    "relax_dpo": ArmSpec(
        reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=False, trainer="dpo"
    ),
    # edge_dpo: plain DPO on a ConPO run's dumped edge buffer
    # (conpo.pairs.build_edge_pairs) -- isolates ConPO's objective from its
    # counterfactual-construction machinery.
    "edge_dpo": ArmSpec(
        reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=False, trainer="dpo"
    ),
}


def validate(name: str) -> None:
    """Raise ValueError if `name` is not a known arm."""
    if name not in ARMS:
        raise ValueError(f"unknown arm: {name!r}, expected one of {sorted(ARMS)}")


@dataclass(frozen=True)
class Overrides:
    """RUN_* ablation knobs, layered on top of whichever arm is running.

    selection: "transfer" (mu(c|x), the paper's active selection),
    "uniform", or "half" (conpo.selection / conpo.reverse selection modes).
    no_veto/no_exactness/no_distance: disable that admission gate (reverse.py
    exactness/distance_gate, forward.py/reverse.py veto). lambda_p: override
    for config.LAMBDA_P (H5: lambda_P = 0 ablates L_prompt's contribution
    even when arm.prompt_loss is True). multi (H6 part B, RUN_MULTI=1):
    conpo.trainer.ConPOTrainer owns a conpo.compositor.PairStats and passes
    it to conpo.reverse.reverse_batch, which then attempts a multi-family
    edge on a share of its draws (config.MULTI_EDGE_SHARE) instead of always
    drawing a single family. Only meaningful when the arm reverse-
    constructs at all (arm.reverse=True, i.e. "full" or "reverse_only");
    "forward_only" and any reverse=False arm ignore it since no reverse
    draw ever happens.
    """

    selection: str = "transfer"
    no_veto: bool = False
    no_exactness: bool = False
    no_distance: bool = False
    # default_factory, not `= config.LAMBDA_P`: a plain default binds
    # config.LAMBDA_P once at import time, so a test's monkeypatch of
    # config.LAMBDA_P after import would be silently ignored.
    lambda_p: float = field(default_factory=lambda: config.LAMBDA_P)
    multi: bool = False


def resolve_arm(name: str) -> tuple[ArmSpec, Overrides]:
    """The named ArmSpec plus Overrides read from the RUN_* environment."""
    validate(name)
    overrides = Overrides(
        selection=config.env("RUN_SELECTION", "transfer"),
        no_veto=config.env("RUN_NO_VETO", "0") == "1",
        no_exactness=config.env("RUN_NO_EXACTNESS", "0") == "1",
        no_distance=config.env("RUN_NO_DISTANCE", "0") == "1",
        lambda_p=float(config.env("RUN_LAMBDA_P", str(config.LAMBDA_P))),
        multi=config.env("RUN_MULTI", "0") == "1",
    )
    return ARMS[name], overrides


class Hypothesis(NamedTuple):
    """One pre-registered comparison (ConPO.pdf section 4.1).

    treatment/control name an arm in ARMS, or an arm plus the Overrides
    field that ablates it (e.g. "full:selection=uniform",
    "full:lambda_p=0") for the H4/H5 rows, which compare `full` against
    itself under one disabled mechanism rather than against a different arm.
    expected_sign is ">" when treatment's metric is expected to exceed
    control's under the hypothesis.
    """

    id: str
    treatment: str
    control: str
    metric: str
    expected_sign: str


HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis("H1", "full", "ifrlvr", "ifbench_prompt_loose", ">"),
    Hypothesis("H1_attribution_full_vs_solve_only", "full", "solve_only", "ifbench_prompt_loose", ">"),
    Hypothesis("H1_attribution_solve_only_vs_ifrlvr", "solve_only", "ifrlvr", "ifbench_prompt_loose", ">"),
    Hypothesis("H3_forward", "full", "forward_only", "ifbench_prompt_loose", ">"),
    Hypothesis("H3_reverse", "full", "reverse_only", "ifbench_prompt_loose", ">"),
    Hypothesis("H4_uniform", "full", "full:selection=uniform", "ifbench_prompt_loose", ">"),
    Hypothesis("H4_half", "full", "full:selection=half", "ifbench_prompt_loose", ">"),
    Hypothesis("H5_lambda_p_zero", "full", "full:lambda_p=0", "ifbench_prompt_loose", ">"),
    # H2 proxy (locked decision 2026-09-02(c)): full vs the verifier-filtered
    # independent-pairs DPO baseline, on the one metric this registry wires
    # today (see the module docstring's ifbench_prompt_loose note above).
    Hypothesis("H2_vf_dpo", "full", "vf_dpo", "ifbench_prompt_loose", ">"),
    # Objective-isolation row: full vs plain DPO on full's own dumped edge
    # buffer, isolating ConPO's contrastive objective from its
    # counterfactual-construction machinery (same edges, different training).
    Hypothesis("objective_isolation_edge_dpo", "full", "edge_dpo", "ifbench_prompt_loose", ">"),
    # H6 (multi-constraint compositional generalisation, design 2026-09-03:
    # H6 is a built arm): full with RUN_MULTI=1 (Overrides.multi, this
    # module's resolve_arm; conpo/compositor.py, conpo/reverse.py,
    # conpo/trainer.py) vs full, no overrides. "full:multi=1" is the same
    # "arm:field=value" identifier convention H4/H5 use against `full`
    # itself (see resolve_arm's Overrides field names). The metric string
    # names both PAcc splits the plan's H6 row reads (eval_ii and eval_iii)
    # while staying parseable by scripts/analyze.py's _parse_metric (a
    # "<source>_prompt_strict"/"<source>_prompt_loose" suffix, same as
    # every other row's placeholder metric field) -- scripts/dashboard.py
    # derives the real per-split metric from the id's H6 prefix instead
    # (pacc_eval_iii for the Delta/Verdict cells, eval_iii being the harder
    # split; "PAcc eval_ii, eval_iii" for the Metric column), the same
    # pattern it already uses for H2-H5 (see dashboard.py's
    # hypothesis_metric/hypothesis_metric_display).
    Hypothesis("H6_multi", "full:multi=1", "full", "pacc_eval_ii_and_pacc_eval_iii_prompt_loose", ">"),
)

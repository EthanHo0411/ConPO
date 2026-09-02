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
    are added to the training objective.
    """

    reward: str
    forward: bool
    reverse: bool
    prompt_loss: bool
    resp_loss: bool


ARMS: dict[str, ArmSpec] = {
    # ConPO in full: both construction directions, both counterfactual losses.
    "full": ArmSpec(reward="conjunctive", forward=True, reverse=True, prompt_loss=True, resp_loss=True),
    # IF-RLVR baseline (locked decision 2026-09-02): compute-matched GRPO
    # with open-instruct's additive per-constraint reward, no counterfactual
    # construction at all.
    "ifrlvr": ArmSpec(reward="additive", forward=False, reverse=False, prompt_loss=False, resp_loss=False),
    # Solve-only ablation: conjunctive reward, no counterfactual construction.
    "solve_only": ArmSpec(reward="conjunctive", forward=False, reverse=False, prompt_loss=False, resp_loss=False),
    # H3 ablations: one construction direction at a time.
    "forward_only": ArmSpec(reward="conjunctive", forward=True, reverse=False, prompt_loss=True, resp_loss=True),
    "reverse_only": ArmSpec(reward="conjunctive", forward=False, reverse=True, prompt_loss=True, resp_loss=True),
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
    even when arm.prompt_loss is True).
    """

    selection: str = "transfer"
    no_veto: bool = False
    no_exactness: bool = False
    no_distance: bool = False
    # default_factory, not `= config.LAMBDA_P`: a plain default binds
    # config.LAMBDA_P once at import time, so a test's monkeypatch of
    # config.LAMBDA_P after import would be silently ignored.
    lambda_p: float = field(default_factory=lambda: config.LAMBDA_P)


def resolve_arm(name: str) -> tuple[ArmSpec, Overrides]:
    """The named ArmSpec plus Overrides read from the RUN_* environment."""
    validate(name)
    overrides = Overrides(
        selection=config.env("RUN_SELECTION", "transfer"),
        no_veto=config.env("RUN_NO_VETO", "0") == "1",
        no_exactness=config.env("RUN_NO_EXACTNESS", "0") == "1",
        no_distance=config.env("RUN_NO_DISTANCE", "0") == "1",
        lambda_p=float(config.env("RUN_LAMBDA_P", str(config.LAMBDA_P))),
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
)

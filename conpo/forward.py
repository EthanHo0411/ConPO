"""Forward (repair) counterfactual construction (ConPO.pdf section 2.2).

A response that fails some verifiers is repaired by the current policy: a
fixed-template prompt shows the failed verifiers' ids and their
`verifiers.describe_failure` text, the policy proposes at most
`max_candidates` rewrites per step for at most `max_steps` steps, and a
rewrite is admitted only when it fixes at least one previously failed
verifier without breaking any previously satisfied one (`monotone_admit`),
stays within the fidelity bound d_Y <= delta, and passes the quality veto.
Every admitted transition is a forward `Edge`; the trajectory's final
response is recorded as a feasible endpoint when it satisfies every
constraint.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from conpo import config
from conpo.data import PoolRow, TokenMeter
from conpo.distance import token_distance
from conpo.edges import Edge
from conpo.engine import Engine
from conpo.judge import Verdict, VetoRequest
from conpo.verifiers import Constraint, describe_failure, description, verifier_vector


class Judge(Protocol):
    """Structural contract shared by `judge.HttpJudge` and `judge.StaticJudge`."""

    def veto(self, req: VetoRequest) -> Verdict: ...


_REPAIR_INSTRUCTIONS = (
    "The response above does not satisfy every requirement of the task. It "
    "fails the following requirement(s):\n\n{failures}\n\n"
    "Rewrite the response so that it satisfies ONLY the requirement(s) listed "
    "above, changing as little else about the response as possible. Output "
    "nothing but the rewritten response: no preamble, no explanation, no "
    "markdown fences."
)


def build_repair_messages(
    row: PoolRow, response: str, failed: Sequence[tuple[Constraint, str]]
) -> list[dict[str, str]]:
    """Three-turn repair conversation.

    Turn 1 is the original prompt, turn 2 is the response that failed some
    verifiers, and turn 3 is a fixed instruction listing, for each entry in
    `failed`, the constraint's id next to its `describe_failure` text (which
    already carries the requirement sentence).
    """
    failures = "\n".join(f"- {c.iid}: {failure_text}" for c, failure_text in failed)
    return [
        {"role": "user", "content": row.prompt},
        {"role": "assistant", "content": response},
        {"role": "user", "content": _REPAIR_INSTRUCTIONS.format(failures=failures)},
    ]


def monotone_admit(
    constraints: Sequence[Constraint],
    before: Sequence[bool],
    after: Sequence[bool],
    d: float,
    delta: float = config.DELTA,
    distance_gate: bool = True,
) -> tuple[bool, str]:
    """Repair-proposal admission gate (section 2.2).

    Admitted only if no previously satisfied verifier regresses, at least
    one previously failed verifier now passes, and (when `distance_gate`,
    the RUN_NO_DISTANCE ablation's off-switch, matching reverse.py's own
    `distance_gate` parameter) the fidelity bound d_Y(y_t, y_{t+1}) <= delta
    holds. `before`/`after`/`constraints` are index-aligned. Reason strings:
    "regression:<iid>" (naming the first constraint that regressed),
    "no_improvement", "distance", "admitted".
    """
    for c, was_ok, is_ok in zip(constraints, before, after):
        if was_ok and not is_ok:
            return False, f"regression:{c.iid}"
    if not any(not was_ok and is_ok for was_ok, is_ok in zip(before, after)):
        return False, "no_improvement"
    if distance_gate and d > delta:
        return False, "distance"
    return True, "admitted"


@dataclass
class RepairResult:
    """Outcome of one or more repair trajectories.

    Edge.logp_ref_plus/minus are left as `nan` and Edge.len_plus/minus as 0:
    placeholders the trainer fills in once it has a reference-model forward
    pass available, so this module never has to load the policy as a scorer.

    endpoint_sources is index-aligned with feasible_endpoints: entry i is the
    (possibly empty) list of edges the trajectory that produced
    feasible_endpoints[i] built along the way. A caller under the batched
    veto (conpo/trainer.py, USER DECISIONS 2026-09-02) needs this to know an
    endpoint is only as trustworthy as those edges' real veto verdicts turn
    out to be -- the endpoint's own text was reached under a placeholder
    keep=True from _QueueingVetoCollector, not a real judgment.
    """

    edges: list[Edge] = field(default_factory=list)
    feasible_endpoints: list[str] = field(default_factory=list)
    endpoint_sources: list[list[Edge]] = field(default_factory=list)
    proposals: int = 0
    admitted: int = 0
    generated_tokens: int = 0
    reasons: Counter[str] = field(default_factory=Counter)


def repair_trajectory(
    row: PoolRow,
    response: str,
    engine: Engine,
    judge: Judge,
    rng: random.Random,
    meter: TokenMeter,
    current_update: int,
    max_steps: int = config.MAX_STEPS,
    max_candidates: int = config.MAX_CANDIDATES,
    veto: bool = True,
    distance_gate: bool = True,
) -> RepairResult:
    """Run at most `max_steps` repair steps on `response` (section 2.2).

    Every admitted transition becomes a forward `Edge`, `step` set to the
    repair-step index it was admitted at. The trajectory's final response
    (whether or not any step was admitted) is recorded as a feasible
    endpoint if it satisfies every constraint in `row.constraints`.
    `distance_gate=False` (RUN_NO_DISTANCE) threads through to every
    `monotone_admit` call, same convention as reverse.py's own
    `distance_gate` parameter.
    """
    result = RepairResult()
    y_t = response
    before = verifier_vector(row.constraints, y_t)

    for step in range(max_steps):
        if all(before):
            break
        failed_idx = [i for i, ok in enumerate(before) if not ok]
        failed = [(row.constraints[i], describe_failure(row.constraints[i], y_t)) for i in failed_idx]
        messages = build_repair_messages(row, y_t, failed)

        proposals = engine.generate(
            [messages],
            max_candidates,
            config.MAX_COMPLETION_TOKENS,
            config.TEMPERATURE,
            config.TOP_P,
            rng.randint(0, 2**31 - 1),
        )[0]
        result.proposals += len(proposals)
        n_tokens = engine.count_tokens(proposals)
        meter.add(n_tokens)
        result.generated_tokens += n_tokens

        admitted_y: str | None = None
        admitted_after: tuple[bool, ...] = ()
        admitted_reason = ""
        admitted_distance = 0.0
        for candidate in proposals:
            d = token_distance(y_t, candidate)
            after = verifier_vector(row.constraints, candidate)
            ok, reason = monotone_admit(row.constraints, before, after, d, distance_gate=distance_gate)
            if not ok:
                result.reasons[reason] += 1
                continue
            newly = [row.constraints[i] for i in failed_idx if after[i]]
            if veto:
                verdict = judge.veto(
                    VetoRequest(
                        prompt=row.prompt,
                        y_plus=candidate,
                        y_minus=y_t,
                        families=tuple(c.iid for c in newly),
                        descriptions=tuple(description(c) for c in newly),
                    )
                )
                if not verdict.keep:
                    result.reasons["veto"] += 1
                    continue
            admitted_y, admitted_after, admitted_reason, admitted_distance = candidate, after, reason, d
            break

        if admitted_y is None:
            break

        result.reasons[admitted_reason] += 1
        result.admitted += 1
        result.edges.append(
            Edge(
                prompt_id=row.key,
                prompt=row.prompt,
                y_plus=admitted_y,
                y_minus=y_t,
                families=frozenset(row.constraints[i].iid for i in failed_idx if admitted_after[i]),
                direction="forward",
                step=step,
                logp_ref_plus=float("nan"),
                logp_ref_minus=float("nan"),
                len_plus=0,
                len_minus=0,
                created_update=current_update,
                meta={"reason": admitted_reason, "distance": admitted_distance},
            )
        )
        y_t = admitted_y
        before = admitted_after

    if all(before):
        result.feasible_endpoints.append(y_t)
        result.endpoint_sources.append(list(result.edges))
    return result


def repair_failed_group(
    row: PoolRow,
    responses: Sequence[str],
    engine: Engine,
    judge: Judge,
    rng: random.Random,
    meter: TokenMeter,
    current_update: int,
    max_steps: int = config.MAX_STEPS,
    max_candidates: int = config.MAX_CANDIDATES,
    veto: bool = True,
    distance_gate: bool = True,
) -> RepairResult:
    """Run `repair_trajectory` on every failed response of one clean group
    and merge the results (section 2.2). `distance_gate` passes straight
    through to every `repair_trajectory` call."""
    merged = RepairResult()
    for response in responses:
        r = repair_trajectory(
            row, response, engine, judge, rng, meter, current_update, max_steps, max_candidates, veto, distance_gate
        )
        merged.edges.extend(r.edges)
        merged.feasible_endpoints.extend(r.feasible_endpoints)
        merged.endpoint_sources.extend(r.endpoint_sources)
        merged.proposals += r.proposals
        merged.admitted += r.admitted
        merged.generated_tokens += r.generated_tokens
        merged.reasons.update(r.reasons)
    return merged

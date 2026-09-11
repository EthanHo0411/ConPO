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


def _admit_candidates(
    row: PoolRow,
    y_t: str,
    before: Sequence[bool],
    failed_idx: Sequence[int],
    proposals: Sequence[str],
    judge: Judge,
    veto: bool,
    distance_gate: bool,
    reasons: Counter[str],
) -> tuple[str | None, tuple[bool, ...], str, float]:
    """Try `proposals` in order against `monotone_admit` then the veto (section
    2.2's inner candidate loop, factored out so repair_trajectory and
    repair_failed_group's lock-step batch share one admission path -- design
    choice, Ethan devbox perf pass, 2026-09-11).

    Every rejection (`monotone_admit` reason, or "veto") is tallied into
    `reasons` as it happens; the winning candidate's "admitted" reason is
    left for the caller to add once, matching the single post-loop tally the
    unbatched version always used. Returns (None, (), "", 0.0) when nothing
    in `proposals` is admitted.
    """
    for candidate in proposals:
        d = token_distance(y_t, candidate)
        after = verifier_vector(row.constraints, candidate)
        ok, reason = monotone_admit(row.constraints, before, after, d, distance_gate=distance_gate)
        if not ok:
            reasons[reason] += 1
            continue
        if veto:
            newly = [row.constraints[i] for i in failed_idx if after[i]]
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
                reasons["veto"] += 1
                continue
        return candidate, after, reason, d
    return None, (), "", 0.0


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

        admitted_y, admitted_after, admitted_reason, admitted_distance = _admit_candidates(
            row, y_t, before, failed_idx, proposals, judge, veto, distance_gate, result.reasons
        )
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


@dataclass
class _Trajectory:
    """One response's mutable repair state inside repair_failed_group's
    lock-step batch: the same y_t/before/edges a repair_trajectory call
    would carry as locals, kept per-response so every trajectory can advance
    independently between shared batched engine.generate calls."""

    y_t: str
    before: tuple[bool, ...]
    edges: list[Edge] = field(default_factory=list)


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
    """Run repair on every failed response of one clean group, lock-step
    across responses (section 2.2).

    Design choice (Ethan devbox perf pass, 2026-09-11): at each repair step,
    every still-active trajectory's proposals are drawn in ONE
    `engine.generate` call (messages_batch holding one entry per active
    response) instead of the one-call-per-response-per-step the old
    per-response loop made -- `engine.generate`'s own contract returns
    completions in `messages_batch` order, so this changes only how many
    engine round trips a step costs, never which candidates a response sees:
    each trajectory's own gate checks, veto calls and admitted edges are
    produced by the same `_admit_candidates` path repair_trajectory uses,
    fed that trajectory's own proposals in the same per-step order. A
    trajectory drops out of the batch the first step it has nothing left to
    admit or is already fully feasible, exactly when repair_trajectory would
    have `break` / stopped looping for it.

    tests/test_forward.py's FakeEngine-based equivalence test pins this: a
    content-keyed FakeEngine (ignores call order and the shared per-step
    seed) admits identical edges whether responses are repaired one at a
    time or batched this way.
    """
    trajs = [_Trajectory(y_t=r, before=tuple(verifier_vector(row.constraints, r))) for r in responses]
    active = [i for i, t in enumerate(trajs) if not all(t.before)]
    merged = RepairResult()

    for step in range(max_steps):
        if not active:
            break
        failed_idx_by_pos: list[list[int]] = []
        messages_batch = []
        for i in active:
            t = trajs[i]
            failed_idx = [j for j, ok in enumerate(t.before) if not ok]
            failed = [(row.constraints[j], describe_failure(row.constraints[j], t.y_t)) for j in failed_idx]
            messages_batch.append(build_repair_messages(row, t.y_t, failed))
            failed_idx_by_pos.append(failed_idx)

        all_proposals = engine.generate(
            messages_batch,
            max_candidates,
            config.MAX_COMPLETION_TOKENS,
            config.TEMPERATURE,
            config.TOP_P,
            rng.randint(0, 2**31 - 1),
        )

        still_active: list[int] = []
        for pos, i in enumerate(active):
            t = trajs[i]
            failed_idx = failed_idx_by_pos[pos]
            proposals = all_proposals[pos]
            merged.proposals += len(proposals)
            n_tokens = engine.count_tokens(proposals)
            meter.add(n_tokens)
            merged.generated_tokens += n_tokens

            admitted_y, admitted_after, admitted_reason, admitted_distance = _admit_candidates(
                row, t.y_t, t.before, failed_idx, proposals, judge, veto, distance_gate, merged.reasons
            )
            if admitted_y is None:
                continue

            merged.reasons[admitted_reason] += 1
            merged.admitted += 1
            edge = Edge(
                prompt_id=row.key,
                prompt=row.prompt,
                y_plus=admitted_y,
                y_minus=t.y_t,
                families=frozenset(row.constraints[j].iid for j in failed_idx if admitted_after[j]),
                direction="forward",
                step=step,
                logp_ref_plus=float("nan"),
                logp_ref_minus=float("nan"),
                len_plus=0,
                len_minus=0,
                created_update=current_update,
                meta={"reason": admitted_reason, "distance": admitted_distance},
            )
            t.edges.append(edge)
            merged.edges.append(edge)
            t.y_t = admitted_y
            t.before = admitted_after
            if not all(t.before):
                still_active.append(i)
        active = still_active

    for t in trajs:
        if all(t.before):
            merged.feasible_endpoints.append(t.y_t)
            merged.endpoint_sources.append(list(t.edges))
    return merged

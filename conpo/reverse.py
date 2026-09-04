"""Reverse counterfactual construction (ConPO.pdf section 2.3).

Draws a feasible response y_plus, samples a constraint family present on the
prompt via mu(c | x) (or an ablation selection rule), applies that family's
registered corruption operator, and admits the pair (y_plus, y_minus) only
when exactness V(x, y_minus) = {c} holds, the fidelity bound d_Y <= delta
holds, and the quality veto keeps it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from conpo import config
from conpo.data import PoolRow
from conpo.distance import token_distance, within_delta
from conpo.edges import Edge, FeasibleBuffer
from conpo.judge import Verdict, VetoRequest
from conpo.operators import registry
from conpo.selection import FamilyStats, sample_family
from conpo.verifiers import Constraint, check, description


class Judge(Protocol):
    """Structural contract shared by `judge.HttpJudge`, `judge.StaticJudge`,
    and the batched-veto `_QueueingVetoCollector` (conpo/trainer.py) --
    matches forward.py's own local `Judge` Protocol so both modules accept
    the same set of callers without a nominal `HttpJudge | StaticJudge`
    union excluding anything structurally compatible."""

    def veto(self, req: VetoRequest) -> Verdict: ...


@dataclass(frozen=True)
class ReverseResult:
    """Outcome of one reverse_edge attempt. edge is None on any rejection."""

    edge: Edge | None
    family: str | None
    reason: str
    distance: float | None
    veto_kept: bool | None


def eligible_families(row: PoolRow) -> list[str]:
    """Families on `row` that have a registered corruption operator.

    First-appearance order, deduplicated.
    """
    seen: list[str] = []
    for c in row.constraints:
        if c.iid in registry.OPERATORS and c.iid not in seen:
            seen.append(c.iid)
    return seen


def _choose_family(selection: str, stats: FamilyStats, eligible: list[str], rng: random.Random) -> str:
    """Pick one family from `eligible` under the given selection mode."""
    if selection == "transfer":
        return sample_family(stats, eligible, rng)
    if selection == "uniform":
        return rng.choice(eligible)
    if selection == "half":
        weights = [stats.p_c(f) * (1.0 - stats.p_c(f)) for f in eligible]
        if sum(weights) <= 0.0:
            # ponytail: every p_c starts at 0 (weight 0 everywhere), and
            # random.choices raises on an all-zero weight vector; fall back
            # to uniform rather than crash. Add a warm-up-aware weighting if
            # the all-zero case ever needs to bias away from uniform.
            weights = [1.0] * len(eligible)
        return rng.choices(eligible, weights=weights, k=1)[0]
    raise ValueError(f"unknown selection mode: {selection!r}")


def _corrupt_without_exactness(
    target: Constraint, y_plus: str, rng: random.Random, max_candidates: int = 8
) -> tuple[str | None, str]:
    """Exactness-ablation corruption (ConPO.pdf section 2.3, ablated).

    First candidate that changes the text and fails the target check, with
    no test that other constraints still pass: collateral damage is allowed
    through. Mirrors registry.corrupt's candidate loop and cap, minus the
    exact() call.
    """
    for count, candidate in enumerate(registry.candidates(y_plus, target, rng)):
        if count >= max_candidates:
            break
        if candidate == y_plus:
            continue
        if not check(target, candidate):
            return candidate, "target_failed"
    return None, registry.NO_CANDIDATE


def reverse_edge(
    row: PoolRow,
    y_plus: str,
    stats: FamilyStats,
    judge: Judge,
    rng: random.Random,
    current_update: int,
    selection: str = "transfer",
    exactness: bool = True,
    distance_gate: bool = True,
    veto: bool = True,
) -> ReverseResult:
    """One reverse-construction attempt on a feasible response `y_plus`."""
    if not row.english:
        return ReverseResult(edge=None, family=None, reason="not_english", distance=None, veto_kept=None)

    eligible = eligible_families(row)
    if not eligible:
        return ReverseResult(edge=None, family=None, reason="no_eligible_family", distance=None, veto_kept=None)

    family = _choose_family(selection, stats, eligible, rng)
    target_index = next(i for i, c in enumerate(row.constraints) if c.iid == family)
    target = row.constraints[target_index]

    if exactness:
        corrupted, reason = registry.corrupt(row.constraints, target_index, y_plus, rng)
    else:
        corrupted, reason = _corrupt_without_exactness(target, y_plus, rng)
    if corrupted is None:
        return ReverseResult(edge=None, family=family, reason=reason, distance=None, veto_kept=None)

    d = token_distance(y_plus, corrupted)
    # delta is read fresh from config so a monkeypatched config.DELTA takes
    # effect: within_delta's own default parameter is bound once at import.
    if distance_gate and not within_delta(y_plus, corrupted, delta=config.DELTA):
        return ReverseResult(edge=None, family=family, reason="distance", distance=d, veto_kept=None)

    veto_kept: bool | None = None
    if veto:
        verdict = judge.veto(
            VetoRequest(
                prompt=row.prompt,
                y_plus=y_plus,
                y_minus=corrupted,
                families=(family,),
                descriptions=(description(target),),
            )
        )
        veto_kept = verdict.keep
        if not verdict.keep:
            return ReverseResult(edge=None, family=family, reason="veto", distance=d, veto_kept=False)

    edge = Edge(
        prompt_id=row.key,
        prompt=row.prompt,
        y_plus=y_plus,
        y_minus=corrupted,
        families=frozenset({family}),
        direction="reverse",
        step=0,
        logp_ref_plus=math.nan,
        logp_ref_minus=math.nan,
        len_plus=0,
        len_minus=0,
        created_update=current_update,
        meta={"reason": reason, "distance": d, "selection": selection},
    )
    stats.record_admission(family)
    return ReverseResult(edge=edge, family=family, reason="admitted", distance=d, veto_kept=veto_kept)


def reverse_batch(
    rows_by_key: Mapping[str, PoolRow],
    feasible: FeasibleBuffer,
    n: int,
    stats: FamilyStats,
    judge: Judge,
    rng: random.Random,
    current_update: int,
    selection: str = "transfer",
    exactness: bool = True,
    distance_gate: bool = True,
    veto: bool = True,
) -> list[ReverseResult]:
    """Draw `n` (prompt_id, y_plus) samples from `feasible` and attempt reverse_edge on each."""
    results: list[ReverseResult] = []
    for _ in range(n):
        prompt_id, y_plus = feasible.sample_prompt(rng)
        row = rows_by_key[prompt_id]
        results.append(
            reverse_edge(
                row,
                y_plus,
                stats,
                judge,
                rng,
                current_update,
                selection=selection,
                exactness=exactness,
                distance_gate=distance_gate,
                veto=veto,
            )
        )
    return results

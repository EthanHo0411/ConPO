"""Reverse counterfactual construction (ConPO.pdf section 2.3).

Draws a feasible response y_plus, samples a constraint family present on the
prompt via mu(c | x) (or an ablation selection rule), applies that family's
registered corruption operator, and admits the pair (y_plus, y_minus) only
when exactness V(x, y_minus) = {c} holds, the fidelity bound d_Y <= delta
holds, and the quality veto keeps it.

H6 part B (design 2026-09-04): reverse_edge also accepts an explicit
`target_indices` tuple, corrupting every named constraint jointly via
registry.corrupt_set/exact_set instead of drawing (and corrupting) exactly
one family -- a multi-family edge. reverse_batch's own `pair_stats` argument
draws that target set (compositor.sample_family_set) on a share of its
draws (config.MULTI_EDGE_SHARE) when Overrides.multi is on; every other
mechanism (distance gate, veto request/admission, stats.record_admission)
is shared with the single-family path unchanged.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from conpo import compositor, config
from conpo.data import PoolRow
from conpo.distance import token_distance, within_delta
from conpo.edges import Edge, FeasibleBuffer
from conpo.judge import Verdict, VetoRequest
from conpo.operators import registry
from conpo.selection import FamilyStats, sample_family
from conpo.verifiers import Constraint, check, description, verifier_vector


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


def _corrupt_set_without_exactness(
    constraints: tuple[Constraint, ...],
    target_indices: tuple[int, ...],
    y_plus: str,
    rng: random.Random,
    max_candidates: int = 8,
) -> str | None:
    """Exactness-ablation multi-target corruption, generalising
    `_corrupt_without_exactness` to a target set the way registry.corrupt_set
    generalises registry.corrupt: composes each target's first candidate
    (each target's own operator, applied in `target_indices` order to the
    running response) and admits as soon as every target fails, with no
    check that any other constraint still passes. None when any target
    family has no operator, or `max_candidates` attempts never land a
    composition that changes the text and fails every target.
    """
    targets = [constraints[i] for i in target_indices]
    if any(t.iid in registry.NO_OPERATOR for t in targets):
        return None
    for _ in range(max_candidates):
        current = y_plus
        buildable = True
        for target in targets:
            candidate = next(registry.candidates(current, target, rng), None)
            if candidate is None:
                buildable = False
                break
            current = candidate
        if not buildable or current == y_plus:
            continue
        verdicts = verifier_vector(constraints, current)
        if all(not verdicts[i] for i in target_indices):
            return current
    return None


def reverse_edge(
    row: PoolRow,
    y_plus: str,
    stats: FamilyStats,
    judge: Judge,
    rng: random.Random,
    current_update: int,
    target_indices: tuple[int, ...] | None = None,
    selection: str = "transfer",
    exactness: bool = True,
    distance_gate: bool = True,
    veto: bool = True,
) -> ReverseResult:
    """One reverse-construction attempt on a feasible response `y_plus`.

    `target_indices`, when given, names the exact constraint indices to
    corrupt jointly -- a multi-family edge (H6 part B): registry.corrupt_set/
    exact_set replace the single-target corrupt/exact, the resulting edge's
    `families` carries every target family, and a failed composition is
    reported as "no_multi_candidate" (distinct from corrupt's own
    NO_CANDIDATE/TARGET_STILL_PASSES/COLLATERAL/UNCHANGED reasons). Raises
    ValueError if given as an empty tuple. Left at its default (None), this
    is exactly the single-family behaviour reverse_edge always had: one
    family drawn from `eligible_families(row)` under `selection`.
    """
    if not row.english:
        return ReverseResult(edge=None, family=None, reason="not_english", distance=None, veto_kept=None)

    if target_indices is None:
        eligible = eligible_families(row)
        if not eligible:
            return ReverseResult(edge=None, family=None, reason="no_eligible_family", distance=None, veto_kept=None)
        chosen = _choose_family(selection, stats, eligible, rng)
        indices = (next(i for i, c in enumerate(row.constraints) if c.iid == chosen),)
    else:
        if not target_indices:
            raise ValueError("target_indices must be non-empty when given")
        indices = target_indices

    families = tuple(row.constraints[i].iid for i in indices)
    # A comma only ever appears here for a multi-family draw (families[0]
    # alone for len(indices) == 1): trainer.py's mechanism log reads this to
    # tell a multi-family attempt apart from a single-family one.
    family_label = families[0] if len(indices) == 1 else ",".join(sorted(families))

    if len(indices) == 1:
        if exactness:
            corrupted, reason = registry.corrupt(row.constraints, indices[0], y_plus, rng)
        else:
            corrupted, reason = _corrupt_without_exactness(row.constraints[indices[0]], y_plus, rng)
    else:
        if exactness:
            corrupted = registry.corrupt_set(row.constraints, indices, y_plus, rng)
        else:
            corrupted = _corrupt_set_without_exactness(row.constraints, indices, y_plus, rng)
        reason = "admitted" if corrupted is not None else "no_multi_candidate"

    if corrupted is None:
        return ReverseResult(edge=None, family=family_label, reason=reason, distance=None, veto_kept=None)

    d = token_distance(y_plus, corrupted)
    # delta is read fresh from config so a monkeypatched config.DELTA takes
    # effect: within_delta's own default parameter is bound once at import.
    if distance_gate and not within_delta(y_plus, corrupted, delta=config.DELTA):
        return ReverseResult(edge=None, family=family_label, reason="distance", distance=d, veto_kept=None)

    veto_kept: bool | None = None
    if veto:
        verdict = judge.veto(
            VetoRequest(
                prompt=row.prompt,
                y_plus=y_plus,
                y_minus=corrupted,
                families=families,
                descriptions=tuple(description(row.constraints[i]) for i in indices),
            )
        )
        veto_kept = verdict.keep
        if not verdict.keep:
            return ReverseResult(edge=None, family=family_label, reason="veto", distance=d, veto_kept=False)

    edge = Edge(
        prompt_id=row.key,
        prompt=row.prompt,
        y_plus=y_plus,
        y_minus=corrupted,
        families=frozenset(families),
        direction="reverse",
        step=0,
        logp_ref_plus=math.nan,
        logp_ref_minus=math.nan,
        len_plus=0,
        len_minus=0,
        created_update=current_update,
        meta={"reason": reason, "distance": d, "selection": selection},
    )
    for family in families:
        stats.record_admission(family)
    return ReverseResult(edge=edge, family=family_label, reason="admitted", distance=d, veto_kept=veto_kept)


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
    pair_stats: compositor.PairStats | None = None,
) -> list[ReverseResult]:
    """Draw `n` (prompt_id, y_plus) samples from `feasible` and attempt reverse_edge on each.

    `pair_stats` (H6 part B; the trainer only ever passes one when
    `Overrides.multi` is set) makes each draw independently try a
    multi-family edge first, with probability config.MULTI_EDGE_SHARE:
    compositor.sample_family_set over the row's constraints and `pair_stats`
    picks the target pair. An empty draw -- no qualifying pair, or fewer
    than two operator-capable families on the row -- falls back to the
    ordinary single-family draw through `selection`, exactly what
    `pair_stats=None` always does.
    """
    results: list[ReverseResult] = []
    for _ in range(n):
        prompt_id, y_plus = feasible.sample_prompt(rng)
        row = rows_by_key[prompt_id]
        target_indices: tuple[int, ...] | None = None
        if pair_stats is not None and rng.random() < config.MULTI_EDGE_SHARE:
            n_eligible = sum(1 for c in row.constraints if c.iid not in registry.NO_OPERATOR)
            if n_eligible >= 2:
                target_indices = compositor.sample_family_set(row.constraints, rng, pair_stats) or None
        results.append(
            reverse_edge(
                row,
                y_plus,
                stats,
                judge,
                rng,
                current_update,
                target_indices=target_indices,
                selection=selection,
                exactness=exactness,
                distance_gate=distance_gate,
                veto=veto,
            )
        )
    return results

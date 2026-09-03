"""Operator contract and exactness admission (ConPO.pdf section 2.3).

An operator takes a response that satisfies constraint A and yields candidate
corruptions, most minimal first. Operators may be heuristic: the verifier is
the source of truth. `exact` implements V(x, y~) = {A}: the target constraint
fails and every other constraint on the prompt still passes. The distance gate
d_Y <= delta and the quality veto are applied by the caller (reverse.py), not
here, so this module depends only on the verifiers layer.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable, Iterator

from conpo.verifiers import Constraint, verifier_vector

Operator = Callable[[str, Constraint, random.Random], Iterator[str]]
Fixture = Callable[[Constraint, random.Random], str]

OPERATORS: dict[str, Operator] = {}
FIXTURES: dict[str, Fixture] = {}

# Families with no programmatic corruption. Translation needs a model, so
# language:response_language stays operator-free, as in stackelberg-if.
NO_OPERATOR: frozenset[str] = frozenset({"language:response_language"})

EXACT = "exact"
UNCHANGED = "unchanged"
TARGET_STILL_PASSES = "target_still_passes"
COLLATERAL = "collateral"
NO_CANDIDATE = "no_candidate"


def register(iid: str, operator: Operator, fixture: Fixture) -> None:
    """Register one family. Re-registration is a programming error."""
    if iid in OPERATORS:
        raise ValueError(f"operator already registered for {iid}")
    OPERATORS[iid] = operator
    FIXTURES[iid] = fixture


def candidates(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    """Candidate corruptions for `constraint`, empty when the family has no operator."""
    operator = OPERATORS.get(constraint.iid)
    if operator is None:
        return iter(())
    return operator(response, constraint, rng)


def exact(constraints: tuple[Constraint, ...], target_index: int, original: str, corrupted: str) -> tuple[bool, str]:
    """Exactness test: target fails, all other constraints pass.

    Returns (ok, reason); reason is EXACT, UNCHANGED, TARGET_STILL_PASSES or
    "collateral:<iid>" naming the first non-target constraint that broke.
    """
    if corrupted == original:
        return False, UNCHANGED
    verdicts = verifier_vector(constraints, corrupted)
    if verdicts[target_index]:
        return False, TARGET_STILL_PASSES
    for index, (constraint, ok) in enumerate(zip(constraints, verdicts)):
        if index != target_index and not ok:
            return False, f"{COLLATERAL}:{constraint.iid}"
    return True, EXACT


def corrupt(
    constraints: tuple[Constraint, ...],
    target_index: int,
    response: str,
    rng: random.Random,
    max_candidates: int = 8,
) -> tuple[str | None, str]:
    """First candidate that passes `exact`, or (None, last reason).

    `response` must satisfy every constraint; callers draw it from the
    feasible buffer. At most `max_candidates` candidates are verified.
    """
    reason = NO_CANDIDATE
    target = constraints[target_index]
    for count, candidate in enumerate(candidates(response, target, rng)):
        if count >= max_candidates:
            break
        ok, reason = exact(constraints, target_index, response, candidate)
        if ok:
            return candidate, reason
    return None, reason


# ---------------------------------------------------------------------------
# Multi-target variants (H6 part A, design 2026-09-03): exact_set/corrupt_set
# generalise exact/corrupt from one target constraint to a SET of targets,
# for compositional (multi-constraint) counterfactual edges. Single-target
# exact/corrupt above are unchanged and unused by these.
# ---------------------------------------------------------------------------


def exact_set(
    constraints: tuple[Constraint, ...],
    target_indices: tuple[int, ...],
    original: str,
    corrupted: str,
) -> tuple[bool, str]:
    """Exactness test for a target set: every target fails, every other constraint passes.

    Returns (ok, reason); reason is EXACT, UNCHANGED, TARGET_STILL_PASSES (the
    first target index found still passing) or "collateral:<iid>" naming the
    first non-target constraint that broke.

    `target_indices` must be non-empty: raises ValueError otherwise, since an
    empty set makes "every target fails" vacuously true regardless of
    `corrupted`.
    """
    if not target_indices:
        raise ValueError("target_indices must be non-empty")
    if corrupted == original:
        return False, UNCHANGED
    targets = set(target_indices)
    verdicts = verifier_vector(constraints, corrupted)
    for index in target_indices:
        if verdicts[index]:
            return False, TARGET_STILL_PASSES
    for index, (constraint, ok) in enumerate(zip(constraints, verdicts)):
        if index not in targets and not ok:
            return False, f"{COLLATERAL}:{constraint.iid}"
    return True, EXACT


def _rank_shells(width: int) -> Iterator[tuple[int, ...]]:
    """Rank tuples of length `width`, breadth-first by max(rank): every
    all-zero tuple first, then every tuple whose highest rank is 1, then 2,
    and so on. Infinite; callers must cap total attempts (itertools.islice).
    """
    level = 0
    while True:
        for combo in itertools.product(range(level + 1), repeat=width):
            if max(combo) == level:
                yield combo
        level += 1


def corrupt_set(
    constraints: tuple[Constraint, ...],
    target_indices: tuple[int, ...],
    response: str,
    rng: random.Random,
    max_candidates: int = 8,
) -> str | None:
    """First composition of the targets' operators that passes `exact_set`, or None.

    Applies each target's operator to the running response, in
    `target_indices` order (each operator's own most-minimal candidate
    first), composing them into one candidate. Rank combinations across the
    targets are explored breadth-first (`_rank_shells`): the all-minimal
    composition first, then every composition one rank away, and so on, so a
    small composed edit is always tried before a larger one. At most
    `max_candidates` rank combinations are attempted (built or not).

    A target whose family has no operator (`NO_OPERATOR`) makes this return
    None outright, since no candidate for it could ever exist.

    `target_indices` must be non-empty: raises ValueError otherwise, since
    `_rank_shells` cannot rank a zero-width composition (`max()` of an empty
    combo).
    """
    if not target_indices:
        raise ValueError("target_indices must be non-empty")
    targets = [constraints[i] for i in target_indices]
    if any(t.iid in NO_OPERATOR for t in targets):
        return None
    # ponytail: each rank combination re-runs every target's operator from
    # scratch on the running response, even when a shared prefix of ranks was
    # already built for an earlier combination; fine at max_candidates<=~dozens,
    # revisit with a per-prefix cache if corrupt_set ever needs a much larger budget.
    for ranks in itertools.islice(_rank_shells(len(target_indices)), max_candidates):
        current = response
        buildable = True
        for target, rank in zip(targets, ranks):
            candidate = next(itertools.islice(candidates(current, target, rng), rank, rank + 1), None)
            if candidate is None:
                buildable = False
                break
            current = candidate
        if not buildable:
            continue
        ok, _reason = exact_set(constraints, target_indices, response, current)
        if ok:
            return current
    return None

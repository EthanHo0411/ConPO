"""Operator contract and exactness admission (ConPO.pdf section 2.3).

An operator takes a response that satisfies constraint A and yields candidate
corruptions, most minimal first. Operators may be heuristic: the verifier is
the source of truth. `exact` implements V(x, y~) = {A}: the target constraint
fails and every other constraint on the prompt still passes. The distance gate
d_Y <= delta and the quality veto are applied by the caller (reverse.py), not
here, so this module depends only on the verifiers layer.
"""

from __future__ import annotations

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

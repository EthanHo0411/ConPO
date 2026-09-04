"""Pairwise constraint-family co-violation statistics (H6 part A, design 2026-09-03).

PairStats accumulates, per unordered pair of distinct constraint families
active on the same prompt, a 2x2 contingency table over verifier outcomes
across G samples: n11 (both fail), n10 (first family fails, second passes),
n01 (first passes, second fails), n00 (both pass) -- "first"/"second" meaning
the two family ids in sorted order, not constraint-index order, so the table
for a pair is the same regardless of which order its two constraints appear
in on a prompt. phi() is the phi coefficient (equivalent to Matthews
correlation for a 2x2 table): how correlated the two families' failures are.
A high phi means corrupting one family alone tends to also break the other,
which is the signal sample_family_set uses to pick worth-building joint
(multi-constraint) counterfactual edges (ConPO.pdf section 2.3's exact()
extended to a target SET instead of one target, see operators/registry.py's
exact_set/corrupt_set).

Part B (trainer.py, config.py, reverse.py, design 2026-09-04): PHI_MIN,
MIN_COUNT, ETA_PHI and MAX_EDGE_FAMILIES now live in config.py (imported
here under the same names, so every reference below is unchanged);
conpo/reverse.py:reverse_batch draws a multi-family edge with config.py's
MULTI_EDGE_SHARE, and conpo/trainer.py owns the PairStats instance those
draws read from.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path

from conpo.config import ETA_PHI, MAX_EDGE_FAMILIES, MIN_COUNT, PHI_MIN
from conpo.operators.registry import NO_OPERATOR
from conpo.verifiers import Constraint

_CountRow = dict[str, int]


class PairStats:
    """Per-family-pair 2x2 contingency counts, keyed by unordered family id pair."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str], _CountRow] = {}

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def update(self, constraints: tuple[Constraint, ...], vectors: tuple[tuple[bool, ...], ...]) -> None:
        """Fold the G verifier vectors of one prompt into every distinct-family pair on it.

        `vectors[g][i]` is constraint `i`'s pass/fail verdict on sample `g`
        (True = passed, matching conpo.verifiers.verifier_vector). Same-family
        index pairs (two constraints sharing one iid on the same prompt) are
        skipped, since phi is only meaningful between distinct families.
        """
        families = [c.iid for c in constraints]
        for i, j in itertools.combinations(range(len(constraints)), 2):
            fam_i, fam_j = families[i], families[j]
            if fam_i == fam_j:
                continue
            key = self._key(fam_i, fam_j)
            first_is_i = key[0] == fam_i
            row = self._counts.setdefault(key, {"n11": 0, "n10": 0, "n01": 0, "n00": 0})
            for vector in vectors:
                fail_i, fail_j = not vector[i], not vector[j]
                fail_first, fail_second = (fail_i, fail_j) if first_is_i else (fail_j, fail_i)
                if fail_first and fail_second:
                    row["n11"] += 1
                elif fail_first:
                    row["n10"] += 1
                elif fail_second:
                    row["n01"] += 1
                else:
                    row["n00"] += 1

    def phi(self, a: str, b: str) -> float:
        """Phi coefficient for the pair (a, b), 0.0 if unseen or any contingency margin is zero."""
        row = self._counts.get(self._key(a, b))
        if row is None:
            return 0.0
        n11, n10, n01, n00 = row["n11"], row["n10"], row["n01"], row["n00"]
        denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
        if denom == 0:
            return 0.0
        return (n11 * n00 - n10 * n01) / math.sqrt(denom)

    def count(self, a: str, b: str) -> int:
        """Total samples observed for the pair (a, b), 0 if unseen."""
        row = self._counts.get(self._key(a, b))
        return sum(row.values()) if row is not None else 0

    def top_pairs(self, k: int) -> list[tuple[str, str, float, int]]:
        """The `k` pairs with the highest phi, as (a, b, phi, count), for logging."""
        scored = [(a, b, self.phi(a, b), self.count(a, b)) for a, b in self._counts]
        scored.sort(key=lambda row: row[2], reverse=True)
        return scored[:k]

    def save(self, path: str | Path) -> None:
        rows = [{"a": a, "b": b, **counts} for (a, b), counts in self._counts.items()]
        Path(path).write_text(json.dumps(rows), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        """Replace this instance's counts with those saved at `path`."""
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        self._counts = {
            (row["a"], row["b"]): {
                "n11": int(row["n11"]),
                "n10": int(row["n10"]),
                "n01": int(row["n01"]),
                "n00": int(row["n00"]),
            }
            for row in rows
        }


def sample_family_set(
    constraints: tuple[Constraint, ...],
    rng: random.Random,
    stats: PairStats,
    size: int = MAX_EDGE_FAMILIES,
    phi_min: float = PHI_MIN,
    min_count: int = MIN_COUNT,
    eta: float = ETA_PHI,
) -> tuple[int, ...]:
    """One pair of constraint indices sampled proportional to exp(phi / eta).

    Eligible pairs: distinct-family index pairs on `constraints`, neither
    family in NO_OPERATOR (mirroring selection.mu, since a NO_OPERATOR family
    can never yield a corrupt_set candidate), with stats.count >= min_count
    and stats.phi >= phi_min. Empty tuple when none qualify (the caller then
    falls back to a single-family edge).
    """
    if size != 2:
        # ponytail: phi is a pairwise statistic, so only size=2 is scored today;
        # part B can extend to size > 2 (e.g. summed pairwise phi over a triple)
        # if a compositional need for larger sets arises.
        raise NotImplementedError("sample_family_set only supports size=2 in part A")
    families = [c.iid for c in constraints]
    eligible: list[tuple[int, int]] = []
    scores: list[float] = []
    for i, j in itertools.combinations(range(len(constraints)), 2):
        if families[i] == families[j]:
            continue
        if families[i] in NO_OPERATOR or families[j] in NO_OPERATOR:
            continue
        if stats.count(families[i], families[j]) < min_count:
            continue
        p = stats.phi(families[i], families[j])
        if p < phi_min:
            continue
        eligible.append((i, j))
        scores.append(p / eta)
    if not eligible:
        return ()
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    choice = rng.choices(range(len(eligible)), weights=weights, k=1)[0]
    return eligible[choice]

"""Internal evaluation metrics (ifbench-eval-protocol.md section 4.1): pure
functions over parallel lists of (PoolRow, response) pairs.

PAcc = mean over prompts of the product of per-constraint verifier verdicts
(equivalently: fraction of prompts whose response is fully feasible).
IAcc = micro-averaged verdicts over every constraint instance across every
row. Macro / worst-decile family accuracy and the per-constraint-count
stratification derive from the same per-constraint verdicts. Reward-hacking
checks (mean length, repetition rate) are computed on the response text
alone. `task_relevance` is a documented hook: the paper scores it with the
quality-veto judge (conpo/judge.py), which is not called here.
"""

from collections.abc import Sequence
from typing import Any, Protocol

from conpo.data import PoolRow
from conpo.verifiers import feasible, verifier_vector


class _Tokenizer(Protocol):
    def __call__(self, text: str, add_special_tokens: bool) -> Any: ...


def _check_lengths(rows: Sequence[PoolRow], responses: Sequence[str]) -> None:
    if len(rows) != len(responses):
        raise ValueError(f"rows/responses length mismatch: {len(rows)} vs {len(responses)}")


def _flat_verdicts(rows: Sequence[PoolRow], responses: Sequence[str]) -> list[tuple[str, bool]]:
    """(family_id, passed) for every constraint instance across all rows."""
    _check_lengths(rows, responses)
    out: list[tuple[str, bool]] = []
    for row, response in zip(rows, responses):
        for constraint, ok in zip(row.constraints, verifier_vector(row.constraints, response)):
            out.append((constraint.iid, ok))
    return out


def pacc(rows: Sequence[PoolRow], responses: Sequence[str]) -> float:
    """Mean over prompts of the product of that prompt's verifier verdicts."""
    _check_lengths(rows, responses)
    if not rows:
        return 0.0
    return sum(1.0 for row, response in zip(rows, responses) if feasible(row.constraints, response)) / len(rows)


def iacc(rows: Sequence[PoolRow], responses: Sequence[str]) -> float:
    """Micro-averaged verdicts over every constraint instance."""
    flat = _flat_verdicts(rows, responses)
    if not flat:
        return 0.0
    return sum(1 for _, ok in flat if ok) / len(flat)


def family_accuracies(rows: Sequence[PoolRow], responses: Sequence[str]) -> dict[str, float]:
    """Per-family accuracy: fraction of that family's constraint instances that passed."""
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for family, ok in _flat_verdicts(rows, responses):
        totals[family] = totals.get(family, 0) + 1
        correct[family] = correct.get(family, 0) + int(ok)
    return {family: correct[family] / total for family, total in totals.items()}


def macro_family_accuracy(rows: Sequence[PoolRow], responses: Sequence[str]) -> float:
    """Unweighted mean of family_accuracies (each family counts once)."""
    accuracies = family_accuracies(rows, responses)
    if not accuracies:
        return 0.0
    return sum(accuracies.values()) / len(accuracies)


def worst_decile_family_accuracy(rows: Sequence[PoolRow], responses: Sequence[str]) -> float:
    """Mean accuracy of the bottom 10 percent of families, at least one family."""
    accuracies = sorted(family_accuracies(rows, responses).values())
    if not accuracies:
        return 0.0
    n = max(1, round(len(accuracies) * 0.1))
    return sum(accuracies[:n]) / n


def stratified_by_count(rows: Sequence[PoolRow], responses: Sequence[str]) -> dict[int, float]:
    """PAcc grouped by row.num_constraints."""
    _check_lengths(rows, responses)
    groups: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault(row.num_constraints, []).append(index)
    return {
        count: pacc([rows[i] for i in indices], [responses[i] for i in indices]) for count, indices in groups.items()
    }


def repetition_rate(text: str, n: int = 4) -> float:
    """Fraction of `text`'s word n-grams that are repeats of an earlier one."""
    words = text.split()
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams)


def mean_length(texts: Sequence[str], tokenizer: _Tokenizer | None = None) -> float:
    """Mean response length: token count via `tokenizer` if given, else whitespace-split words."""
    if not texts:
        return 0.0
    if tokenizer is not None:
        lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
    else:
        lengths = [len(text.split()) for text in texts]
    return sum(lengths) / len(lengths)


def summarise(
    rows: Sequence[PoolRow], responses: Sequence[str], split_name: str, tokenizer: _Tokenizer | None = None
) -> dict[str, Any]:
    """Every metric above, keyed for one eval split."""
    # ponytail: recomputes verifier_vector per row several times over (once
    # per metric that touches verdicts) instead of threading one shared
    # _flat_verdicts result through every helper below. Each helper stays a
    # pure, independently-testable function over (rows, responses); add a
    # shared-verdicts cache if profiling ever shows this redundant
    # re-verification (regex/langdetect per constraint) costs real time.
    return {
        "split": split_name,
        "num_prompts": len(rows),
        "pacc": pacc(rows, responses),
        "iacc": iacc(rows, responses),
        "family_accuracies": family_accuracies(rows, responses),
        "macro_family_accuracy": macro_family_accuracy(rows, responses),
        "worst_decile_family_accuracy": worst_decile_family_accuracy(rows, responses),
        "stratified_by_count": stratified_by_count(rows, responses),
        "mean_length": mean_length(responses, tokenizer),
        "repetition_rate": (sum(repetition_rate(r) for r in responses) / len(responses)) if responses else 0.0,
        # Hook only: paper section 4.1's task-relevance check is scored by the
        # quality-veto judge (conpo/judge.py's HttpJudge), a separate,
        # network-calling pass over (prompt, response) not run here.
        "task_relevance": None,
    }

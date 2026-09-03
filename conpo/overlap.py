"""Contamination overlap check: training-subset base prompts vs. benchmark
prompt texts.

13-gram overlap over whitespace tokens is the standard decontamination
convention (GPT-3 Appendix C, PaLM section 3, and the Dolma/OLMo pipelines
all use 13-token spans): long enough that a shared span is not coincidence,
short enough that a genuinely reused passage still produces one.
"""

import re
import string
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_WS_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation."""
    lowered = text.lower().translate(_PUNCT_TABLE)
    return _WS_RE.sub(" ", lowered).strip()


def ngrams(text: str, n: int = 13) -> set[tuple[str, ...]]:
    """Whitespace-token n-grams of `text`'s normalised form.

    Normalises internally so two texts differing only in case, punctuation,
    or whitespace still share the same n-grams. Fewer than `n` tokens
    yields an empty set (no near-miss inflation from short texts).
    """
    tokens = [t for t in normalise(text).split(" ") if t]
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass(frozen=True)
class OverlapReport:
    exact_matches: int
    ngram_hits: int
    hit_keys: list[str]
    total_subset: int
    total_benchmark: int


def check_overlap(subset_texts: Mapping[str, str], benchmark_texts: Sequence[str], n: int = 13) -> OverlapReport:
    """Exact and n-gram overlap between a keyed subset and a benchmark.

    exact_matches counts subset entries whose normalised text equals a
    benchmark text's normalised text; ngram_hits counts subset entries
    sharing at least one n-gram with the benchmark's n-gram set (a superset
    of the exact matches long enough to have n tokens). hit_keys lists every
    subset key hit by either check, in `subset_texts` iteration order.
    """
    benchmark_norm = {normalise(t) for t in benchmark_texts}
    benchmark_grams: set[tuple[str, ...]] = set()
    for t in benchmark_texts:
        benchmark_grams |= ngrams(t, n)

    exact_matches = 0
    ngram_hits = 0
    hit_keys: list[str] = []
    for key, text in subset_texts.items():
        is_exact = normalise(text) in benchmark_norm
        is_ngram_hit = bool(ngrams(text, n) & benchmark_grams)
        exact_matches += int(is_exact)
        ngram_hits += int(is_ngram_hit)
        if is_exact or is_ngram_hit:
            hit_keys.append(key)

    return OverlapReport(
        exact_matches=exact_matches,
        ngram_hits=ngram_hits,
        hit_keys=hit_keys,
        total_subset=len(subset_texts),
        total_benchmark=len(benchmark_texts),
    )

"""Pool loading, filtering and deterministic splitting for allenai/IF_multi_constraints_upto5.

Row schema (verified against the cached parquet): `key` (str), `messages`
(list of one {"role": "user", "content": ...}), `ground_truth` (a Python-repr
string of `[{"instruction_id": [...], "kwargs": [None | dict, ...]}]`), and
`constraint` (the constraint sentences, tab-separated, appended verbatim at
the end of the message content, in order).
"""

import ast
import hashlib
import itertools
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

import datasets
from langdetect import DetectorFactory, LangDetectException, detect

from conpo import config
from conpo.verifiers import FAMILIES, HELDOUT_TEST_FAMILIES, IFEVAL_FAMILIES, Constraint

DetectorFactory.seed = 0

_FAMILIES_SET = set(FAMILIES)
_HELDOUT_SET = set(HELDOUT_TEST_FAMILIES)
_IFEVAL25_SET = set(IFEVAL_FAMILIES)


def parse_ground_truth(ground_truth: str) -> tuple[Constraint, ...]:
    """Parse a row's `ground_truth` string into Constraints, in dataset order."""
    spec = ast.literal_eval(ground_truth)[0]
    ids: list[str] = spec["instruction_id"]
    kwargs_list: list[dict | None] = spec["kwargs"]
    return tuple(Constraint.from_ground_truth(iid, kw) for iid, kw in zip(ids, kwargs_list))


_WS_RE = re.compile(r"\s+")


def _tail_pattern(joined: str) -> re.Pattern[str]:
    """Regex matching `joined` at the end of a string, tolerant of whitespace
    differences (newline vs space, doubled spaces) between and after tokens."""
    tokens = [t for t in _WS_RE.split(joined) if t]
    body = r"\s+".join(re.escape(t) for t in tokens)
    return re.compile(body + r"\s*\Z", re.DOTALL)


def split_constraints_text(row: Mapping[str, object]) -> tuple[str, tuple[str, ...]] | None:
    """Base task text and constraint sentences.

    Verifies the `constraint` column's tab-separated sentences appear
    verbatim, in order, at the end of the message content; whitespace
    differences are the only tolerated slack. Returns None when they cannot
    be located there.
    """
    messages = row["messages"]
    content: str = messages[0]["content"]
    constraint_col: str = row["constraint"]
    sentences = tuple(s.strip() for s in constraint_col.split("\t") if s.strip())
    if not sentences:
        return None
    joined = " ".join(sentences)
    stripped = content.rstrip()
    if stripped.endswith(joined):
        base = stripped[: len(stripped) - len(joined)].rstrip()
        return base, sentences
    match = _tail_pattern(joined).search(stripped)
    if match is not None:
        return stripped[: match.start()].rstrip(), sentences
    return None


def is_english(text: str) -> bool:
    """Whether `text` detects as English, deterministic (DetectorFactory.seed = 0)."""
    if not text.strip():
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


@dataclass(frozen=True)
class PoolRow:
    key: str
    base_prompt: str
    prompt: str
    sentences: tuple[str, ...]
    constraints: tuple[Constraint, ...]
    english: bool
    num_constraints: int


# 76 corpus rows carry nth_paragraph_first_word kwargs with nth_paragraph >
# num_paragraphs, which forces the vendored build_description to redraw nth
# from the global `random` module (non-deterministic, unsatisfiable as
# authored). Ported from stackelberg-if/probe_ifrepair.py:bad_nth_kwargs.
_NTH_IID = "length_constraints:nth_paragraph_first_word"


def _bad_nth_kwargs(constraints: tuple[Constraint, ...]) -> bool:
    for c in constraints:
        if c.iid != _NTH_IID:
            continue
        num = c.kwargs.get("num_paragraphs")
        nth = c.kwargs.get("nth_paragraph")
        if num is not None and nth is not None and nth > num:
            return True
    return False


@dataclass(frozen=True)
class FilterReport:
    total: int
    kept: int
    dropped_unmapped: int
    dropped_bad_kwargs: int
    dropped_unparseable_constraint_text: int
    would_drop_under_ifeval25_filter: int
    english_share: float
    constraint_count_histogram: dict[int, int]
    per_family_counts: dict[str, int]


def load_pool(limit: int | None = None) -> tuple[list[PoolRow], FilterReport]:
    """Load and filter the pool dataset.

    Keeps rows whose constraint ids are all in FAMILIES (54) and none in
    HELDOUT_TEST_FAMILIES, then drops bad_nth_kwargs rows and rows whose
    constraint text cannot be split verbatim from the message content.
    `limit` caps the number of raw rows considered (a prefix slice), for
    fast tests; None loads the full pool.
    """
    ds = datasets.load_dataset(config.POOL_DATASET, split="train")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    total = len(ds)
    dropped_unmapped = 0
    dropped_bad_kwargs = 0
    dropped_unparseable_constraint_text = 0
    would_drop_25 = 0
    english_count = 0
    histogram: Counter = Counter()
    family_counts: Counter = Counter()
    rows: list[PoolRow] = []

    for raw in ds:
        constraints = parse_ground_truth(raw["ground_truth"])
        ids = tuple(c.iid for c in constraints)
        if any(iid not in _FAMILIES_SET for iid in ids) or any(iid in _HELDOUT_SET for iid in ids):
            dropped_unmapped += 1
            continue
        if _bad_nth_kwargs(constraints):
            dropped_bad_kwargs += 1
            continue
        split = split_constraints_text(raw)
        if split is None:
            dropped_unparseable_constraint_text += 1
            continue
        base_prompt, sentences = split
        prompt: str = raw["messages"][0]["content"]
        english = is_english(base_prompt)

        rows.append(
            PoolRow(
                key=raw["key"],
                base_prompt=base_prompt,
                prompt=prompt,
                sentences=sentences,
                constraints=constraints,
                english=english,
                num_constraints=len(constraints),
            )
        )

        if any(iid not in _IFEVAL25_SET for iid in ids):
            would_drop_25 += 1
        if english:
            english_count += 1
        histogram[len(constraints)] += 1
        family_counts.update(ids)

    kept = len(rows)
    report = FilterReport(
        total=total,
        kept=kept,
        dropped_unmapped=dropped_unmapped,
        dropped_bad_kwargs=dropped_bad_kwargs,
        dropped_unparseable_constraint_text=dropped_unparseable_constraint_text,
        would_drop_under_ifeval25_filter=would_drop_25,
        english_share=(english_count / kept if kept else 0.0),
        constraint_count_histogram=dict(histogram),
        per_family_counts=dict(family_counts),
    )
    return rows, report


# ---------------------------------------------------------------------------
# Deterministic splitting
# ---------------------------------------------------------------------------

_DEV_SIZE = 1000
_EVAL_I_SIZE = 1500
_EVAL_II_PAIR_COUNT = 20
_EVAL_II_MIN_COOCCURRENCE = 60


def _stable_unit(seed: int, salt: str, key: str) -> float:
    """Deterministic pseudo-random float in [0, 1), from sha1(seed:salt:key)."""
    digest = hashlib.sha1(f"{seed}:{salt}:{key}".encode()).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def make_splits(rows: list[PoolRow], seed: int) -> dict[str, list[PoolRow]]:
    """Deterministic, disjoint-by-key split into train/dev/eval_i/eval_ii/eval_iii.

    Every row lands in exactly one split, in this allocation order:
      1. eval_iii: every row with 5 constraints (so train keeps at most 4).
      2. eval_ii: every remaining row containing any of up to 20 seeded-drawn
         family pairs that co-occur in at least 60 rows of `rows`.
      3. eval_i: up to 1,500 rows drawn (sha1-seeded) from what is left --
         prompts unseen at train time, over constraint families that are
         otherwise familiar (eval_ii already removed the held-out pairs).
      4. dev: up to 1,000 rows drawn the same way from what remains.
      5. train: everything left over.
    Sizes below 1,000 / 1,500 / 20 pairs happen only when `rows` itself is
    too small to support them (e.g. a small test sample); they are capped,
    not padded.
    """
    eval_iii = [r for r in rows if r.num_constraints == 5]
    eval_iii_keys = {r.key for r in eval_iii}
    remaining = [r for r in rows if r.key not in eval_iii_keys]

    pair_counts: Counter = Counter()
    for r in rows:
        ids = sorted({c.iid for c in r.constraints})
        for a, b in itertools.combinations(ids, 2):
            pair_counts[frozenset((a, b))] += 1
    eligible_pairs = [p for p, n in pair_counts.items() if n >= _EVAL_II_MIN_COOCCURRENCE]
    eligible_pairs.sort(key=lambda p: _stable_unit(seed, "eval_ii_pairs", "|".join(sorted(p))))
    chosen_pairs = set(eligible_pairs[:_EVAL_II_PAIR_COUNT])

    def _has_chosen_pair(r: PoolRow) -> bool:
        ids = sorted({c.iid for c in r.constraints})
        return any(frozenset((a, b)) in chosen_pairs for a, b in itertools.combinations(ids, 2))

    eval_ii = [r for r in remaining if _has_chosen_pair(r)]
    eval_ii_keys = {r.key for r in eval_ii}
    remaining = [r for r in remaining if r.key not in eval_ii_keys]

    remaining.sort(key=lambda r: _stable_unit(seed, "eval_i", r.key))
    n_eval_i = min(_EVAL_I_SIZE, len(remaining))
    eval_i, remaining = remaining[:n_eval_i], remaining[n_eval_i:]

    remaining.sort(key=lambda r: _stable_unit(seed, "dev", r.key))
    n_dev = min(_DEV_SIZE, len(remaining))
    dev, train = remaining[:n_dev], remaining[n_dev:]

    return {"train": train, "dev": dev, "eval_i": eval_i, "eval_ii": eval_ii, "eval_iii": eval_iii}


@dataclass
class TokenMeter:
    """Generated-token budget tracker. budget <= 0 means unlimited."""

    budget: int = field(default_factory=lambda: int(config.env("RUN_TOKEN_BUDGET", "0")))
    total: int = 0

    def add(self, n: int) -> None:
        self.total += n

    def exhausted(self) -> bool:
        return self.budget > 0 and self.total >= self.budget

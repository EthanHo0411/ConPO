import datasets
import pytest

from conpo import config
from conpo.data import (
    FilterReport,
    PoolRow,
    TokenMeter,
    load_pool,
    make_splits,
    parse_ground_truth,
    split_constraints_text,
)
from conpo.verifiers import Constraint

# Slow (~10 min: full pool load_pool(), langdetect per row): skip with `-k "not full_sweep"`.
FULL_SWEEP_TEST_NAME = "test_full_sweep_verbatim_rate_and_filter_report"

_SAMPLE_LIMIT = 3000


@pytest.fixture(scope="module")
def pool_sample() -> tuple[list[PoolRow], FilterReport]:
    return load_pool(limit=_SAMPLE_LIMIT)


# (a) parse_ground_truth on row 0 and row 1 -----------------------------------


def test_parse_ground_truth_row0_row1() -> None:
    ds = datasets.load_dataset(config.POOL_DATASET, split="train")

    row0 = parse_ground_truth(ds[0]["ground_truth"])
    assert [c.iid for c in row0] == ["detectable_format:sentence_hyphens", "last_word:last_word_answer"]
    assert row0[0].kwargs == {}
    assert row0[1].kwargs == {"last_word": "brief"}

    row1 = parse_ground_truth(ds[1]["ground_truth"])
    assert [c.iid for c in row1] == [
        "copy:copying_multiple",
        "length_constraints:number_paragraphs",
        "keywords:exclude_word_harder",
    ]
    assert row1[0].kwargs["N"] == 3
    assert "prompt_to_repeat" in row1[0].kwargs
    assert row1[1].kwargs == {"num_paragraphs": 6}
    assert row1[2].kwargs == {"keyword": "2."}


# (b) split_constraints_text verbatim-match rate on the first 3,000 rows -----


def test_split_constraints_text_verbatim_rate_on_sample() -> None:
    ds = datasets.load_dataset(config.POOL_DATASET, split="train")
    ds = ds.select(range(min(_SAMPLE_LIMIT, len(ds))))

    hits = 0
    for raw in ds:
        result = split_constraints_text(raw)
        if result is None:
            continue
        base, sentences = result
        assert sentences == tuple(s.strip() for s in raw["constraint"].split("\t") if s.strip())
        assert base
        hits += 1

    rate = hits / len(ds)
    print(f"\nverbatim-append match rate on first {len(ds)} rows: {rate:.4f}")
    assert rate > 0.95


# (c) would_drop_under_ifeval25_filter and dropped_unmapped on the sample ----


def test_filter_report_ifeval25_and_unmapped(pool_sample: tuple[list[PoolRow], FilterReport]) -> None:
    _, report = pool_sample
    assert report.would_drop_under_ifeval25_filter > 0
    if report.dropped_unmapped != 0:
        ds = datasets.load_dataset(config.POOL_DATASET, split="train")
        ds = ds.select(range(min(_SAMPLE_LIMIT, len(ds))))
        outside: set[str] = set()
        from conpo.verifiers import FAMILIES

        families = set(FAMILIES)
        for raw in ds:
            for c in parse_ground_truth(raw["ground_truth"]):
                if c.iid not in families:
                    outside.add(c.iid)
        pytest.fail(f"dropped_unmapped == {report.dropped_unmapped}, ids outside FAMILIES: {sorted(outside)}")
    assert report.dropped_unmapped == 0


# (d) splits disjoint; eval_ii carries no train pair leakage -----------------


def test_splits_disjoint_on_sample(pool_sample: tuple[list[PoolRow], FilterReport]) -> None:
    rows, _ = pool_sample
    splits = make_splits(rows, seed=20260805)

    assert sum(len(s) for s in splits.values()) == len(rows)

    seen: set[str] = set()
    for name, split_rows in splits.items():
        keys = {r.key for r in split_rows}
        assert len(keys) == len(split_rows), f"duplicate keys within {name}"
        assert keys.isdisjoint(seen), f"{name} overlaps an earlier split"
        seen |= keys

    for r in splits["train"]:
        assert r.num_constraints <= 4


def test_make_splits_eval_ii_pairs_never_leak_into_train() -> None:
    """Synthetic fixture: a pair that co-occurs >=60 times must be captured
    whole into eval_ii, and none of it may leak into train/dev/eval_i."""
    a, b = Constraint("punctuation:no_comma", {}), Constraint("change_case:english_lowercase", {})
    filler = Constraint("startend:quotation", {})

    def row(key: str, constraints: tuple[Constraint, ...]) -> PoolRow:
        return PoolRow(
            key=key,
            base_prompt="filler task text",
            prompt="filler task text " + " ".join(f"c{i}" for i in range(len(constraints))),
            sentences=tuple(f"c{i}" for i in range(len(constraints))),
            constraints=constraints,
            english=True,
            num_constraints=len(constraints),
        )

    pair_rows = [row(f"pair_{i}", (a, b)) for i in range(65)]
    filler_rows = [row(f"filler_{i}", (filler,)) for i in range(3000)]
    rows = pair_rows + filler_rows

    splits = make_splits(rows, seed=20260805)

    pair_keys = {r.key for r in pair_rows}
    eval_ii_keys = {r.key for r in splits["eval_ii"]}
    assert eval_ii_keys == pair_keys

    assert splits["train"], "synthetic filler pool must leave a non-empty train split to test leakage on"
    for r in splits["train"]:
        assert r.key not in pair_keys
        ids = {c.iid for c in r.constraints}
        assert not {a.iid, b.iid}.issubset(ids)


# (e) TokenMeter budget logic -- counterfactual_prompts / prompt_with_only tests
# live in tests/test_prompts.py alongside conpo/prompts.py.


def test_token_meter_unlimited_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_TOKEN_BUDGET", raising=False)
    meter = TokenMeter()
    assert meter.budget == 0
    meter.add(10**9)
    assert meter.exhausted() is False


def test_token_meter_respects_env_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_TOKEN_BUDGET", "100")
    meter = TokenMeter()
    assert meter.budget == 100
    meter.add(50)
    assert meter.total == 50
    assert meter.exhausted() is False
    meter.add(50)
    assert meter.total == 100
    assert meter.exhausted() is True


def test_token_meter_explicit_budget_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUN_TOKEN_BUDGET", "999")
    meter = TokenMeter(budget=5)
    meter.add(5)
    assert meter.exhausted() is True


# Full-pool sweep (slow) -------------------------------------------------------


def test_full_sweep_verbatim_rate_and_filter_report() -> None:
    """Verbatim-append rate and FilterReport over the ENTIRE pool (~95k rows).

    Slow (~10 min, dominated by langdetect once per kept row inside
    load_pool()). Run the fast subset only with:
        pytest tests/test_data.py -k "not full_sweep"
    """
    ds = datasets.load_dataset(config.POOL_DATASET, split="train")

    hits = sum(1 for raw in ds if split_constraints_text(raw) is not None)
    rate = hits / len(ds)
    print(f"\nverbatim-append match rate over all {len(ds)} rows: {rate:.4f}")
    assert rate > 0.95

    rows, report = load_pool(limit=None)
    print(f"FilterReport (full pool): {report}")
    assert report.total == len(ds)
    assert report.kept == len(rows)
    assert report.dropped_unmapped == 0
    assert report.would_drop_under_ifeval25_filter > 0

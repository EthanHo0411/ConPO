"""Shared locality property: every registered corruption operator must change
only the characters its own family needs and must leave every other byte of
the response untouched, in particular irregular whitespace elsewhere in the
text (double spaces, tabs, blank-line paragraph dividers).

For each registered family this test builds a passing fixture, injects two
position-independent whitespace markers (a double space after the third
word, a blank-line paragraph divider before the last sentence-initial word),
then runs corrupt() on the resulting variant and checks that:

- corrupt() still finds an exact single-constraint corruption,
- both injected markers survive verbatim in the corrupted text (a family
  whose own target IS whitespace/paragraph structure may destroy at most
  one of the two, since its own edit may legitimately land on one of them),
- the diff between the variant and the corrupted text spans at most 4
  non-equal opcode blocks (a global re-join of the whole response, such as
  " ".join(response.split()), spans far more than 4).

This test does not modify any operator module. It is expected to fail for
several families today; that is the point.
"""

from __future__ import annotations

import difflib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

import conpo.operators  # noqa: F401  (registers all 53 families)
from conpo.operators import registry
from conpo.verifiers import Constraint, check

KWARGS: dict[str, dict[str, object]] = json.loads((Path(__file__).parent / "fixtures" / "family_kwargs.json").read_text())

REGISTERED_FAMILIES: tuple[str, ...] = tuple(sorted(registry.OPERATORS))

# Families whose own target IS a whitespace or paragraph structure (contract,
# task prompt): their operator may legitimately alter one of the two injected
# markers when it falls inside the target span.
WHITESPACE_TARGET_FAMILIES: frozenset[str] = frozenset(
    {
        "paragraphs:paragraphs",
        "paragraphs:paragraphs2",
        "length_constraints:number_paragraphs",
        "length_constraints:nth_paragraph_first_word",
        "detectable_format:number_bullet_lists",
        "detectable_format:multiple_sections",
        "combination:two_responses",
        "copy:copying_multiple",
        "count:counting_composition",
    }
)

LOCALITY_BLOCK_LIMIT = 4

# ---------------------------------------------------------------------------
# variant construction
# ---------------------------------------------------------------------------


def _double_space_edit(text: str) -> tuple[int, str] | None:
    """Insertion point right after the 3rd whitespace-separated word, and a
    marker substring ("word3  word4") that proves the double space survived,
    independent of where it ends up in an edited text. None when `text` has
    fewer than 4 words."""
    words = list(re.finditer(r"\S+", text))
    if len(words) < 4:
        return None
    third, fourth = words[2], words[3]
    marker = text[third.start() : third.end()] + "  " + text[fourth.start() : fourth.end()]
    return third.end(), marker


def _divider_edit(text: str) -> tuple[int, int, str] | None:
    """Span of the single space to replace with a blank-line divider (the
    space right after the LAST ". " that precedes an uppercase word), and a
    marker substring (".\\n\\nWord") that proves the divider survived. None
    when no such sentence boundary exists."""
    matches = list(re.finditer(r"\. ([A-Z]\w*)", text))
    if not matches:
        return None
    m = matches[-1]
    space_start = m.start() + 1
    marker = ".\n\n" + m.group(1)
    return space_start, space_start + 1, marker


def make_variant(text: str) -> tuple[str | None, str | None, str | None]:
    """Irregular-whitespace variant of `text`, plus the two marker
    substrings to check for survival. variant is None when `text` has too
    few words for the double-space injection; the divider marker is None
    when `text` has no later sentence to inject it before."""
    ds = _double_space_edit(text)
    if ds is None:
        return None, None, None
    ds_pos, ds_marker = ds
    edits: list[tuple[int, int, str]] = [(ds_pos, ds_pos, " ")]

    div_marker: str | None = None
    div = _divider_edit(text)
    if div is not None:
        d_start, d_end, div_marker = div
        edits.append((d_start, d_end, "\n\n"))

    variant = text
    for start, end, repl in sorted(edits, key=lambda e: e[0], reverse=True):
        variant = variant[:start] + repl + variant[end:]
    return variant, ds_marker, div_marker


# ---------------------------------------------------------------------------
# per-(family, seed) case
# ---------------------------------------------------------------------------


@dataclass
class Case:
    fixture_ok: bool
    skip_reason: str | None
    variant: str | None
    corrupted: str | None
    reason: str
    markers_total: int
    markers_missing: int
    blocks: int | None


def _build_case(iid: str, seed: int) -> Case:
    c = Constraint.from_ground_truth(iid, KWARGS[iid])
    rng = random.Random(seed)
    text = registry.FIXTURES[iid](c, rng)
    if check(c, text) is not True:
        return Case(False, "fixture does not satisfy its own constraint", None, None, "", 0, 0, None)

    variant, ds_marker, div_marker = make_variant(text)
    if variant is None:
        return Case(True, "fixture has fewer than 4 words, cannot inject a double space", None, None, "", 0, 0, None)

    if not check(c, variant):
        return Case(True, "irregular-whitespace variant no longer satisfies the constraint", variant, None, "", 0, 0, None)

    markers = [m for m in (ds_marker, div_marker) if m is not None]
    corrupted, reason = registry.corrupt((c,), 0, variant, rng)
    if corrupted is None:
        return Case(True, None, variant, None, reason, len(markers), len(markers), None)

    missing = sum(1 for m in markers if m not in corrupted)
    blocks = sum(1 for tag, *_ in difflib.SequenceMatcher(None, variant, corrupted).get_opcodes() if tag != "equal")
    return Case(True, None, variant, corrupted, reason, len(markers), missing, blocks)


FAMILY_SEED_CASES: list[tuple[str, int]] = [(iid, seed) for iid in REGISTERED_FAMILIES for seed in (0, 1)]


@pytest.mark.parametrize("iid,seed", FAMILY_SEED_CASES)
def test_operator_is_locality_preserving(iid: str, seed: int) -> None:
    case = _build_case(iid, seed)
    assert case.fixture_ok, f"{iid} seed {seed}: fixture does not satisfy its own constraint"
    if case.skip_reason is not None:
        pytest.skip(f"{iid} seed {seed}: {case.skip_reason}")

    assert case.corrupted is not None, f"{iid} seed {seed}: corrupt() produced no candidate ({case.reason})"

    if iid in WHITESPACE_TARGET_FAMILIES:
        assert case.markers_missing <= 1, (
            f"{iid} seed {seed}: both injected locality markers destroyed, "
            f"expected at most one for a whitespace-target family"
        )
    else:
        assert case.markers_missing == 0, (
            f"{iid} seed {seed}: injected locality marker(s) destroyed "
            f"({case.markers_total - case.markers_missing}/{case.markers_total} preserved)"
        )

    assert case.blocks is not None and case.blocks <= LOCALITY_BLOCK_LIMIT, (
        f"{iid} seed {seed}: {case.blocks} non-equal diff opcode blocks, "
        f"exceeds the locality bound of {LOCALITY_BLOCK_LIMIT}"
    )


# ---------------------------------------------------------------------------
# summary report: no assertions, the parametrised tests above carry those
# ---------------------------------------------------------------------------


def test_locality_summary_table() -> None:
    print()
    header = f"{'family':<52} {'skipped':<9} {'passed':<7} {'markers':<9} {'blocks':<6}"
    print(header)
    print("-" * len(header))

    n_skipped = 0
    n_passed = 0
    for iid in REGISTERED_FAMILIES:
        case = _build_case(iid, 0)
        if case.skip_reason is not None:
            n_skipped += 1
            print(f"{iid:<52} {'yes':<9} {'-':<7} {'-':<9} {'-':<6}")
            continue

        marker_ok = case.markers_missing <= 1 if iid in WHITESPACE_TARGET_FAMILIES else case.markers_missing == 0
        passed = case.corrupted is not None and marker_ok and case.blocks is not None and case.blocks <= LOCALITY_BLOCK_LIMIT
        n_passed += int(passed)
        markers = f"{case.markers_total - case.markers_missing}/{case.markers_total}"
        blocks = str(case.blocks) if case.blocks is not None else "-"
        print(f"{iid:<52} {'no':<9} {str(passed):<7} {markers:<9} {blocks:<6}")

    n_total = len(REGISTERED_FAMILIES)
    print(f"\n{n_total} families, {n_skipped} skipped, {n_passed} passed, {n_total - n_skipped - n_passed} failed")

"""Exactness admission on a hand-built two-constraint prompt with a stub operator."""

import random
from collections.abc import Iterator

from conpo.operators import registry
from conpo.verifiers import Constraint

NO_COMMA = Constraint("punctuation:no_comma", {})
LOWERCASE = Constraint("change_case:english_lowercase", {})
PROMPT = (NO_COMMA, LOWERCASE)
RESPONSE = "the platform rotates at a constant angular velocity of half a radian per second"


def _insert_comma(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    yield response  # unchanged, must be rejected
    yield response.replace(" at ", ", at ", 1)


def _insert_comma_and_capital(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    yield response.replace(" at ", ", At ", 1)  # breaks both, must be rejected as collateral


def test_exact_accepts_only_the_target_failure() -> None:
    ok, reason = registry.exact(PROMPT, 0, RESPONSE, RESPONSE.replace(" at ", ", at ", 1))
    assert ok and reason == registry.EXACT
    ok, reason = registry.exact(PROMPT, 0, RESPONSE, RESPONSE)
    assert not ok and reason == registry.UNCHANGED
    ok, reason = registry.exact(PROMPT, 0, RESPONSE, RESPONSE + " indeed")
    assert not ok and reason == registry.TARGET_STILL_PASSES
    ok, reason = registry.exact(PROMPT, 0, RESPONSE, RESPONSE.replace(" at ", ", At ", 1))
    assert not ok and reason == f"{registry.COLLATERAL}:change_case:english_lowercase"


def test_corrupt_skips_rejected_candidates(monkeypatch) -> None:
    monkeypatch.setitem(registry.OPERATORS, NO_COMMA.iid, _insert_comma)
    corrupted, reason = registry.corrupt(PROMPT, 0, RESPONSE, random.Random(0))
    assert corrupted == RESPONSE.replace(" at ", ", at ", 1) and reason == registry.EXACT
    monkeypatch.setitem(registry.OPERATORS, NO_COMMA.iid, _insert_comma_and_capital)
    corrupted, reason = registry.corrupt(PROMPT, 0, RESPONSE, random.Random(0))
    assert corrupted is None and reason.startswith(registry.COLLATERAL)


def test_family_without_operator_yields_nothing() -> None:
    lang = Constraint("language:response_language", {"language": "en"})
    assert list(registry.candidates(RESPONSE, lang, random.Random(0))) == []
    assert lang.iid in registry.NO_OPERATOR

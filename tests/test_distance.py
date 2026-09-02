"""Tests for conpo.distance."""

from conpo import config
from conpo.distance import token_distance, tokens, within_delta


def test_identical_texts_have_zero_distance() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    assert token_distance(text, text) == 0.0


def test_one_substitution_in_ten_tokens_gives_point_one() -> None:
    original = "one two three four five six seven eight nine ten"
    substituted = "one two three four five six seven eight nine ZEBRA"
    assert token_distance(original, substituted) == 0.1


def test_insertion_is_counted() -> None:
    original = "a b c d e"
    inserted = "a b c X d e"
    assert token_distance(original, inserted) == 1 / 5


def test_within_delta_at_the_boundary() -> None:
    original = "one two three four five six seven eight nine ten"
    substituted = "one two three four five six seven eight nine ZEBRA"
    assert within_delta(original, substituted, delta=0.1)
    assert not within_delta(original, substituted, delta=0.09999)


def test_within_delta_uses_config_default() -> None:
    original = "a " * 100
    far = "b " * 100
    o, f = original.strip(), far.strip()
    assert within_delta(o, f) == (token_distance(o, f) <= config.DELTA)


def test_tokens_splits_on_whitespace() -> None:
    assert tokens("one  two\tthree\nfour") == ["one", "two", "three", "four"]

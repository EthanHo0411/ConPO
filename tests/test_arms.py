"""Tests for conpo/arms.py's "arm:field=value" identifier parsing
(resolve_arm) and the HYPOTHESES registry's metric strings."""

import pytest

from conpo.arms import HYPOTHESES, ARMS, resolve_arm


def test_resolve_arm_plain_name_uses_environment_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_NO_EXACTNESS", raising=False)
    monkeypatch.delenv("RUN_NO_DISTANCE", raising=False)
    arm, overrides = resolve_arm("full")
    assert arm is ARMS["full"]
    assert overrides.no_exactness is False
    assert overrides.no_distance is False


def test_resolve_arm_no_exactness_identifier_sets_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_NO_EXACTNESS", raising=False)
    arm, overrides = resolve_arm("full:no_exactness=1")
    assert arm is ARMS["full"]
    assert overrides.no_exactness is True
    assert overrides.no_distance is False  # untouched by this identifier


def test_resolve_arm_no_distance_identifier_sets_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUN_NO_DISTANCE", raising=False)
    arm, overrides = resolve_arm("full:no_distance=1")
    assert arm is ARMS["full"]
    assert overrides.no_distance is True
    assert overrides.no_exactness is False


def test_resolve_arm_identifier_overrides_the_environment_value(monkeypatch: pytest.MonkeyPatch) -> None:
    # Environment says the gate is off; the identifier's own "=1" still wins.
    monkeypatch.setenv("RUN_NO_EXACTNESS", "0")
    _, overrides = resolve_arm("full:no_exactness=1")
    assert overrides.no_exactness is True


def test_resolve_arm_unknown_override_field_raises() -> None:
    with pytest.raises(ValueError):
        resolve_arm("full:not_a_field=1")


def test_resolve_arm_unknown_arm_in_identifier_raises() -> None:
    with pytest.raises(ValueError):
        resolve_arm("not_an_arm:no_exactness=1")


def test_hypotheses_h2_rows_read_internal_eval_i_pacc() -> None:
    by_id = {h.id: h for h in HYPOTHESES}
    assert by_id["H2_no_exactness"].metric == "eval_i_prompt_loose"
    assert by_id["H2_no_exactness"].treatment == "full"
    assert by_id["H2_no_exactness"].control == "full:no_exactness=1"
    assert by_id["H2_no_distance"].metric == "eval_i_prompt_loose"
    assert by_id["H2_no_distance"].control == "full:no_distance=1"
    assert by_id["H2_vf_dpo"].metric == "eval_i_prompt_loose"


def test_hypotheses_h1_rows_stay_on_ifbench() -> None:
    by_id = {h.id: h for h in HYPOTHESES}
    assert by_id["H1"].metric == "ifbench_prompt_loose"


def test_hypotheses_h6_multi_unchanged() -> None:
    by_id = {h.id: h for h in HYPOTHESES}
    assert by_id["H6_multi"].metric == "pacc_eval_ii_and_pacc_eval_iii_prompt_loose"
    assert by_id["H6_multi"].treatment == "full:multi=1"

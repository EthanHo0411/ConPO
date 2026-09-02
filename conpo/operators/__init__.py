"""Programmatic corruption operators I_A, one per constraint family.

The contract lives in `registry.py`. Family groups register on import:
`ifeval_a`, `ifeval_b` (the 25 IFEval families), `iftrain_c`, `iftrain_d`
(the 29 IFBench-Train families). Group imports are added here once all four
exist so that a partially built package still imports.
"""

from conpo.operators import ifeval_a, ifeval_b, iftrain_c, iftrain_d  # noqa: F401
from conpo.operators import registry
from conpo.verifiers import FAMILIES


def coverage() -> dict[str, object]:
    """Registration coverage: operator/fixture counts and families with no operator."""
    missing = sorted(set(FAMILIES) - registry.NO_OPERATOR - set(registry.OPERATORS))
    return {
        "registered_operators": len(registry.OPERATORS),
        "registered_fixtures": len(registry.FIXTURES),
        "missing": missing,
    }

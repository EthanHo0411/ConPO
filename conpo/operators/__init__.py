"""Programmatic corruption operators I_A, one per constraint family.

The contract lives in `registry.py`. Family groups register on import:
`ifeval_a`, `ifeval_b` (the 25 IFEval families), `iftrain_c`, `iftrain_d`
(the 29 IFBench-Train families). Group imports are added here once all four
exist so that a partially built package still imports.
"""

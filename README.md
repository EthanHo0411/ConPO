# ConPO

Post-training for verifiable instruction following: repaired/corrupted response pairs trained alongside a clean-generation GRPO objective.

Locked: Qwen3-4B policy every run (later ablations: Qwen3-8B, Llama-3.1-Tulu-3-8B-SFT, Olmo-3-7B-Instruct-SFT, unused today); LoRA r64 alpha128 dropout0.05; 54 constraint families (IFEval 25 + IFBench-Train 29); pool `allenai/IF_multi_constraints_upto5`, eval on `google/IFEval` + held-out `allenai/IFBench_test` (58 disjoint families); official vendored IFEvalG scorers only; judge Qwen3-14B.

Tests: `conda run -n conpo --no-capture-output python -m pytest tests/test_verifiers.py -q`

Layout: `conpo/config.py` constants; `conpo/verifiers.py` Constraint type + build/check/describe_failure + family registry; `conpo/vendor/ifevalg/` vendored checkers (see its VENDOR.md); `tests/` suite + kwargs fixture.

# Vendored: Google IFEval

Source: https://github.com/google-research/google-research/tree/master/instruction_following_eval
Commit: not available. These files were fetched individually via the GitHub
contents API from the `google-research/google-research` monorepo (path
`instruction_following_eval/`), which does not expose a single commit hash
for a subtree fetch the way a full clone would. The file mtimes on disk
reflect the fetch time, not an upstream commit.
Fetched: 2026-09-02

Files copied verbatim except for the import rewrite below:
- evaluation_lib.py
- evaluation_main.py
- instructions.py
- instructions_registry.py
- instructions_util.py
- instructions_test.py
- README.md

Skipped: run.sh (shell wrapper, not needed for library use), requirements.txt
(deps are declared in conpo's own pyproject.toml instead).

## Import rewrite

Every `from instruction_following_eval import X` became `from conpo.vendor.ifeval import X`:
- evaluation_lib.py: `from instruction_following_eval import instructions_registry` -> `from conpo.vendor.ifeval import instructions_registry`
- evaluation_main.py: `from instruction_following_eval import evaluation_lib` -> `from conpo.vendor.ifeval import evaluation_lib`
- instructions.py: `from instruction_following_eval import instructions_util` -> `from conpo.vendor.ifeval import instructions_util`
- instructions_registry.py: `from instruction_following_eval import instructions` -> `from conpo.vendor.ifeval import instructions`
- instructions_test.py: `from instruction_following_eval import instructions` -> `from conpo.vendor.ifeval import instructions`

instructions_util.py has no `instruction_following_eval` imports; unchanged
apart from the header. README.md is kept as upstream documentation verbatim
(its usage snippet still shows the original `instruction_following_eval`
module path and `python3 -m ...` invocation; that CLI path is not how this
vendor copy is called from conpo, see `conpo/scoring.py` instead).

No other edits were made to vendored code. No broken lines were found at
vendor time. `instructions_registry.INSTRUCTION_DICT` has 25 entries after
the rewrite (verified by import).

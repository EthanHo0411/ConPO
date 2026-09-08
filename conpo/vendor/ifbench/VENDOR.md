# Vendored: allenai/IFBench

Source: https://github.com/allenai/IFBench
Commit: fcd289db21d43aaa96c6d9291d32561cd6e19305 (2026-08-30)
Fetched: 2026-09-02

Upstream layout is a repo root (`run_eval.py`, `evaluation_lib.py`) plus a
`ifbench/` subpackage (`instructions.py`, `instructions_registry.py`,
`instructions_util.py`, `classic_instructions.py`, `__init__.py`). This
vendor copy flattens both into one flat `conpo/vendor/ifbench/` package, the
same layout the sibling `conpo/vendor/ifevalg/` vendor uses, so the whole
thing imports as `conpo.vendor.ifbench`.

Files copied (verbatim except for the import rewrite below):
- evaluation_lib.py, run_eval.py (from repo root)
- ifbench/instructions.py, instructions_registry.py, instructions_util.py,
  classic_instructions.py, __init__.py (rewritten, see below)
- README.md, LICENSE
- data/IFBench_test.jsonl, data/sample_output.jsonl (repo root `data/`)
- eval/eval_results_strict.jsonl, eval/eval_results_loose.jsonl (repo root `eval/`)

Skipped: generate_responses.py, config.py, uv.lock, pyproject.toml,
.coverage, __pycache__ (all build/CLI/test-runner plumbing, not needed for
library use); `ifbench/data/IFBench_test.jsonl`, a second, near-identical
copy of the same file bundled inside the upstream package for `importlib.resources`
packaging (differs from the root copy only in trailing whitespace on 2 of
300 prompt strings) -- kept a single copy at `data/IFBench_test.jsonl` since
`__init__.py`'s `data_path()` resolves relative to this package's own
`data/` folder regardless of which copy backs it.

## Import rewrite

Every `from ifbench import X` / `from ifbench.Y import Z` became
`from conpo.vendor.ifbench import X` / `from conpo.vendor.ifbench.Y import Z`:
- __init__.py: `from ifbench import classic_instructions, instructions, instructions_registry, instructions_util` -> `from conpo.vendor.ifbench import ...` (same names)
- evaluation_lib.py: `from ifbench import instructions_registry` -> `from conpo.vendor.ifbench import instructions_registry`
- run_eval.py: `import evaluation_lib` -> `from conpo.vendor.ifbench import evaluation_lib`
- instructions.py: `from ifbench import instructions_util` -> `from conpo.vendor.ifbench import instructions_util`
- instructions_registry.py: `from ifbench import classic_instructions, instructions` -> `from conpo.vendor.ifbench import classic_instructions, instructions`
- classic_instructions.py: `from ifbench import instructions_util` -> `from conpo.vendor.ifbench import instructions_util`; `from ifbench.instructions import Instruction` -> `from conpo.vendor.ifbench.instructions import Instruction`

instructions_util.py has no internal `ifbench` imports; unchanged apart from
the header. README.md is kept as upstream documentation verbatim (its usage
snippets still show `from ifbench import ...` and `pip install ifbench`;
not how this vendor copy is used, see `conpo/scoring.py`).

`instructions.py` imports `emoji` and `syllapy` at module level (used by the
`format:emoji` and `words:odd_even_syllables` checkers); neither was in the
conda env's declared package list, so both were `pip install`ed into the
`conpo` env as a direct consequence of vendoring this module verbatim.
`instructions_util.py` auto-downloads missing NLTK data (`punkt`, `punkt_tab`,
`stopwords`, `averaged_perceptron_tagger_eng`) the first time the module is
imported. Both `instructions.py` and `instructions_util.py` now point nltk at
the repo-root `nltk_data/` (shared with `conpo.verifiers.ensure_nltk`, which
stages the same four packages) before importing nltk, and every
`nltk.download` call passes `download_dir=` explicitly. Both are needed:
nltk's module-level downloader snapshots its default directory at the first
`import nltk` in the process (`~/nltk_data` unless an existing writable entry
is already on `nltk.data.path`), so a path insert after that import changes
lookups but not the download target. On a shared box `~` is another account's
home (changed 2026-09-08 after exactly that happened). A cold machine still
needs network for the first download.

## How scoring works (from run_eval.py + evaluation_lib.py)

A test example is `{key, prompt, instruction_id_list, kwargs}`; a response
file is `{prompt, response}` rows, joined to test examples by exact prompt
text (falling back to a whitespace-stripped match). For each instruction id
in an example, a checker class is built from that instruction's `kwargs`
(None-valued kwargs are dropped before `build_description` in the *strict*
path only -- an asymmetry in the upstream code, not something we changed)
and `check_following(response)` is run. **Strict** checks the raw response
verbatim. **Loose** builds 8 variants of the response (as-is, `*` stripped,
first line dropped, last line dropped, both dropped, and each of the last
three with `*` also stripped) and counts the instruction as followed if ANY
variant passes -- an upper bound on strict. `follow_all_instructions` per
example is AND over its instructions; **prompt-level accuracy** is the
fraction of examples with `follow_all_instructions=True`. **Instruction-level
accuracy** is the fraction of individual (example, instruction_id) checks
that passed, out of the sum of `len(instruction_id_list)` over all examples
(one example can carry more than one instruction).

## Data note: eval/eval_results_*.jsonl predates data/IFBench_test.jsonl

`data/IFBench_test.jsonl` at this commit has 300 prompts;
`eval/eval_results_strict.jsonl` and `eval_results_loose.jsonl` only have 294
rows each. Matching by prompt text (exact-or-stripped) shows the shipped
eval covers 293 of the 300 test prompts cleanly, plus one row (`custom:csv_city`,
its stored `response` is a valid CSV completion) whose stored `prompt` field
is a corrupted/unrelated string that doesn't match anything in the current
test set -- an artifact in the shipped file, not something reproducible from
current `data/`. The other 6 test prompts (positions 268-273, instruction
families `words:vowel`, `count:pronouns`, `custom:character_reverse`) have no
counterpart in the shipped eval at all -- content added to the test set after
the shipped eval was archived. `tests/test_scoring.py` documents and asserts
against this directly rather than silently matching a smaller population.

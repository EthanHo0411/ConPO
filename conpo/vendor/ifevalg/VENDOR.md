# Vendored: open-instruct IFEvalG

Source: https://github.com/allenai/open-instruct, path `open_instruct/IFEvalG/`
Commit: e6c3044aa6201c8bfa4573c1b03142af599e730d
Fetched: 2026-09-02

Files copied verbatim except for the import rewrite below:
- instructions.py
- instructions_registry.py
- instructions_util.py

## Import rewrite

Every `from open_instruct.IFEvalG import X` became `from conpo.vendor.ifevalg import X`:
- instructions.py: `from open_instruct.IFEvalG import instructions_util` -> `from conpo.vendor.ifevalg import instructions_util`
- instructions_registry.py: `from open_instruct.IFEvalG import instructions` -> `from conpo.vendor.ifevalg import instructions`

instructions_util.py has no open_instruct imports; unchanged apart from the header.

No other edits were made to vendored code. No broken lines were found at vendor time.

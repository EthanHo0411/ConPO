# Lessons Learned

> Read FIRST at every session start. Apply every rule below before touching anything.

## Log (newest first -- each entry IS a standing rule)

### 2026-09-08 | offline launchers with an un-staged model
- **What went wrong**: `scripts/fetch_data.py` cached only the policy tokenizer, while every launcher exports `HF_HUB_OFFLINE=1`. The first real run would have failed at `from_pretrained` on the weights.
- **Rule**: a staging step must download everything the offline path loads: weights for `config.POLICY_MODEL` and `config.JUDGE_MODEL`, every dataset in `config`, nltk data. Check by grepping `from_pretrained|load_dataset|snapshot_download` against the staging script.
- **Category**: verification

### 2026-09-08 | a launcher that depends on a package the lock does not hold
- **What went wrong**: `slurm/accelerate_fullft.yaml` configured DeepSpeed ZeRO-2, but `deepspeed` is absent from `pyproject.toml` and `uv.lock`; the script would have failed at `accelerate launch`.
- **Rule**: every import or launcher backend must resolve to a package in `uv.lock` (`grep '^name = "<pkg>"' uv.lock`). The devbox replacement uses torch-native FSDP through accelerate (no new dependency).
- **Category**: planning

### 2026-09-08 | devbox guard scans sourced scripts
- **What went wrong**: `source local/env.sh` was denied because `env.sh` computed its root with `cd "$(dirname ...)/.." && pwd`; the PreToolUse guard scans sourced files and refuses a `cd` whose target is a command substitution.
- **Rule**: launcher scripts on this box derive paths with `readlink -f` and `dirname`, never `cd $(...)`. Probe a candidate command against the guard's stdin before wiring it into a launcher (see `/home/dev/project/ethan/SERVER.md` §3).
- **Category**: implementation

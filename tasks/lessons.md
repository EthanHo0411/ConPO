# Lessons Learned

> Read FIRST at every session start. Apply every rule below before touching anything.

## Log (newest first -- each entry IS a standing rule)

### 2026-09-11 | a colocated vLLM sampler silently sampled from the base model for two full runs
- **What went wrong**: TRL's `VLLMGeneration.generate` calls `self.llm.collective_rpc("reload_weights")` on every single call (a documented workaround for vllm issue 29341), and vLLM's `reload_weights` with no arguments reloads the *original* model from disk, unconditionally -- after `sync_weights()` had already merged and pushed the trained policy for the step. Every GRPO rollout, forward-repair and reverse-corruption sample in both `full_s42` and `ifrlvr_s42` was drawn from base `Qwen/Qwen3-4B`, not the policy being trained. Nothing crashed; rewards never moved, and `kl`/`sampling/sampling_logp_difference/max` sat in the hundreds/tens of nats instead of near zero.
- **Rule**: a colocated sampler (any sleep-mode + vLLM wiring, not this one alone) is verified by comparing its sampling log-probs against the policy's own log-probs on real sampled sequences, before a long run -- not by the run finishing without an exception. Fixed in `ConPOTrainer._install_vllm_reload_guard` (conpo/trainer.py): wraps `collective_rpc` so a `"reload_weights"` call is always followed by `sync_weights()`. GPU-verified on `runs/full_s42/checkpoint-75` + 3 steps: `sampling/sampling_logp_difference/max` per step went from [15.31, 11.94, 12.97] to [1.37, 1.63, 2.28]; `kl` from [337.3, 7.14, 11.0] to [0.048, 0.039, 0.042]; sequences with |vLLM sampling logp - old policy logp| above 5 nats went from 60/384 (max 15.31) to 0/384 (max 2.28), while |ref - policy| gaps stayed the same order (62/384 above 5 nats, max 15.31 -> 78/384, max 29.6 -- the adapter divergence from the frozen reference is real, expected to grow once training runs on-policy).
- **Category**: verification

### 2026-09-08 | transformers 5 changed what apply_chat_template(tokenize=True) returns
- **What went wrong**: the vLLM colocate smoke died in vLLM's input validation with `'>' not supported between 'str' and 'int'`. `conpo/engine.py` passed `apply_chat_template(tokenize=True)` output as token-id lists; under transformers 5.14.1 that call returns a `BatchEncoding` dict, so vLLM's `max(prompt_ids)` saw the key `"input_ids"`.
- **Rule**: every `apply_chat_template(..., tokenize=True)` that wants a plain id list passes `return_dict=False` (valid in transformers 4 and 5). The CPU test suite could not catch this because the tiny-model tests never reach a real vLLM engine; a GPU smoke of each generation path is part of done.
- **Category**: implementation

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

### 2026-09-11 | Two builders hung for 40 minutes on node-local writes typed on the command line
- **What went wrong**: Builder E launched its smoke with a chained `rm -rf /node-local-cache/... && mkdir -p ... && nohup bash local/train.sh` and builder G wrote a keys file with `mkdir -p /node-local-cache/... && cat > ...`; the devbox guard turns any write under `/node-local-cache` typed on a command line into an "ask" prompt, which no one could answer with Ethan away, so both tool calls waited forever while both GPUs sat idle.
- **Rule**: Subagent briefs say it in words: never `mkdir`, `rm`, `cp`, `cat >` or `tee` into `/node-local-cache` from a command line; launchers create `RUN_OUT_DIR` themselves, helper files (key lists, configs) go in the session scratchpad, and a stuck subagent is detected by an untouched transcript file plus an idle GPU, then stopped and its work resumed by the session.
- **Category**: implementation

### 2026-09-11 | Resumed run kept the old checkpoint cadence
- **What went wrong**: full_s42 was resumed from checkpoint-25 with RUN_SAVE_STEPS=15 and saved nothing at step 30. transformers copies args.save_steps into TrainerState, then replaces the state with the checkpoint's trainer_state.json (only a warning: "save_steps: 15 (from args) != 25 (from trainer_state.json)"), and DefaultFlowCallback saves on state.save_steps.
- **Rule**: a launch knob that lives in TrainerState (save_steps, logging_steps) must be re-asserted in on_train_begin (scripts/train.py SaveCadenceCallback, c390297). After any resume, read the launch log for that transformers warning and confirm the first expected checkpoint lands.
- **Category**: implementation

### 2026-09-11 | Node-local keeps only the last two checkpoints
- **What went wrong**: a diagnostic eval of full_s42's checkpoint-50 pointed at the node-local run directory and crashed: `save_total_limit=2` (scripts/train.py) had already deleted checkpoints 25 and 50 when 75 and 90 were written.
- **Rule**: evaluate or resume older checkpoints from the NFS harvest (`runs/<run>/checkpoint-N`, kept by `KEEP_CHECKPOINTS=1 bash local/harvest.sh <run>` every tick); node-local holds only the newest two.
- **Category**: implementation

### 2026-09-13 | Pod restart removed the policy model from the shared hub
- **What went wrong**: after the restart wiped node-local, both resumed runs died at model load ("couldn't find them in the cached files"): the shared hub had lost `models--Qwen--Qwen3-4B` while keeping Qwen3-14B. The venv fallback to `ConPO/.venv` on NFS worked, so the wipe surfaced one layer later than expected.
- **Rule**: after any node-local wipe, verify the three layers in order before launching: venv (`ls $CONPO_VENV/bin/python`), hub models (`ls $HF_HUB_CACHE/models--Qwen--Qwen3-4B/snapshots/*/model.safetensors.index.json` and the judge), run directory (newest checkpoint with conpo_state.json). Re-download from a script with the offline variables unset; never launch chains that cascade into the next run before the first step of the resumed run is confirmed.
- **Category**: implementation

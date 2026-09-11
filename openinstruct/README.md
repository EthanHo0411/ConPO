# IF-RLVR on open-instruct's GRPO trainer, with ConPO's settings

Runs AllenAI open-instruct's `grpo_fast.py` (the code the IFBench paper trained with) on
`Qwen/Qwen3-4B` non-thinking, LoRA, on ConPO's own sampling/optimiser knobs and 15,000-prompt
subset, on one H100. Four source patches are required (below); everything else is a flag.

## Files

- `setup_venv.sh` -- copies `refs/open-instruct` into a working copy under
  `/node-local-cache/cache/open-instruct-src`, applies `patches/*.patch` to that copy, then
  `uv sync --frozen` into `/node-local-cache/cache/open-instruct-venv`. `refs/open-instruct`
  itself is never modified.
- `patches/lora_peft.patch`, `patches/enable_thinking_flag.patch`,
  `patches/local_dataset_tool_discovery.patch`, `patches/lora_vllm_weight_sync.patch` -- see
  "Patches" below.
- `convert_subset.py` -- converts ConPO's 15k IF-RLVR subset to the schema
  `IFEvalVerifier` expects. `--limit N` for a head-slice (smoke test).
- `train_ifrlvr.sh` -- single-GPU launch. `GPU=<n> [SMOKE=1] [STEPS=<n>] bash train_ifrlvr.sh`.
- `export_adapter.py` -- copies the saved adapter into a clean directory and (unless
  `--skip-verify`) loads it with PEFT on CPU.
- `../tests/test_openinstruct_convert.py` -- converter tests (schema, family mapping, limit).

Run everything with `bash <file>`, never by typing a `/node-local-cache` write on the command
line directly (it hangs on an unanswerable "ask" prompt with nobody there to answer it).

## Feasibility: what open-instruct honours, what it cannot

| ConPO setting | open-instruct equivalent | Verdict |
|---|---|---|
| `Qwen/Qwen3-4B` | `--model_name_or_path Qwen/Qwen3-4B` | direct |
| Non-thinking (`enable_thinking=False`) | `--enable_thinking False` (new flag, patch 2) | **needs patch** |
| LoRA r=64/alpha=128/dropout=0.05 on q,k,v,o,gate,up,down | `--use_peft --lora_r 64 --lora_alpha 128 --lora_dropout 0.05 --lora_target_modules ...` | **needs patch** (flags parse but were never wired to `get_peft_model`, patch 1) |
| G=8 samples/prompt, 16 prompts/step (batch 128) | `--num_samples_per_prompt_rollout 8 --num_unique_prompts_rollout 16` | direct |
| 1024 completion / 2048 prompt tokens | `--response_length 1024 --max_prompt_token_length 2048` | direct |
| `MAX_MODEL_LEN=3584` (2048+1024+512 KV-cache margin) | always `max_prompt_token_length + response_length` = 3072, no override flag | **gap, accepted** (see below) |
| temperature 0.8, top-p 0.95 | `--temperature 0.8 --vllm_top_p 0.95` | direct |
| LR 1e-5, warmup 0.03, cosine, grad clip 1.0 | `--learning_rate 1e-5 --warmup_ratio 0.03 --lr_scheduler_type cosine --max_grad_norm 1.0` | direct |
| KL beta 0.001 | `--beta 0.001` | direct |
| Symmetric clip epsilon 0.2 | `--clip_lower 0.2 --clip_higher 0.2` (DAPO's asymmetric default is 0.2/0.272; symmetric is reachable, just not the default) | direct once set |
| GRPO group-std advantage normalisation | `--advantage_normalization_type standard` | **needs flag** (see below) |
| 937 steps over the subset | no `--num_steps`; derived as `total_episodes // 128` | direct but indirect: `--total_episodes $((937*128))` |
| seed 42 | `--seed 42` (already the open-instruct default) | direct |
| 15,000-prompt subset | `--dataset_mixer_list <converted.jsonl> 1.0` | direct (via `convert_subset.py`) |
| single H100 | `--single_gpu_mode` | direct |
| Token-weighted vs response-weighted loss aggregation | no equivalent (`--loss_denominator` is `"token"` or a fixed float only) | **gap, accepted** (see below) |

**The `MAX_MODEL_LEN` gap, accepted for this run:** ConPO's extra 512-token margin over
prompt+completion is a KV-cache safety margin, not a correctness requirement -- every prompt is
already capped to <= 2048 tokens and every completion to <= 1024 (`open_instruct/data_loader.py`'s
`vllm_max_model_len = max_prompt_token_length + max(response_length, eval_response_length)`), so
open-instruct's derived 3072 never truncates a real sequence; it only removes ConPO's rounding
headroom. Not patched: it only affects vLLM's own internal accounting, and a flags-only launch
was preferred per the brief over adding a third patch for a non-correctness margin.

**Advantage normalisation, matched by flag:** ConPO's `scripts/train.py` builds `GRPOConfig`
without setting `scale_rewards`, so it runs under TRL's default `scale_rewards="group"`
(`trl/trainer/grpo_config.py:779-790`), which divides each group's advantages by that group's
reward std (`trl/trainer/grpo_trainer.py:2687-2706`). open-instruct's
`StreamingDataLoaderConfig.advantage_normalization_type` defaults to `"centered"` (mean-only,
no std division; `open_instruct/data_loader.py:421`, applied at `:1479-1484`), but already
supports a `"standard"` branch that divides by std the same way
(`open_instruct/data_loader.py:1479-1481`). `train_ifrlvr.sh` passes
`--advantage_normalization_type standard` so both runs use the same std-normalised advantage
estimator.

**Loss aggregation gap, accepted for this run:** ConPO's `conpo/config.py:79` sets
`LOSS_TYPE = "grpo"` explicitly (TRL 1.9.2 defaults to `"dapo"`), which TRL's
`grpo_trainer.py:3124-3125` computes as
`((per_token_loss*mask).sum(-1)/mask.sum(-1).clamp(min=1.0)).mean()`: each response is
normalised by its own token count first, then averaged across responses (response-weighted).
open-instruct's `grpo_fast.py:719` divides by `GRPOExperimentConfig.loss_denominator`
(`open_instruct/grpo_utils.py:136-139`), which defaults to `"token"` (one scalar denominator
over the whole batch, i.e. token-weighted, open-instruct's bnpo/dapo family) and only otherwise
accepts a fixed float, not a per-sequence-then-batch-average mode. There is no
`--loss_denominator` setting that reproduces TRL's `loss_type="grpo"` averaging, so this is a
genuine cannot-honour gap: the two runs weight tokens in longer vs. shorter responses
differently, disclosed here rather than silently inherited.

## Patches (both to a copy, `refs/open-instruct` untouched; `patch -p1` from `setup_venv.sh`)

### `patches/lora_peft.patch`

**Why:** `ModelConfig`'s `use_peft`/`lora_r`/`lora_alpha`/`lora_dropout`/`lora_target_modules`
fields parse from the CLI (`open_instruct/model_utils.py:191-207`,
`open_instruct/grpo_fast.py:2392-2411`), but nothing in `grpo_fast.py` ever calls
`get_peft_model`/`LoraConfig` on the policy: `from_pretrained` builds it with plain
`AutoModelForCausalLM.from_pretrained` (`open_instruct/grpo_fast.py:353-359`) and hands that
straight to `deepspeed.initialize` (`:384-391`). `save_model`'s
`isinstance(model_to_save, PeftModel)` branch (`:883-891`) is consequently dead code: the policy
is never a `PeftModel`, so a training run with `--use_peft` set produces a full-weight
checkpoint, never an adapter. Confirmed by grep: zero hits for `get_peft_config`/`get_peft_model`/
`LoraConfig` anywhere in `grpo_fast.py`. Sibling scripts already do this the same way this patch
does: `open_instruct/dpo_tune_cache.py:363-373`, `open_instruct/finetune.py` (same pattern).

**What it does:** after the base model loads and before `deepspeed.initialize`, wraps
`self.policy` with `get_peft_model(self.policy, LoraConfig(...))` built from `model_config`'s
existing fields, only when `model_config.use_peft` is set. This makes the existing
`isinstance(model_to_save, PeftModel)` save-path branch actually fire, and
`get_optimizer_grouped_parameters` (already filters on `p.requires_grad`,
`open_instruct/utils.py:1507-1533`) automatically restricts the optimizer to the LoRA
parameters, matching the sibling scripts' behaviour with no further change needed.

The same patch also fixes one more `save_model` bug the PEFT wrap exposes: its
`tie_word_embeddings` corner case (`open_instruct/grpo_fast.py:875-877`) removes the exact string
`"lm_head.weight"` from the saved `state_dict`'s key set, but a `PeftModel`'s keys are all
prefixed `"base_model.model."`, so the exact-string check never matches and the mismatch assert
right after it fails (confirmed empirically:
`AssertionError: mismatch keys {'base_model.model.lm_head.weight'}`, the smoke run's actual first
`save_model` attempt, all 3 training steps otherwise already correct). Matched by suffix instead
of exact string, so it covers both the plain and the PEFT-wrapped case.

One more bug the same branch exposes, unrelated to PEFT naming: `if self.stage == 3:`
(`open_instruct/grpo_fast.py:886`) references an attribute nothing on `PolicyTrainerRayProcess`
ever sets (grep confirms this is the class's only use of `self.stage`) -- dead code before this
patch ever produced a `PeftModel` for it to run against. Confirmed empirically:
`AttributeError: 'PolicyTrainerRayProcess' object has no attribute 'stage'`. Fixed to
`self.args.deepspeed_stage`, the field name used everywhere else in the class for the same
DeepSpeed ZeRO stage; with `--deepspeed_stage 2` this branch is a no-op either way (it only
matters for stage 3, where the ZeRO-3 gather makes `save_pretrained`'s own adapter write
incomplete), but the attribute error crashed every save unconditionally before this fix, at any
stage.

### `patches/enable_thinking_flag.patch`

**Why:** the one call site that renders the RL rollout prompt, `rlvr_tokenize_v3`
(`open_instruct/dataset_transformation.py:1802-1830`, registered as `"rlvr_tokenize_v1"` in
`TRANSFORM_FNS`), calls `tokenizer.apply_chat_template(..., tools=tools_for_template)` with no
`enable_thinking` kwarg, and neither `TokenizerConfig` nor `StreamingDataLoaderConfig` exposes a
way to pass one through -- `setup_datasets`'s `transform_fn_args[0]` dict
(`open_instruct/grpo_fast.py:1140-1147`) is built from a fixed set of fields with no CLI hook.
Qwen3's own chat template (read directly from the cached
`tokenizer_config.json`) only emits the closing `<think>\n\n</think>\n\n` block when
`enable_thinking` is explicitly passed and `False`
(`{%- if enable_thinking is defined and enable_thinking is false %}`); left unpassed, the
variable is undefined and Qwen3 renders in its default thinking-enabled mode, the opposite of
ConPO's locked `POLICY_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}`. There is no existing
flag or `CHAT_TEMPLATES` entry for Qwen3 non-thinking (that dict's only non-thinking entries are
OLMo-specific, `open_instruct/dataset_transformation.py:387,472`).

**What it does:** adds one new field, `StreamingDataLoaderConfig.enable_thinking: bool | None =
None` (same `Optional[bool]` pattern `ModelConfig.use_cache` already uses in this codebase, so it
gets the same CLI parsing for free), threads it through `setup_datasets`'s `transform_fn_args[0]`
dict, and adds a matching `enable_thinking: bool | None = None` parameter to `rlvr_tokenize_v3`
that gets passed straight into `apply_chat_template`. Default `None` changes nothing for every
existing open-instruct caller (an unreferenced template variable is simply ignored); our launch
script passes `--enable_thinking False`.

### `patches/local_dataset_tool_discovery.patch`

**Why:** found empirically (not in the read-only review): `main()` unconditionally calls
`initialize_tools_and_envs`, which calls `_discover_tools_from_datasets`
(`open_instruct/grpo_fast.py:2055-2072`) to scan every dataset in `--dataset_mixer_list` for a
`tools`/`env_config` column, whether or not any `--tools` are configured. That scan calls
`datasets.load_dataset(dataset_name, split=split)` directly with no local-file branch, unlike
the main dataset loader's own `__post_init__`
(`open_instruct/dataset_transformation.py:1969-1974`, `os.path.exists(...) and
name.endswith(".jsonl")` -> `load_dataset("json", data_files=..., split=...)`). Any
`--dataset_mixer_list <local .jsonl path> 1.0` therefore crashes this scan with
`FileNotFoundError: Couldn't find any data file at <path>` before touching a GPU -- confirmed by
running the smoke launch unpatched first.

**What it does:** gives `_discover_tools_from_datasets`'s loop the same local `.jsonl`/`.parquet`
branch the main loader already has, verbatim.

### `patches/lora_vllm_weight_sync.patch`

**Why:** found empirically (also not in the read-only review, since it only surfaces once
`--use_peft` actually produces a `PeftModel`, i.e. after patch 1): `broadcast_to_vllm` hands
`self.model.module` straight to `vllm_utils.broadcast_weights_to_vllm`, which iterates
`model.named_parameters()` directly and sends every name/tensor to the colocated vLLM engine
as-is (`open_instruct/vllm_utils.py:1336-1353,1356-1379`, no PEFT awareness anywhere in that
file). None of a `PeftModel`'s own parameter names exist on vLLM's plain `Qwen3ForCausalLM`, for
three separate reasons, the first two each hit as their own traceback while getting the smoke
test running: (1) every name is prefixed `base_model.model.`
(`ValueError: There is no module or parameter named 'base_model' in Qwen3ForCausalLM`); (2) PEFT's
`lora.Linear` wrapper renames the wrapped layer's own weight from `<layer>.weight` to
`<layer>.base_layer.weight`, one level deeper (vLLM's own qkv-fusion loader raised
`KeyError: 'layers.0.self_attn.qkv_proj.base_layer.weight'`); (3) the interleaved `lora_A`/
`lora_B` submodule parameters have no counterpart at all on the plain model (necessary for the
same reason as (2), not separately hit as its own traceback since (1) and (2) already had to be
fixed first).

**What it does:** in `broadcast_to_vllm`, when the policy is a `PeftModel`: calls
`merge_adapter()` (folds the LoRA delta into the base weight tensors in place -- the same PEFT
pattern TRL uses for LoRA + colocated vLLM), builds a small `_NamedParamsView` over the merged
parameters with the `base_model.model.` prefix stripped, `.base_layer.` collapsed back to `.`,
and the `lora_A`/`lora_B` entries filtered out (so names line up with vLLM's plain
`Qwen3ForCausalLM`), passes that view (not the `PeftModel` itself) to `broadcast_weights_to_vllm`,
then calls `unmerge_adapter()` in a `finally` to restore the trainable split before the next
training step. Safe under the IPC backend (`single_gpu_mode`): `_broadcast_weights_ipc` reads
`p.data` and sends it synchronously before returning -- nothing is cloned or sent after
`unmerge_adapter()` runs.

## Data: dataset schema and family mapping

`IFEvalVerifier.__call__` (`open_instruct/ground_truth_utils.py:310-354`) reads three row
columns: `messages` (a single user turn), `ground_truth` (`ast.literal_eval`'d, NEVER
`json.loads`'d, to `[{"instruction_id": [...], "kwargs": [...]}]`), and `dataset` (must
case-insensitively equal `"ifeval"`, `ground_truth_utils.py:321` + `:1273`). `convert_subset.py`
builds exactly this from ConPO's `PoolRow.constraints` (`{"iid", "kwargs"}` pairs), reusing
ConPO's own `cap_pool_rows`/`cap_prompt_messages` (`conpo/prompts.py`) so the prompt text is
byte-identical to what ConPO's own TRL run sees for the same row, and writes `repr(...)`, never
`json.dumps(...)`.

Family-id mapping is the **identity** mapping: ConPO's vendored `conpo/vendor/ifevalg` checker
package is a copy of open-instruct's own `open_instruct/IFEvalG`
(`conpo/vendor/ifevalg/VENDOR.md`, commit `e6c3044a`), and `conpo/verifiers.py:99` already asserts
ConPO's 54-family registry equals that vendored registry's key set -- `constraints[i]["iid"]`
passes straight through as `instruction_id[i]`, no rename table. `convert_subset.py`'s
`count_unmapped_families` re-checks every converted row against that same registry regardless
(defensive, in case a future subset draw ever picks up a family outside ConPO's 54): both the
64-row smoke slice and the full 15,000-row subset converted with **0** unmapped rows (see the
Report at the repo root of this task's output for the exact run).

## Environment

`refs/open-instruct/pyproject.toml`'s default `uv` groups (`dev`, `cuda12`) already resolve
`torch==2.10.0+cu128` / `vllm==0.19.1`, and the box's driver (580.126.20, CUDA 13 runtime) runs
cu128 wheels fine -- no group override needed. `flash-attn`, `flash-attn-4` and `tilelang` all
resolve to prebuilt wheels (no local CUDA compile); `deepspeed` has no PyPI wheel and builds from
its sdist, but that build is pure Python (GPU ops JIT-compile at first use). `refs/open-instruct`
has no git tags, so `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_OPEN_INSTRUCT` must be set before the
editable install, exactly as open-instruct's own `Dockerfile` and debug script already do.

`TRITON_CACHE_DIR` must be set to an in-boundary path before anything imports `deepspeed`:
DeepSpeed's op_builder reads that variable directly (not the `XDG_CACHE_HOME` `.ethan-env`
already redirects) and otherwise falls back to `$HOME/.triton/autotune`, outside the boundary
(discovered while writing `setup_venv.sh`; see `research/tasks/lessons.md`).

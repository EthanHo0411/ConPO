"""Hypothesis test: does merging the LoRA delta into bf16 base weights (the
way TRL syncs the colocated vLLM engine) round the delta away?

See /home/dev/project/ethan/.cache/tmp/claude-1001/-home-dev-project-ethan-research/
84bb2180-5c56-4110-9294-50df603afb3c/scratchpad/brief_diag_merge.md for the
full brief. Standalone script, no dependency on scripts/train.py or the
GRPOTrainer: loads checkpoint-75's adapter through plain PEFT, generates 64
sequences, and scores each one's completion tokens under five conditions
(C1..C5, see the brief), all HF-side, no vLLM.

FOOTGUN, verified empirically before writing the scoring code (a toy LoRA
Linear, see the commit history / this session's transcript): PEFT's
`disable_adapter()` context manager checks `if self.merged: self.unmerge()`
inside every wrapped layer's forward (peft/tuners/lora/layer.py Linear.forward).
Calling it AFTER `merge_adapter()` therefore silently unmerges back to pure
base weights before scoring, instead of scoring the merged (rounded) state --
C3 and C4 would come out identical by construction, not by the hypothesis
being false. TRL's own vLLM sync (trl/generation/vllm_generation.py
sync_weights, ~line 459) never calls disable_adapter after merging either: it
merges, reads `named_parameters()` directly, then unmerges. C4 below mirrors
that: merge, score with a plain forward (no disable_adapter), unmerge.

Conditions, same completion tokens throughout:
  C1: adapter enabled, model.train() (LoRA dropout 0.05 active)
  C2: adapter enabled, model.eval()
  C3: model.eval(), `with model.disable_adapter():` (base weights, no merge)
  C4: model.merge_adapter() then a plain scored forward (no disable_adapter,
      see FOOTGUN above), then model.unmerge_adapter()
  C5: a separate float32 copy of the base model with the LoRA delta
      (scaling * B @ A, computed in float32) added directly into each target
      Linear's weight -- no PEFT wrapper, no bf16 rounding anywhere.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import PeftModel
from peft.tuners.lora.layer import Linear as LoraLinear
from transformers import AutoModelForCausalLM, AutoTokenizer

from conpo import config
from conpo.prompts import cap_prompt_messages

REPO = Path("/home/dev/project/ethan/ConPO")
CHECKPOINT_DIR = REPO / "runs/full_s42/checkpoint-75"
TRAIN_JSONL = REPO / "data/train.jsonl"
SUBSET_KEYS = REPO / "subsets/train_15000_seed42.keys"
DEFAULT_OUT_DIR = Path(
    "/home/dev/project/ethan/.cache/tmp/claude-1001/-home-dev-project-ethan-research/"
    "84bb2180-5c56-4110-9294-50df603afb3c/scratchpad/diag_merge_out"
)
NATS_THRESHOLD = 5.0
GEN_SEED = 0
PAIRS = [("C1", "C2"), ("C1", "C3"), ("C2", "C3"), ("C3", "C4"), ("C2", "C4"), ("C2", "C5"), ("C3", "C5")]


def load_prompt_rows(num_prompts: int) -> list[dict[str, str]]:
    """First `num_prompts` keys of SUBSET_KEYS, matched to their prompt text
    in TRAIN_JSONL (streamed, stops once every key is found; not a full
    conpo.data.load_subset, see the brief's lighter-path allowance). Returns
    rows in SUBSET_KEYS order.
    """
    target_keys: list[str] = []
    with SUBSET_KEYS.open(encoding="utf-8") as f:
        for line in f:
            key = line.strip()
            if key:
                target_keys.append(key)
            if len(target_keys) >= num_prompts:
                break
    remaining = set(target_keys)
    found: dict[str, str] = {}
    with TRAIN_JSONL.open(encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            row = json.loads(line)
            key = row["key"]
            if key in remaining:
                found[key] = row["prompt"]
                remaining.discard(key)
    missing = [k for k in target_keys if k not in found]
    if missing:
        raise RuntimeError(f"subset key(s) not found in {TRAIN_JSONL}: {missing}")
    return [{"key": k, "prompt": found[k]} for k in target_keys]


def build_prompt_ids(rows: list[dict[str, str]], tokenizer: Any) -> list[dict[str, Any]]:
    """Caps each row's prompt with conpo.prompts.cap_prompt_messages (the
    same function scripts/train.py calls) and renders it through the policy
    chat template exactly as GRPOTrainer._tokenize_prompts does: add_generation_prompt=True,
    tokenize=True, return_dict=True, config.POLICY_CHAT_TEMPLATE_KWARGS (non-thinking).
    """
    built: list[dict[str, Any]] = []
    for row in rows:
        messages, truncated = cap_prompt_messages(
            [{"role": "user", "content": row["prompt"]}], tokenizer, config.MAX_PROMPT_TOKENS
        )
        tokenized = tokenizer.apply_chat_template(
            conversation=[messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            **config.POLICY_CHAT_TEMPLATE_KWARGS,
        )
        prompt_ids = list(tokenized["input_ids"][0])
        built.append({"key": row["key"], "prompt_ids": prompt_ids, "truncated": truncated})
    return built


def _trim_trailing_pad(ids: list[int], pad_token_id: int) -> list[int]:
    end = len(ids)
    while end > 1 and ids[end - 1] == pad_token_id:
        end -= 1
    return ids[:end]


def generate_completions(
    model: PeftModel,
    prompt_entry: dict[str, Any],
    num_completions: int,
    max_new_tokens: int,
    pad_token_id: int,
    device: torch.device,
) -> list[list[int]]:
    """`num_completions` samples from `model` (adapter enabled, eval mode,
    caller's responsibility) at config.TEMPERATURE / config.TOP_P, trimmed of
    trailing padding.
    """
    prompt_ids = prompt_entry["prompt_ids"]
    input_ids = torch.tensor([prompt_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_completions,
            pad_token_id=pad_token_id,
        )
    prompt_len = len(prompt_ids)
    return [_trim_trailing_pad(row[prompt_len:].tolist(), pad_token_id) for row in out]


def score_sequence(
    model: torch.nn.Module, prompt_ids: list[int], completion_ids: list[int], temperature: float, device: torch.device
) -> list[float]:
    """Per-token log-probs of `completion_ids` given `prompt_ids`, temperature-scaled
    the way GRPOTrainer._get_per_token_logps_and_entropies does (trl/trainer/grpo_trainer.py
    ~line 1435): forward the full sequence, keep the last len(completion_ids) logit
    positions, divide by temperature, log_softmax, gather at the actual token ids.
    Batch size 1 (one sequence, no padding) so no attention-mask bookkeeping is needed.
    """
    full_ids = torch.tensor([prompt_ids + completion_ids], device=device)
    attention_mask = torch.ones_like(full_ids)
    n_completion = len(completion_ids)
    with torch.no_grad():
        outputs = model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    logits = logits[:, -n_completion:, :]
    logits = logits / temperature
    logps = F.log_softmax(logits, dim=-1)
    token_ids = full_ids[:, -n_completion:]
    gathered = torch.gather(logps, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    return gathered[0].float().cpu().tolist()


def build_fp32_merged_model(peft_model: PeftModel, model_name: str, device: torch.device) -> tuple[torch.nn.Module, int]:
    """A fresh float32 copy of the base model with the trained adapter's
    delta (scaling * B @ A, computed in float32) added directly into each
    target Linear's weight. No PEFT wrapper: this is "what vLLM would hold
    if its merge kept the true delta instead of rounding it into bf16".
    """
    fp32_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    fp32_model.to(device)
    fp32_model.eval()
    applied = 0
    for name, module in peft_model.named_modules():
        if not isinstance(module, LoraLinear) or "default" not in module.lora_A:
            continue
        plain_name = name.removeprefix("base_model.model.")
        target = fp32_model.get_submodule(plain_name)
        weight_a = module.lora_A["default"].weight.detach().to(torch.float32)
        weight_b = module.lora_B["default"].weight.detach().to(torch.float32)
        scaling = module.scaling["default"]
        delta = (weight_b @ weight_a) * scaling
        if delta.shape != target.weight.shape:
            raise RuntimeError(f"delta shape {tuple(delta.shape)} != {plain_name} weight shape {tuple(target.weight.shape)}")
        with torch.no_grad():
            target.weight.add_(delta.to(device=target.weight.device))
        applied += 1
    expected = sum(1 for m in peft_model.modules() if isinstance(m, LoraLinear) and "default" in m.lora_A)
    if applied != expected:
        raise RuntimeError(f"fp32 merge touched {applied} modules, expected {expected}")
    return fp32_model, applied


def score_all_conditions(
    peft_model: PeftModel, fp32_model: torch.nn.Module, sequences: list[dict[str, Any]], temperature: float, device: torch.device
) -> None:
    peft_model.train()
    for seq in sequences:
        seq["logps"]["C1"] = score_sequence(peft_model, seq["prompt_ids"], seq["completion_ids"], temperature, device)

    peft_model.eval()
    for seq in sequences:
        seq["logps"]["C2"] = score_sequence(peft_model, seq["prompt_ids"], seq["completion_ids"], temperature, device)

    with peft_model.disable_adapter():
        for seq in sequences:
            seq["logps"]["C3"] = score_sequence(peft_model, seq["prompt_ids"], seq["completion_ids"], temperature, device)

    peft_model.merge_adapter()  # no disable_adapter() after this -- see module docstring FOOTGUN
    try:
        for seq in sequences:
            seq["logps"]["C4"] = score_sequence(peft_model, seq["prompt_ids"], seq["completion_ids"], temperature, device)
    finally:
        peft_model.unmerge_adapter()

    for seq in sequences:
        seq["logps"]["C5"] = score_sequence(fp32_model, seq["prompt_ids"], seq["completion_ids"], temperature, device)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[idx]


def _flatten_pair(sequences: list[dict[str, Any]], a: str, b: str) -> list[tuple[dict[str, Any], int, float]]:
    out: list[tuple[dict[str, Any], int, float]] = []
    for seq in sequences:
        for pos, (va, vb) in enumerate(zip(seq["logps"][a], seq["logps"][b])):
            out.append((seq, pos, abs(va - vb)))
    return out


def _decode_context(tokenizer: Any, seq: dict[str, Any], pos: int, window: int = 3) -> dict[str, Any]:
    full_ids = seq["prompt_ids"] + seq["completion_ids"]
    abs_pos = seq["prompt_len"] + pos
    lo = max(0, abs_pos - window)
    hi = min(len(full_ids), abs_pos + window + 1)
    context_ids = full_ids[lo:hi]
    return {
        "token": tokenizer.decode([full_ids[abs_pos]]),
        "context": tokenizer.decode(context_ids),
        "context_tokens": [tokenizer.decode([t]) for t in context_ids],
    }


def write_outputs(out_dir: Path, sequences: list[dict[str, Any]], tokenizer: Any, meta: dict[str, Any]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "diag_merge.jsonl").open("w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(json.dumps(seq) + "\n")

    pair_stats: dict[str, Any] = {}
    for a, b in PAIRS:
        gaps = [g for _, _, g in _flatten_pair(sequences, a, b)]
        pair_stats[f"{a}-{b}"] = {
            "n_tokens": len(gaps),
            "max": max(gaps) if gaps else 0.0,
            "p99": _percentile(gaps, 99),
            "count_gt_5nats": sum(1 for g in gaps if g > NATS_THRESHOLD),
        }

    c2_c3_sorted = sorted(_flatten_pair(sequences, "C2", "C3"), key=lambda item: -item[2])[:10]
    top10 = []
    for seq, pos, gap in c2_c3_sorted:
        ctx = _decode_context(tokenizer, seq, pos)
        top10.append(
            {
                "key": seq["key"],
                "sample_index": seq["sample_index"],
                "position": pos,
                "gap_C2_C3": gap,
                "token": ctx["token"],
                "context": ctx["context"],
                "context_tokens": ctx["context_tokens"],
                "logp": {c: seq["logps"][c][pos] for c in ("C1", "C2", "C3", "C4", "C5")},
            }
        )

    summary = {"meta": meta, "pair_stats": pair_stats, "top10_c2_c3_gap": top10, "n_sequences": len(sequences)}
    (out_dir / "diag_merge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--num-completions", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    device = torch.device("cuda")
    t0 = time.monotonic()

    tokenizer = AutoTokenizer.from_pretrained(config.POLICY_MODEL)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rows = load_prompt_rows(args.num_prompts)
    prompt_entries = build_prompt_ids(rows, tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(config.POLICY_MODEL, dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(base_model, str(CHECKPOINT_DIR))
    peft_model.to(device)
    peft_model.eval()
    t_load = time.monotonic()

    torch.manual_seed(GEN_SEED)
    sequences: list[dict[str, Any]] = []
    for entry in prompt_entries:
        completions = generate_completions(peft_model, entry, args.num_completions, args.max_new_tokens, pad_token_id, device)
        for i, completion_ids in enumerate(completions):
            sequences.append(
                {
                    "key": entry["key"],
                    "sample_index": i,
                    "prompt_ids": entry["prompt_ids"],
                    "prompt_len": len(entry["prompt_ids"]),
                    "completion_ids": completion_ids,
                    "completion_len": len(completion_ids),
                    "prompt_truncated": entry["truncated"],
                    "logps": {},
                }
            )
    t_gen = time.monotonic()

    fp32_model, fp32_modules_applied = build_fp32_merged_model(peft_model, config.POLICY_MODEL, device)
    t_fp32 = time.monotonic()

    score_all_conditions(peft_model, fp32_model, sequences, config.TEMPERATURE, device)
    t_score = time.monotonic()

    meta = {
        "num_prompts": args.num_prompts,
        "num_completions": args.num_completions,
        "max_new_tokens": args.max_new_tokens,
        "checkpoint": str(CHECKPOINT_DIR),
        "policy_model": config.POLICY_MODEL,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "gen_seed": GEN_SEED,
        "fp32_modules_merged": fp32_modules_applied,
        "prompt_keys": [e["key"] for e in prompt_entries],
        "prompts_truncated": sum(1 for e in prompt_entries if e["truncated"]),
        "timing_seconds": {
            "model_load": t_load - t0,
            "generation": t_gen - t_load,
            "fp32_model_build": t_fp32 - t_gen,
            "scoring": t_score - t_fp32,
            "total": t_score - t0,
        },
    }
    summary = write_outputs(Path(args.out_dir), sequences, tokenizer, meta)
    print(f"diag_merge: {len(sequences)} sequences, total {meta['timing_seconds']['total']:.1f}s, outputs under {args.out_dir}")
    print(json.dumps(summary["pair_stats"], indent=2))


if __name__ == "__main__":
    main()

"""External eval: IFEval (google/IFEval) and IFBench (IFBench_test), greedy decoding.

--score-only skips generation entirely and scores canned responses instead.
A canned record is either a full scoring record (already carrying
`instruction_id_list` and `kwargs`, e.g. hand-written for a test) or a
minimal {"key", "response"} pair joined by key against the loaded reference
row of whichever dataset (IFEval / IFBench) has a matching key.

Alongside `--out`'s JSON, `<out>.verdicts.jsonl` carries one line per
scored prompt (key, benchmark, prompt_strict, prompt_loose, per-instruction
strict/loose lists) -- the per-prompt paired bootstrap in conpo.stats and
scripts/analyze.py need per-prompt verdicts, not just conpo.scoring's
aggregate ScoreReport.
"""

import argparse
import dataclasses
import functools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import datasets

from conpo import config, scoring
from conpo.vendor.ifbench import evaluation_lib as _ifbench_lib
from conpo.vendor.ifeval import evaluation_lib as _ifeval_lib
from conpo.verifiers import IFEVAL_FAMILIES

_IFEVAL_SET = set(IFEVAL_FAMILIES)


def left_truncate_prompt(tokenizer: Any, prompt: str, max_tokens: int) -> tuple[str, bool]:
    """Left-truncate `prompt`'s tokenisation (policy tokenizer) to its last
    `max_tokens` tokens, the eval-side counterpart of the load-time cap
    training applies through conpo/prompts.py's cap_pool_rows (TRL 1.9.2's
    GRPOConfig has no max_prompt_length, so ConPO caps prompts itself).

    vllm 0.26.0's offline `SamplingParams` has no `truncate_prompt_tokens`
    field to do this at generation time (that field lives only on the
    OpenAI-server request/render objects, vllm/renderers/params.py and
    vllm/entrypoints/openai/*/protocol.py; verified against the installed
    package: `SamplingParams(truncate_prompt_tokens=...)` raises
    `TypeError: Unexpected keyword argument 'truncate_prompt_tokens'`), so
    both backends truncate here, before generation, instead.

    Returns the (possibly truncated) prompt text and whether it was truncated.
    """
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return prompt, False
    return tokenizer.decode(ids[-max_tokens:], skip_special_tokens=True), True


def truncate_prompts(tokenizer: Any, prompts: Sequence[str], max_tokens: int) -> tuple[list[str], int]:
    """Apply `left_truncate_prompt` across `prompts`; returns the (possibly
    truncated) prompts and how many of them were truncated."""
    out: list[str] = []
    truncated = 0
    for prompt in prompts:
        text, was_truncated = left_truncate_prompt(tokenizer, prompt, max_tokens)
        out.append(text)
        truncated += int(was_truncated)
    return out, truncated


def generate_greedy(engine: Any, prompts: Sequence[str], max_tokens: int) -> list[str]:
    """One greedy (temperature 0) completion per prompt, via `engine.generate`."""
    if not prompts:
        return []
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    completions = engine.generate(messages_batch, n=1, max_tokens=max_tokens, temperature=0.0, top_p=1.0, seed=0)
    return [c[0] for c in completions]


def _load_hf_engine(model_name: str, adapter: str | None) -> Any:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from conpo.engine import HFEngine

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HFEngine(model, tokenizer, device=device)


def _eval_vllm_gpu_memory_utilization() -> float | None:
    """EVAL_VLLM_GPU_MEMORY_UTILIZATION: unset or empty (default) leaves
    vllm.LLM's own gpu_memory_utilization default untouched (None, no
    argument passed to LLM(...), byte-identical to today); a float in
    (0, 1] overrides it, so an offline eval engine can be told to take a
    smaller share of the card and co-run beside a training job that
    already holds part of it. Any other value raises SystemExit before any
    engine is built, mirroring conpo/retention.py's retention_lenient()."""
    raw = config.env("EVAL_VLLM_GPU_MEMORY_UTILIZATION", "")
    if raw == "":
        return None
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"EVAL_VLLM_GPU_MEMORY_UTILIZATION must be a float in (0, 1], got {raw!r}") from None
    if not (0 < value <= 1):
        raise SystemExit(f"EVAL_VLLM_GPU_MEMORY_UTILIZATION must be a float in (0, 1], got {raw!r}")
    return value


@functools.lru_cache(maxsize=None)
def _build_vllm_engine(model_name: str, adapter: str | None) -> Any:
    """UNTESTED: no vllm / GPU available in this environment. Memoised per
    (model_name, adapter): vllm.LLM() reloads and shards the model on every
    construction, so a process scoring both IFEval and IFBench (two
    _generate_vllm calls, same model) must reuse one engine instead of
    building it twice."""
    gpu_memory_utilization = _eval_vllm_gpu_memory_utilization()

    from vllm import LLM

    # max_lora_rank: vLLM's default is 16 and refuses a rank-64 adapter at load
    # time ("LoRA rank 64 is greater than max_lora_rank 16", pipeline probe
    # 2026-09-11); the trained adapters are config.LORA_R.
    kwargs: dict[str, Any] = {}
    if gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    return LLM(model=model_name, enable_lora=adapter is not None, max_lora_rank=config.LORA_R, **kwargs)


def _generate_vllm(model_name: str, adapter: str | None, prompts: Sequence[str], max_tokens: int) -> list[str]:
    """UNTESTED: no vllm / GPU available in this environment. Standalone
    offline vllm.LLM (not the trainer-colocated engine in conpo/engine.py),
    since eval_external runs outside any GRPOTrainer."""
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=False,
            **config.POLICY_CHAT_TEMPLATE_KWARGS,
        )
        for p in prompts
    ]
    llm = _build_vllm_engine(model_name, adapter)
    lora_request = LoRARequest("adapter", 1, adapter) if adapter is not None else None
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(rendered, params, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]


def _load_ifeval_rows() -> list[dict[str, Any]]:
    ds = datasets.load_dataset(config.IFEVAL_DATASET, split="train")
    return [dict(row) for row in ds]


def _load_canned(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _is_ifeval_ids(instruction_id_list: Sequence[str]) -> bool:
    return all(iid in _IFEVAL_SET for iid in instruction_id_list)


def _clean_kwargs(kwargs_list: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop None-valued keys (conpo.verifiers.Constraint.from_ground_truth's
    convention). The raw HF google/IFEval and vendored IFBench_test rows
    store one union-schema kwargs dict per instruction (every field name
    across every family, irrelevant ones None); google's own per-instruction
    build_description signatures only accept their own field names, so an
    unfiltered dict raises TypeError on an unrecognised keyword."""
    return [{k: v for k, v in kw.items() if v is not None} for kw in kwargs_list]


def _merge(reference_row: dict[str, Any], response: str) -> dict[str, Any]:
    return {
        "key": reference_row["key"],
        "prompt": reference_row["prompt"],
        "response": response,
        "instruction_id_list": list(reference_row["instruction_id_list"]),
        "kwargs": _clean_kwargs(reference_row["kwargs"]),
    }


def join_by_prompt(canned: Sequence[dict[str, Any]], ifbench_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join minimal {"prompt", "response"} canned records to IFBench_test by
    exact prompt text, for canned files with no "key" (e.g. the vendored
    ifbench/data/sample_output.jsonl, whose 300 rows carry only prompt and
    response -- verified to match all 300 IFBench_test prompts).

    Raises KeyError if a canned prompt matches no IFBench_test row.
    """
    ifbench_by_prompt = {row["prompt"]: row for row in ifbench_rows}
    out: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for record in canned:
        ref = ifbench_by_prompt.get(record["prompt"])
        if ref is None:
            unmatched.append(record["prompt"])
            continue
        out.append(_merge(ref, record["response"]))
    if unmatched:
        raise KeyError(f"{len(unmatched)} canned record(s) matched no IFBench_test prompt: {unmatched}")
    return out


def route_canned(
    canned: Sequence[dict[str, Any]], ifeval_rows: Sequence[dict[str, Any]], ifbench_rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split canned records into (ifeval_records, ifbench_records).

    Minimal {key, response} records are joined by exact key equality, not a
    stringified key: google/IFEval's `key` is an int and IFBench_test's is a
    str, and small integers overlap between the two ("13" vs 13) -- coercing
    both to str before comparing would let one dataset's response wrongly
    match the other's row.

    Raises KeyError if a minimal record's key matches neither dataset, so a
    typo'd key or a canned file built against the wrong dataset fails loudly
    instead of silently shrinking the scored set.
    """
    ifeval_by_key = {row["key"]: row for row in ifeval_rows}
    ifbench_by_key = {row["key"]: row for row in ifbench_rows}
    ifeval_out: list[dict[str, Any]] = []
    ifbench_out: list[dict[str, Any]] = []
    unmatched: list[Any] = []
    for record in canned:
        if "instruction_id_list" in record:
            (ifeval_out if _is_ifeval_ids(record["instruction_id_list"]) else ifbench_out).append(record)
            continue
        ref = ifeval_by_key.get(record["key"])
        if ref is not None:
            ifeval_out.append(_merge(ref, record["response"]))
            continue
        ref = ifbench_by_key.get(record["key"])
        if ref is not None:
            ifbench_out.append(_merge(ref, record["response"]))
            continue
        unmatched.append(record["key"])
    if unmatched:
        raise KeyError(f"{len(unmatched)} canned record(s) matched neither dataset: {unmatched}")
    return ifeval_out, ifbench_out


def record_verdicts(records: Sequence[dict[str, Any]], benchmark: str) -> list[dict[str, Any]]:
    """One verdict dict per record: key, benchmark, prompt_strict, prompt_loose,
    instruction_id_list, instruction_strict, instruction_loose.

    Mirrors conpo.scoring._score's per-record InputExample/OutputExample
    construction (that function only returns the aggregate ScoreReport, not
    per-record verdicts) using the same vendored evaluation_lib it calls.
    """
    lib = _ifeval_lib if benchmark == "ifeval" else _ifbench_lib
    out = []
    for index, record in enumerate(records):
        inp = lib.InputExample(
            key=record.get("key", index),
            instruction_id_list=list(record["instruction_id_list"]),
            prompt=str(record["prompt"]),
            kwargs=[dict(kw) for kw in record["kwargs"]],
        )
        scored_response = str(record["response"])
        prompt_to_response = {str(record["prompt"]): scored_response}
        strict_out = lib.test_instruction_following_strict(inp, prompt_to_response)
        loose_out = lib.test_instruction_following_loose(inp, prompt_to_response)
        out.append(
            {
                "key": inp.key,
                "benchmark": benchmark,
                "prompt_strict": bool(strict_out.follow_all_instructions),
                "prompt_loose": bool(loose_out.follow_all_instructions),
                "instruction_id_list": list(inp.instruction_id_list),
                "instruction_strict": [bool(x) for x in strict_out.follow_instruction_list],
                "instruction_loose": [bool(x) for x in loose_out.follow_instruction_list],
                "response": scored_response,
            }
        )
    return out


def write_verdicts(out_path: str, verdicts: Sequence[dict[str, Any]]) -> str:
    """Write `verdicts` to `<out_path>.verdicts.jsonl`, one JSON object per line."""
    verdicts_path = f"{out_path}.verdicts.jsonl"
    with open(verdicts_path, "w", encoding="utf-8") as f:
        for v in verdicts:
            f.write(json.dumps(v) + "\n")
    return verdicts_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="External IFEval / IFBench eval for the ConPO policy.")
    parser.add_argument("--model", default=config.POLICY_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory, or omit for the base model")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument(
        "--score-only",
        default=None,
        help=(
            "jsonl of canned responses; skips generation. A minimal record is "
            '{"key": ..., "response": ...}, joined by exact key equality against the loaded '
            "IFEval / IFBench_test row. Canned files with no key (e.g. the vendored "
            "sample_output.jsonl) need --join-by-prompt instead."
        ),
    )
    parser.add_argument(
        "--join-by-prompt",
        action="store_true",
        help=(
            "join --score-only records to IFBench_test by exact prompt text instead of key "
            "(for canned files like the vendored sample_output.jsonl, which carry only prompt "
            "and response, no key)"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="cap rows per dataset, for smoke runs")
    parser.add_argument("--max-tokens", type=int, default=config.EVAL_MAX_NEW_TOKENS)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    ifeval_rows = _load_ifeval_rows()
    ifbench_rows = scoring.load_ifbench_test()
    if args.limit is not None:
        ifeval_rows = ifeval_rows[: args.limit]
        ifbench_rows = ifbench_rows[: args.limit]

    prompt_truncation: dict[str, Any] | None = None
    if args.score_only is not None:
        canned = _load_canned(args.score_only)
        if args.join_by_prompt:
            ifeval_records = []
            ifbench_records = join_by_prompt(canned, ifbench_rows)
        else:
            ifeval_records, ifbench_records = route_canned(canned, ifeval_rows, ifbench_rows)
    else:
        if args.backend == "hf":
            engine = _load_hf_engine(args.model, args.adapter)
            tokenizer = engine.tokenizer
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model)
        ifeval_prompts, ifeval_truncated = truncate_prompts(
            tokenizer, [r["prompt"] for r in ifeval_rows], config.MAX_PROMPT_TOKENS
        )
        ifbench_prompts, ifbench_truncated = truncate_prompts(
            tokenizer, [r["prompt"] for r in ifbench_rows], config.MAX_PROMPT_TOKENS
        )
        if args.backend == "hf":
            ifeval_responses = generate_greedy(engine, ifeval_prompts, args.max_tokens)
            ifbench_responses = generate_greedy(engine, ifbench_prompts, args.max_tokens)
        else:
            ifeval_responses = _generate_vllm(args.model, args.adapter, ifeval_prompts, args.max_tokens)
            ifbench_responses = _generate_vllm(args.model, args.adapter, ifbench_prompts, args.max_tokens)
        ifeval_records = [_merge(row, response) for row, response in zip(ifeval_rows, ifeval_responses)]
        ifbench_records = [_merge(row, response) for row, response in zip(ifbench_rows, ifbench_responses)]
        prompt_truncation = {
            "max_prompt_tokens": config.MAX_PROMPT_TOKENS,
            "side": "left",
            "truncated_counts": {"ifeval": ifeval_truncated, "ifbench": ifbench_truncated},
        }

    ifeval_score = dataclasses.asdict(scoring.score_ifeval(ifeval_records)) if ifeval_records else None
    ifbench_score = dataclasses.asdict(scoring.score_ifbench(ifbench_records)) if ifbench_records else None

    out = {
        "ifeval": ifeval_score,
        "ifbench": ifbench_score,
        "decoding": {"temperature": 0, "max_tokens": args.max_tokens},
        "model": args.model,
        "adapter": args.adapter,
        "prompt_truncation": prompt_truncation,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

    verdicts = record_verdicts(ifeval_records, "ifeval") + record_verdicts(ifbench_records, "ifbench")
    verdicts_path = write_verdicts(args.out, verdicts)
    print(f"wrote {verdicts_path}")


if __name__ == "__main__":
    main()

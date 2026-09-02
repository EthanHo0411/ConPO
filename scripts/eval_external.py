"""External eval: IFEval (google/IFEval) and IFBench (IFBench_test), greedy decoding.

--score-only skips generation entirely and scores canned responses instead.
A canned record is either a full scoring record (already carrying
`instruction_id_list` and `kwargs`, e.g. hand-written for a test) or a
minimal {"key", "response"} pair joined by key against the loaded reference
row of whichever dataset (IFEval / IFBench) has a matching key.
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import datasets

from conpo import config, scoring
from conpo.verifiers import IFEVAL_FAMILIES

_IFEVAL_SET = set(IFEVAL_FAMILIES)


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


def _generate_vllm(model_name: str, adapter: str | None, prompts: Sequence[str], max_tokens: int) -> list[str]:
    """UNTESTED: no vllm / GPU available in this environment. Standalone
    offline vllm.LLM (not the trainer-colocated engine in conpo/engine.py),
    since eval_external runs outside any GRPOTrainer."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
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
    llm = LLM(model=model_name, enable_lora=adapter is not None)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="External IFEval / IFBench eval for the ConPO policy.")
    parser.add_argument("--model", default=config.POLICY_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory, or omit for the base model")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--score-only", default=None, help="jsonl of canned responses; skips generation")
    parser.add_argument("--limit", type=int, default=None, help="cap rows per dataset, for smoke runs")
    parser.add_argument("--max-tokens", type=int, default=config.MAX_COMPLETION_TOKENS)
    args = parser.parse_args()

    ifeval_rows = _load_ifeval_rows()
    ifbench_rows = scoring.load_ifbench_test()
    if args.limit is not None:
        ifeval_rows = ifeval_rows[: args.limit]
        ifbench_rows = ifbench_rows[: args.limit]

    if args.score_only is not None:
        canned = _load_canned(args.score_only)
        ifeval_records, ifbench_records = route_canned(canned, ifeval_rows, ifbench_rows)
    else:
        if args.backend == "hf":
            engine = _load_hf_engine(args.model, args.adapter)
            ifeval_responses = generate_greedy(engine, [r["prompt"] for r in ifeval_rows], args.max_tokens)
            ifbench_responses = generate_greedy(engine, [r["prompt"] for r in ifbench_rows], args.max_tokens)
        else:
            ifeval_responses = _generate_vllm(
                args.model, args.adapter, [r["prompt"] for r in ifeval_rows], args.max_tokens
            )
            ifbench_responses = _generate_vllm(
                args.model, args.adapter, [r["prompt"] for r in ifbench_rows], args.max_tokens
            )
        ifeval_records = [_merge(row, response) for row, response in zip(ifeval_rows, ifeval_responses)]
        ifbench_records = [_merge(row, response) for row, response in zip(ifbench_rows, ifbench_responses)]

    ifeval_score = dataclasses.asdict(scoring.score_ifeval(ifeval_records)) if ifeval_records else None
    ifbench_score = dataclasses.asdict(scoring.score_ifbench(ifbench_records)) if ifbench_records else None

    out = {
        "ifeval": ifeval_score,
        "ifbench": ifbench_score,
        "decoding": {"temperature": 0, "max_tokens": args.max_tokens},
        "model": args.model,
        "adapter": args.adapter,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

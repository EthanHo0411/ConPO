"""Internal eval: conpo.metrics.summarise over data/{split}.jsonl PoolRow splits.

--score-only skips generation and scores canned responses instead: a jsonl
of {"key", "response"} records, joined by row.key. See scripts/build_corpus.py
for the PoolRow JSON schema this reads (`_row_from_json` below is its inverse).

Alongside `--out`'s JSON, `<out>.verdicts.jsonl` carries one line per scored
prompt (key, split, prompt_strict, prompt_loose, per-instruction
strict/loose lists) -- conpo.verifiers has no strict/loose distinction (that
is an IFEval/IFBench evaluation_lib concept), so both columns here are the
same single verifier_vector verdict, kept for schema parity with
scripts/eval_external.py's verdicts.jsonl so scripts/analyze.py can read
both the same way.
"""

import argparse
import functools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from conpo import config, metrics
from conpo.data import PoolRow
from conpo.verifiers import Constraint, verifier_vector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def left_truncate_prompt(tokenizer: Any, prompt: str, max_tokens: int) -> tuple[str, bool]:
    """Left-truncate `prompt`'s tokenisation (policy tokenizer) to its last
    `max_tokens` tokens, mirroring TRL GRPOConfig.max_prompt_length's left
    truncation that training applies via config.MAX_PROMPT_TOKENS.

    Duplicated from scripts/eval_external.py (see that copy's docstring for
    why vllm 0.26.0's offline SamplingParams cannot do this at generation
    time instead): both scripts run standalone (no shared package).

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
    """One greedy (temperature 0) completion per prompt, via `engine.generate`.

    Duplicated from scripts/eval_external.py: both scripts run standalone
    (no shared package), and the helper is a few lines.
    """
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


@functools.lru_cache(maxsize=None)
def _build_vllm_engine(model_name: str, adapter: str | None) -> Any:
    """UNTESTED: no vllm / GPU available in this environment. Memoised per
    (model_name, adapter): vllm.LLM() reloads and shards the model on every
    construction, so a process scoring multiple splits (multiple
    _generate_vllm calls, same model) must reuse one engine instead of
    building it per split."""
    from vllm import LLM

    return LLM(model=model_name, enable_lora=adapter is not None)


def _generate_vllm(model_name: str, adapter: str | None, prompts: Sequence[str], max_tokens: int) -> list[str]:
    """UNTESTED: no vllm / GPU available in this environment."""
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


def _row_from_json(obj: dict[str, Any]) -> PoolRow:
    """Inverse of scripts/build_corpus.py's `_row_to_json`."""
    return PoolRow(
        key=obj["key"],
        base_prompt=obj["base_prompt"],
        prompt=obj["prompt"],
        sentences=tuple(obj["sentences"]),
        constraints=tuple(Constraint(c["iid"], c["kwargs"]) for c in obj["constraints"]),
        english=obj["english"],
        num_constraints=obj["num_constraints"],
    )


def load_split(name: str, limit: int | None = None) -> list[PoolRow]:
    """Rows from data/{name}.jsonl, capped to `limit` (a prefix slice) if given."""
    rows: list[PoolRow] = []
    with (DATA_DIR / f"{name}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(_row_from_json(json.loads(line)))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _load_canned_responses(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                out[str(record["key"])] = record["response"]
    return out


def record_verdicts(rows: Sequence[PoolRow], responses: Sequence[str], split_name: str) -> list[dict[str, Any]]:
    """One verdict dict per row: key, split, prompt_strict, prompt_loose (both
    the same conpo.verifiers.feasible verdict), instruction_id_list,
    instruction_strict, instruction_loose (both the same verifier_vector)."""
    out = []
    for row, response in zip(rows, responses):
        vector = [bool(v) for v in verifier_vector(row.constraints, response)]
        passed = all(vector)
        out.append(
            {
                "key": row.key,
                "split": split_name,
                "prompt_strict": passed,
                "prompt_loose": passed,
                "instruction_id_list": [c.iid for c in row.constraints],
                "instruction_strict": vector,
                "instruction_loose": vector,
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
    parser = argparse.ArgumentParser(description="Internal ConPO eval: metrics.summarise over data/ splits.")
    parser.add_argument("--splits", default="dev,eval_i,eval_ii,eval_iii")
    parser.add_argument("--model", default=config.POLICY_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory, or omit for the base model")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--score-only", default=None, help="jsonl of {key, response}; skips generation")
    parser.add_argument("--limit", type=int, default=None, help="cap rows per split, for smoke runs")
    parser.add_argument("--max-tokens", type=int, default=config.EVAL_MAX_NEW_TOKENS)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    splits = {name: load_split(name, args.limit) for name in split_names}

    engine = None
    if args.score_only is None and args.backend == "hf":
        engine = _load_hf_engine(args.model, args.adapter)
    canned = _load_canned_responses(args.score_only) if args.score_only is not None else None

    tokenizer = None
    if args.score_only is None:
        if engine is not None:
            tokenizer = engine.tokenizer
        else:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(args.model)

    results: dict[str, Any] = {}
    verdicts: list[dict[str, Any]] = []
    truncated_counts: dict[str, int] = {}
    for name, rows in splits.items():
        prompts = [row.prompt for row in rows]
        if canned is not None:
            missing = [row.key for row in rows if str(row.key) not in canned]
            if missing:
                raise KeyError(f"no canned response for keys {missing} in {args.score_only}")
            responses = [canned[str(row.key)] for row in rows]
        else:
            prompts, truncated_counts[name] = truncate_prompts(tokenizer, prompts, config.MAX_PROMPT_TOKENS)
            try:
                if engine is not None:
                    responses = generate_greedy(engine, prompts, args.max_tokens)
                else:
                    responses = _generate_vllm(args.model, args.adapter, prompts, args.max_tokens)
            except Exception:
                print(f"generation failed on split={name!r}, keys={[row.key for row in rows]}")
                raise
        summary = metrics.summarise(rows, responses, name)
        if name in truncated_counts:
            # Nested per split, not a top-level "prompt_truncation" sibling:
            # scripts/analyze.py's internal_table() iterates every top-level
            # key of this JSON as a split name and reads summary["pacc"]
            # (design choice, builder, 2026-09-11).
            summary["prompt_truncation"] = {
                "max_prompt_tokens": config.MAX_PROMPT_TOKENS,
                "side": "left",
                "truncated_count": truncated_counts[name],
            }
        results[name] = summary
        verdicts.extend(record_verdicts(rows, responses, name))

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

    verdicts_path = write_verdicts(args.out, verdicts)
    print(f"wrote {verdicts_path}")


if __name__ == "__main__":
    main()

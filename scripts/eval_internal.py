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


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal ConPO eval: metrics.summarise over data/ splits.")
    parser.add_argument("--splits", default="dev,eval_i,eval_ii,eval_iii")
    parser.add_argument("--model", default=config.POLICY_MODEL)
    parser.add_argument("--adapter", default=None, help="LoRA adapter directory, or omit for the base model")
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument("--score-only", default=None, help="jsonl of {key, response}; skips generation")
    parser.add_argument("--limit", type=int, default=None, help="cap rows per split, for smoke runs")
    parser.add_argument("--max-tokens", type=int, default=config.MAX_COMPLETION_TOKENS)
    args = parser.parse_args()

    split_names = [s.strip() for s in args.splits.split(",") if s.strip()]
    splits = {name: load_split(name, args.limit) for name in split_names}

    engine = None
    if args.score_only is None and args.backend == "hf":
        engine = _load_hf_engine(args.model, args.adapter)
    canned = _load_canned_responses(args.score_only) if args.score_only is not None else None

    results: dict[str, Any] = {}
    verdicts: list[dict[str, Any]] = []
    for name, rows in splits.items():
        prompts = [row.prompt for row in rows]
        if canned is not None:
            missing = [row.key for row in rows if str(row.key) not in canned]
            if missing:
                raise KeyError(f"no canned response for keys {missing} in {args.score_only}")
            responses = [canned[str(row.key)] for row in rows]
        elif engine is not None:
            responses = generate_greedy(engine, prompts, args.max_tokens)
        else:
            responses = _generate_vllm(args.model, args.adapter, prompts, args.max_tokens)
        results[name] = metrics.summarise(rows, responses, name)
        verdicts.extend(record_verdicts(rows, responses, name))

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")

    verdicts_path = write_verdicts(args.out, verdicts)
    print(f"wrote {verdicts_path}")


if __name__ == "__main__":
    main()

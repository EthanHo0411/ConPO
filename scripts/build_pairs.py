"""Build DPO baseline pairs (vf_dpo | relax_dpo | edge_dpo). See conpo/pairs.py."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from conpo import config, data as data_module
from conpo.data import PoolRow, TokenMeter
from conpo.engine import Engine, HFEngine
from conpo.pairs import Pair, build_edge_pairs, build_relax_pairs, build_vf_pairs, save_pairs
from conpo.verifiers import Constraint

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _row_from_json(data: dict) -> PoolRow:
    """Inverse of scripts/build_corpus.py's `_row_to_json`."""
    return PoolRow(
        key=data["key"],
        base_prompt=data["base_prompt"],
        prompt=data["prompt"],
        sentences=tuple(data["sentences"]),
        constraints=tuple(Constraint(c["iid"], c["kwargs"]) for c in data["constraints"]),
        english=data["english"],
        num_constraints=data["num_constraints"],
    )


def _load_rows(path: Path, limit: int | None) -> list[PoolRow]:
    rows: list[PoolRow] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            rows.append(_row_from_json(json.loads(line)))
    return rows


def _apply_subset(rows: list[PoolRow], subset_path: str) -> list[PoolRow]:
    """Filter/reorder `rows` to the committed subset key list at `subset_path`
    (one key per line -- conpo.data.load_subset's format, written by
    scripts/draw_subset.py). Uses conpo.data.load_subset when present;
    otherwise mirrors its contract directly (fail loud on a subset key
    missing from `rows`).
    """
    load_subset = getattr(data_module, "load_subset", None)
    if load_subset is not None:
        return load_subset(rows, subset_path)
    by_key = {r.key: r for r in rows}
    keys = [line.strip() for line in Path(subset_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [k for k in keys if k not in by_key]
    if missing:
        shown = missing[:5]
        suffix = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise KeyError(f"subset key(s) not found in rows: {shown}{suffix}")
    return [by_key[k] for k in keys]


def _build_engine(kind: str, model_name: str) -> Engine:
    if kind == "hf":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return HFEngine(model, tokenizer, device=device)
    # UNTESTED: no vllm / GPU available in this environment.
    return _VLLMStandaloneEngine(model_name)


class _VLLMStandaloneEngine:
    """Offline vllm.LLM engine implementing the Engine protocol, for pair
    building outside any GRPOTrainer (conpo.engine.VLLMColocateEngine wraps a
    live trainer's colocated generation instead, not usable here).

    UNTESTED: no vllm / GPU available in this environment.
    """

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._llm = LLM(model=model_name)

    def generate(self, messages_batch, n, max_tokens, temperature, top_p, seed):  # noqa: ANN001
        from vllm import SamplingParams

        if not messages_batch:
            return []
        rendered = [
            self._tokenizer.apply_chat_template(
                list(messages), add_generation_prompt=True, tokenize=False, **config.POLICY_CHAT_TEMPLATE_KWARGS
            )
            for messages in messages_batch
        ]
        params = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
        outputs = self._llm.generate(rendered, params)
        return [[o.text for o in out.outputs] for out in outputs]

    def count_tokens(self, texts) -> int:  # noqa: ANN001
        if not texts:
            return 0
        encoded = self._tokenizer(list(texts), add_special_tokens=False)["input_ids"]
        return sum(len(ids) for ids in encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO baseline pairs (vf | relax | edges).")
    parser.add_argument("--mode", required=True, choices=("vf", "relax", "edges"))
    parser.add_argument("--rows", default=str(DATA_DIR / "train.jsonl"))
    parser.add_argument("--subset", default=None, help="path to a committed key list; filters --rows to those keys")
    parser.add_argument("--engine", choices=("hf", "vllm"), default="hf", help="vllm is lazy-imported and UNTESTED")
    parser.add_argument("--model", default=config.POLICY_MODEL)
    parser.add_argument("--edges", default=None, help="edges.jsonl path, required when --mode edges")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None, help="cap rows considered, for smoke runs")
    parser.add_argument("--n-samples", type=int, default=config.NUM_GENERATIONS, help="vf/relax: responses per prompt")
    parser.add_argument("--pairs-per-prompt", type=int, default=1, help="vf mode only: pairs per admitted prompt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    meter = TokenMeter(budget=0)
    rows_considered: int | None = None
    pairs: list[Pair]

    if args.mode == "edges":
        if args.edges is None:
            parser.error("--mode edges requires --edges")
        pairs = build_edge_pairs(args.edges)
    else:
        rows = _load_rows(Path(args.rows), args.limit)
        if args.subset is not None:
            rows = _apply_subset(rows, args.subset)
        rows_considered = len(rows)
        engine = _build_engine(args.engine, args.model)
        if args.mode == "vf":
            pairs = build_vf_pairs(rows, engine, args.n_samples, rng, meter, pairs_per_prompt=args.pairs_per_prompt)
        else:
            pairs = build_relax_pairs(rows, engine, args.n_samples, rng, meter)

    save_pairs(pairs, args.out)

    manifest = {
        "mode": args.mode,
        "count": len(pairs),
        "rows_considered": rows_considered,
        "yield_per_prompt": (len(pairs) / rows_considered) if rows_considered else None,
        "tokens_spent": meter.total,
        "engine": args.engine if args.mode != "edges" else None,
        "model": args.model if args.mode != "edges" else None,
        "out": args.out,
    }
    manifest_path = Path(args.out).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(pairs)} pairs -> {args.out}, manifest -> {manifest_path}")


if __name__ == "__main__":
    main()

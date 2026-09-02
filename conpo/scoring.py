"""Scoring wrappers around the vendored IFEval and IFBench evaluators.

Every check runs through the vendored `evaluation_lib.test_instruction_following_strict`
/ `_loose` functions in `conpo.vendor.ifeval` and `conpo.vendor.ifbench`
verbatim; nothing here reimplements an instruction check.

A record is a mapping with the fields the vendored evaluators expect
(same field names as the upstream IFEval / IFBench JSONL rows):
  - "key": optional, any identifier. Falls back to the record's position
    in the input sequence when absent.
  - "prompt": str. Used both as the example's prompt and as the join key
    to find that example's response (the vendored evaluators look up
    responses by exact prompt text).
  - "response": str. The model completion to score.
  - "instruction_id_list": list[str]. Keys into the vendored registry's
    INSTRUCTION_DICT.
  - "kwargs": list[dict]. One kwargs dict per entry in
    instruction_id_list, passed to that instruction's build_description.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conpo.vendor.ifbench import evaluation_lib as _ifbench_lib
from conpo.vendor.ifeval import evaluation_lib as _ifeval_lib

_IFBENCH_DATA_DIR = Path(__file__).parent / "vendor" / "ifbench" / "data"


@dataclass(frozen=True)
class ScoreReport:
    """Aggregate accuracy for one scoring run.

    prompt_strict / prompt_loose: fraction of records whose every
    instruction passed, under the strict / loose response variants.
    instruction_strict / instruction_loose: fraction of individual
    (record, instruction) checks that passed.
    num_prompts: number of records scored.
    num_instructions: sum of len(instruction_id_list) over all records.
    per_instruction: loose per-instruction-id accuracy (instruction id ->
    fraction of that id's checks that passed under the loose variant).
    """

    prompt_strict: float
    prompt_loose: float
    instruction_strict: float
    instruction_loose: float
    num_prompts: int
    num_instructions: int
    per_instruction: Mapping[str, float]


def score_ifeval(records: Sequence[Mapping[str, Any]]) -> ScoreReport:
    """Score IFEval records with the vendored google-research evaluator."""
    return _score(records, _ifeval_lib)


def score_ifbench(records: Sequence[Mapping[str, Any]]) -> ScoreReport:
    """Score IFBench records with the vendored allenai/IFBench evaluator."""
    return _score(records, _ifbench_lib)


def load_ifbench_test() -> list[dict[str, Any]]:
    """Load the vendored IFBench_test.jsonl (300 prompts)."""
    return _load_jsonl(_IFBENCH_DATA_DIR / "IFBench_test.jsonl")


def load_sample_output() -> list[dict[str, Any]]:
    """Load the vendored sample_output.jsonl (300 prompt/response pairs)."""
    return _load_jsonl(_IFBENCH_DATA_DIR / "sample_output.jsonl")


def _score(records: Sequence[Mapping[str, Any]], lib: Any) -> ScoreReport:
    inputs = [
        lib.InputExample(
            key=record.get("key", index),
            instruction_id_list=list(record["instruction_id_list"]),
            prompt=str(record["prompt"]),
            kwargs=[dict(kw) for kw in record["kwargs"]],
        )
        for index, record in enumerate(records)
    ]
    prompt_to_response = {str(record["prompt"]): str(record["response"]) for record in records}

    strict_outputs = [lib.test_instruction_following_strict(inp, prompt_to_response) for inp in inputs]
    loose_outputs = [lib.test_instruction_following_loose(inp, prompt_to_response) for inp in inputs]

    num_prompts = len(records)
    num_instructions = sum(len(o.follow_instruction_list) for o in strict_outputs)

    per_instruction_total: dict[str, int] = {}
    per_instruction_correct: dict[str, int] = {}
    for output in loose_outputs:
        for instruction_id, passed in zip(output.instruction_id_list, output.follow_instruction_list):
            per_instruction_total[instruction_id] = per_instruction_total.get(instruction_id, 0) + 1
            per_instruction_correct[instruction_id] = per_instruction_correct.get(instruction_id, 0) + int(passed)

    return ScoreReport(
        prompt_strict=_mean(o.follow_all_instructions for o in strict_outputs),
        prompt_loose=_mean(o.follow_all_instructions for o in loose_outputs),
        instruction_strict=sum(sum(o.follow_instruction_list) for o in strict_outputs) / num_instructions,
        instruction_loose=sum(sum(o.follow_instruction_list) for o in loose_outputs) / num_instructions,
        num_prompts=num_prompts,
        num_instructions=num_instructions,
        per_instruction={iid: per_instruction_correct[iid] / total for iid, total in per_instruction_total.items()},
    )


def _mean(flags: Any) -> float:
    values = list(flags)
    return sum(values) / len(values)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

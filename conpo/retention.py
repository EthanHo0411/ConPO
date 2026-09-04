"""Retention suite: zero-shot GSM8K, MMLU and BBH rows, prompts and scorers.

Pure functions, no model code. `rows_from_gsm8k` / `rows_from_mmlu` /
`rows_from_bbh` turn HF-shaped rows (datasets.load_dataset dicts, see
conpo.config.GSM8K_DATASET / MMLU_DATASET / BBH_DATASET) into RetentionRows;
`score` dispatches a generated response to the right per-benchmark scorer.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_LETTER_RE = re.compile(r"\b([A-D])\b")
_WS_RE = re.compile(r"\s+")
_PAREN_LETTER_RE = re.compile(r"^\([A-Za-z]\)$")


@dataclass(frozen=True)
class RetentionRow:
    key: str
    benchmark: str
    prompt: str
    target: str
    group: str


def rows_from_gsm8k(hf_rows: Sequence[Mapping[str, Any]]) -> list[RetentionRow]:
    """GSM8K test rows (fields "question", "answer") to RetentionRows.

    Target is the text after the row's own last "####", commas stripped
    (GSM8K's "answer" field always carries one, so this never falls back).
    """
    rows = []
    for index, row in enumerate(hf_rows):
        question = row["question"]
        target = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
        prompt = (
            f"{question}\n\nSolve step by step, then write the final numeric answer "
            "on its own last line as: #### <number>"
        )
        rows.append(RetentionRow(key=f"gsm8k:{index}", benchmark="gsm8k", prompt=prompt, target=target, group=""))
    return rows


def rows_from_mmlu(hf_rows: Sequence[Mapping[str, Any]]) -> list[RetentionRow]:
    """MMLU test rows (fields "question", "choices", "answer", "subject") to
    RetentionRows. group is the MMLU subject; target is the answer letter."""
    rows = []
    for index, row in enumerate(hf_rows):
        choices = row["choices"]
        prompt = (
            f"{row['question']}\n\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\n"
            "Answer with the letter of the correct choice only."
        )
        target = "ABCD"[row["answer"]]
        rows.append(
            RetentionRow(key=f"mmlu:{index}", benchmark="mmlu", prompt=prompt, target=target, group=row["subject"])
        )
    return rows


def rows_from_bbh(task: str, hf_rows: Sequence[Mapping[str, Any]]) -> list[RetentionRow]:
    """BBH test rows (fields "input", "target") for one task config to
    RetentionRows. group is the BBH task name."""
    rows = []
    for index, row in enumerate(hf_rows):
        prompt = f"{row['input']}\n\nGive only the final answer on the last line."
        rows.append(
            RetentionRow(
                key=f"bbh:{task}:{index}", benchmark="bbh", prompt=prompt, target=row["target"], group=task
            )
        )
    return rows


def score_gsm8k(response: str, target: str) -> bool:
    """Last number in `response`, preferring the one after the last "####",
    commas stripped, compared to `target` (already comma-stripped) with
    tolerance 1e-6."""
    had_marker = "####" in response
    tail = response.rsplit("####", 1)[1] if had_marker else response
    matches = list(_NUMBER_RE.finditer(tail))
    if not matches and had_marker:
        matches = list(_NUMBER_RE.finditer(response))
    if not matches:
        return False
    predicted = matches[-1].group(0).replace(",", "")
    try:
        return abs(float(predicted) - float(target.replace(",", ""))) <= 1e-6
    except ValueError:
        return False


def score_mmlu(response: str, target: str) -> bool:
    """First standalone A-D letter in `response` (matches "B", "(B)" and
    "A." forms via a word-boundary regex) compared to `target`."""
    match = _LETTER_RE.search(response)
    return match is not None and match.group(1) == target


def _normalise_bbh_line(text: str) -> str:
    text = text.strip().lower()
    if text.endswith("."):
        text = text[:-1]
    text = _WS_RE.sub(" ", text).strip()
    return text.strip("'\"")


def score_bbh(response: str, target: str) -> bool:
    """Final non-empty line of `response`, normalised (strip, lowercase,
    drop a trailing full stop, collapse whitespace, strip surrounding
    quotes), compared to the same-normalised `target`. When `target` has
    the form "(X)", a normalised line that merely contains "(x)" also
    counts (BBH's multiple-choice tasks sometimes get "(a)" back out of a
    fuller sentence)."""
    non_empty = [line for line in response.split("\n") if line.strip()]
    line = _normalise_bbh_line(non_empty[-1]) if non_empty else ""
    norm_target = _normalise_bbh_line(target)
    if line == norm_target:
        return True
    if _PAREN_LETTER_RE.match(target.strip()):
        return norm_target in line
    return False


def score(row: RetentionRow, response: str) -> bool:
    """Dispatch to the scorer matching `row.benchmark`."""
    if row.benchmark == "gsm8k":
        return score_gsm8k(response, row.target)
    if row.benchmark == "mmlu":
        return score_mmlu(response, row.target)
    if row.benchmark == "bbh":
        return score_bbh(response, row.target)
    raise ValueError(f"unknown benchmark {row.benchmark!r}")

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


def extract_gsm8k_prediction(response: str) -> str | None:
    """Last number in `response`, preferring the one after the last "####",
    commas stripped. None if no number is found."""
    had_marker = "####" in response
    tail = response.rsplit("####", 1)[1] if had_marker else response
    matches = list(_NUMBER_RE.finditer(tail))
    if not matches and had_marker:
        matches = list(_NUMBER_RE.finditer(response))
    if not matches:
        return None
    return matches[-1].group(0).replace(",", "")


def score_gsm8k(response: str, target: str) -> bool:
    """`extract_gsm8k_prediction(response)` compared to `target` (already
    comma-stripped) with tolerance 1e-6."""
    predicted = extract_gsm8k_prediction(response)
    if predicted is None:
        return False
    try:
        return abs(float(predicted) - float(target.replace(",", ""))) <= 1e-6
    except ValueError:
        return False


def extract_mmlu_prediction(response: str) -> str | None:
    """First standalone A-D letter in `response` (matches "B", "(B)" and
    "A." forms via a word-boundary regex). None if no letter is found."""
    match = _LETTER_RE.search(response)
    return match.group(1) if match else None


def score_mmlu(response: str, target: str) -> bool:
    """`extract_mmlu_prediction(response)` compared to `target`."""
    return extract_mmlu_prediction(response) == target


def _normalise_bbh_line(text: str) -> str:
    text = text.strip().lower()
    if text.endswith("."):
        text = text[:-1]
    text = _WS_RE.sub(" ", text).strip()
    return text.strip("'\"")


def extract_bbh_prediction(response: str) -> str | None:
    """Final non-empty line of `response`, normalised (strip, lowercase,
    drop a trailing full stop, collapse whitespace, strip surrounding
    quotes). None if `response` has no non-empty line."""
    non_empty = [line for line in response.split("\n") if line.strip()]
    return _normalise_bbh_line(non_empty[-1]) if non_empty else None


# Design proposal -- Claude (subagent), 2026-09-11, Task F point 2.
# Not applied to the scoring rule below; verify against real Qwen3-4B
# responses (now captured per-row by record_verdicts) before changing it.
#
# Diagnosis of the base-model BBH zeros (dyck_languages 0.0, web_of_lies 0.0,
# word_sorting 0.068, multistep_arithmetic_two 0.116,
# logical_deduction_seven_objects 0.072):
#
# The scorer expects the LAST non-empty line, after normalisation, to be
# EXACTLY equal to the normalised target -- except when target is "(X)",
# where substring containment is also accepted (norm_target in line).
# The prompt only says "Give only the final answer on the last line"; it
# does not tell the model to give a BARE answer. Qwen3-4B non-thinking, even
# in its final line, tends to keep prose around the answer:
#   - "The answer is No." / "Answer: No" for web_of_lies -- exact-match
#     fails because of the prefix, even though normalisation already
#     strips the trailing period.
#   - "The final answer is -16." for multistep_arithmetic_two -- exact
#     match against a bare number fails the same way; GSM8K's scorer
#     tolerates this by extracting the last number, BBH's does not.
#   - "apple, banana, cherry" for word_sorting -- gold is space-separated
#     ("apple banana cherry"); normalisation collapses whitespace but does
#     not strip commas, so comma-joined output never matches.
#   - dyck_languages closing sequences ("] > )") likely fail the same
#     prefix/spacing way, plus the model may join brackets without spaces.
#   - logical_deduction_seven_objects (target "(A)"): the paren-containment
#     rule only checks norm_target-in-line, i.e. it accepts "(a)" inside a
#     longer line, but a bare "a" (letter without parens, which chat models
#     commonly emit despite the "(X)" gold format) is NOT accepted --
#     containment is one-directional. Markdown emphasis ("**(A)**") already
#     passes by accident (substring check), but a bare letter does not.
#
# Proposed normalisation (none applied yet):
#   1. Strip a leading "answer:", "final answer:", "the answer is" (or
#      similar, case-insensitive) prefix from the line before comparing.
#   2. Strip markdown emphasis markers ("*", "_", "`") in addition to the
#      quote-stripping already done, so "**No**" normalises to "no".
#   3. For a purely numeric target (multistep_arithmetic_two), reuse
#      extract_gsm8k_prediction's last-number extraction instead of exact
#      string equality.
#   4. For word-list targets (word_sorting), also strip commas before
#      collapsing whitespace, so "apple, banana, cherry" and
#      "apple banana cherry" normalise the same.
#   5. For a "(X)" target, also accept a bare letter/word match (strip the
#      target's own parens before the containment check) so "a" matches
#      gold "(A)", not only "(a)".
# Each of these narrows a real gap without loosening exact match elsewhere;
# they should be checked against the actual captured `response` field for
# these five groups once a GPU smoke run exists, then landed as a single
# scoring-rule change with updated tests.
def score_bbh(response: str, target: str) -> bool:
    """Final non-empty line of `response`, normalised (strip, lowercase,
    drop a trailing full stop, collapse whitespace, strip surrounding
    quotes), compared to the same-normalised `target`. When `target` has
    the form "(X)", a normalised line that merely contains "(x)" also
    counts (BBH's multiple-choice tasks sometimes get "(a)" back out of a
    fuller sentence)."""
    line = extract_bbh_prediction(response) or ""
    norm_target = _normalise_bbh_line(target)
    if line == norm_target:
        return True
    if _PAREN_LETTER_RE.match(target.strip()):
        return norm_target in line
    return False


def extract_prediction(row: RetentionRow, response: str) -> str | None:
    """Dispatch to the extractor matching `row.benchmark`; the scorer's
    best-effort parse of `response`, None if extraction found nothing."""
    if row.benchmark == "gsm8k":
        return extract_gsm8k_prediction(response)
    if row.benchmark == "mmlu":
        return extract_mmlu_prediction(response)
    if row.benchmark == "bbh":
        return extract_bbh_prediction(response)
    raise ValueError(f"unknown benchmark {row.benchmark!r}")


def score(row: RetentionRow, response: str) -> bool:
    """Dispatch to the scorer matching `row.benchmark`."""
    if row.benchmark == "gsm8k":
        return score_gsm8k(response, row.target)
    if row.benchmark == "mmlu":
        return score_mmlu(response, row.target)
    if row.benchmark == "bbh":
        return score_bbh(response, row.target)
    raise ValueError(f"unknown benchmark {row.benchmark!r}")

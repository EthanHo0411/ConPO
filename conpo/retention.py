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


_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_MARKER_RE = re.compile(
    r"(?im)^[ \t]*#{0,6}[ \t]*\*{0,2}[ \t]*(?:final\s+answer|answer)[ \t]*\*{0,2}[ \t]*:[ \t]*\*{0,2}[ \t]*(.*)$"
)
_MARKDOWN_CHARS_RE = re.compile(r"[*_`$]")


def _clean_marker_text(text: str) -> str:
    """Strip markdown emphasis/code-fence characters and `$` math
    delimiters, then run the shared BBH line normalisation."""
    return _normalise_bbh_line(_MARKDOWN_CHARS_RE.sub("", text))


def _extract_bbh_marker_prediction(response: str) -> str | None:
    """Design choice -- Task F point 2, diagnosed against the real
    Qwen3-4B responses `record_verdicts` captured for the base-model BBH
    zeros (dyck_languages, web_of_lies, word_sorting,
    multistep_arithmetic_two, logical_deduction_seven_objects). Qwen3-4B
    commonly states its committed answer as the last `\\boxed{...}` or
    behind a "Final Answer:" / "Answer:" line (optional heading/bold
    around it) and then keeps talking -- the plain last-non-empty-line
    read below missed it entirely. Search from the end (last boxed, else
    last marker line) so an earlier "answer" mention inside the reasoning
    is never preferred over a later, real one. When the marker line has
    nothing after its colon (the value is on the following line(s)),
    keep reading forward to the next blank line. None if neither is
    present, so the caller falls back to the last-line heuristic."""
    boxed = list(_BOXED_RE.finditer(response))
    if boxed:
        return _clean_marker_text(boxed[-1].group(1)) or None
    markers = list(_ANSWER_MARKER_RE.finditer(response))
    if not markers:
        return None
    match = markers[-1]
    text = match.group(1).strip()
    if not text:
        collected: list[str] = []
        for line in response[match.end() :].split("\n"):
            if not line.strip():
                if collected:
                    break
                continue
            collected.append(line.strip())
        text = " ".join(collected)
    return _clean_marker_text(text) or None


_LIST_MARKER_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+")


def _strip_list_marker(line: str) -> str:
    """Strip a leading markdown bullet ("-", "*", "+") or numbering
    ("1.", "2)") from `line`, for word_sorting responses that list one
    word per line with list formatting."""
    return _LIST_MARKER_RE.sub("", line)


def extract_bbh_prediction(response: str, group: str | None = None) -> str | None:
    """The model's committed BBH answer, normalised (strip, lowercase,
    drop a trailing full stop, collapse whitespace, strip surrounding
    quotes). Never consults the gold target -- comparison to gold stays
    in `score_bbh`.

    1. `_extract_bbh_marker_prediction` (a `\\boxed{...}` or a "Final
       Answer:" / "Answer:" line), when present.
    2. Otherwise, ONLY when `group` is "word_sorting": the final
       non-empty line, and the contiguous run of non-empty lines
       directly above it, are joined with spaces if they are each a
       single whitespace-free token once a leading markdown bullet or
       numbering is stripped (word_sorting responses that put one word
       per line) -- the whole contiguous run, no length cap, so an
       unrelated single-word line further up (e.g. a "Reasoning."
       lead-in) would need its own space to be excluded. A single-line,
       comma-separated list is accepted the same way, but ONLY when
       every comma-separated part is itself a single token -- ordinary
       prose with an incidental comma (seen in truncated
       logical_deduction_seven_objects responses) is left alone.
    3. Otherwise (no marker, `group` is not "word_sorting", or the list
       checks above did not apply): the final non-empty line as-is.

    None if `response` has no non-empty line and no marker."""
    marker = _extract_bbh_marker_prediction(response)
    if marker is not None:
        return marker
    non_empty = [line.strip() for line in response.split("\n") if line.strip()]
    if not non_empty:
        return None
    last = non_empty[-1]
    if group == "word_sorting":
        if " " not in _strip_list_marker(last):
            tokens: list[str] = []
            for line in reversed(non_empty):
                token = _strip_list_marker(line)
                if " " in token:
                    break
                tokens.append(token)
            tokens.reverse()
            if tokens:
                return _normalise_bbh_line(" ".join(tokens))
        elif "," in last:
            parts = [part.strip() for part in last.split(",") if part.strip()]
            if parts and all(" " not in part for part in parts):
                return _normalise_bbh_line(" ".join(parts))
    return _normalise_bbh_line(last)


def _bracket_suffix_match(prediction: str, target: str) -> bool:
    """Labelled leniency -- Task F point 5, dyck_languages only (`target`
    made up solely of bracket/space characters). Qwen3-4B typically
    restates the WHOLE sequence, the given unbalanced prefix plus its own
    closing brackets, instead of emitting only the closing part the
    prompt asks for. Accept `prediction` when its bracket characters,
    spaces removed, END with `target`'s (also space-stripped). Still
    strict everywhere else: an incomplete or wrongly-nested completion
    (5 of the 8 dyck_languages samples inspected for this task) scores
    wrong, same as before."""
    compact_target = target.replace(" ", "")
    return bool(compact_target) and prediction.replace(" ", "").endswith(compact_target)


def score_bbh(response: str, target: str, group: str | None = None) -> bool:
    """`extract_bbh_prediction(response, group)` compared to the
    same-normalised `target`, with two labelled exceptions:

    - `target` of the form "(X)": also accept a prediction that merely
      CONTAINS "(x)" (unchanged: a fuller sentence around the letter);
      a prediction whose whole normalised text IS the bare letter "x";
      or, only when `response` carries a `_extract_bbh_marker_prediction`
      commitment (a `\\boxed{...}` / "Final Answer:" line, not the
      target-agnostic last-line fallback), a prediction whose first
      token is "x" or "(x)". A bare letter merely appearing inside
      ordinary prose ("...it could be a tough call", "I am not sure...")
      is never accepted -- the model must have actually committed via a
      marker.
    - `target` made only of bracket/space characters: see
      `_bracket_suffix_match` (dyck_languages leniency).
    """
    prediction = extract_bbh_prediction(response, group) or ""
    norm_target = _normalise_bbh_line(target)
    if prediction == norm_target:
        return True
    if _PAREN_LETTER_RE.match(target.strip()):
        letter = norm_target.strip("()")
        if norm_target in prediction:
            return True
        if prediction == letter:
            return True
        if _extract_bbh_marker_prediction(response) is not None:
            tokens = prediction.split()
            if tokens and tokens[0] in (letter, f"({letter})"):
                return True
        return False
    if target and all(ch in "()[]{}<> \t" for ch in target):
        return _bracket_suffix_match(prediction, target)
    return False


def extract_prediction(row: RetentionRow, response: str) -> str | None:
    """Dispatch to the extractor matching `row.benchmark`; the scorer's
    best-effort parse of `response`, None if extraction found nothing."""
    if row.benchmark == "gsm8k":
        return extract_gsm8k_prediction(response)
    if row.benchmark == "mmlu":
        return extract_mmlu_prediction(response)
    if row.benchmark == "bbh":
        return extract_bbh_prediction(response, row.group)
    raise ValueError(f"unknown benchmark {row.benchmark!r}")


def score(row: RetentionRow, response: str) -> bool:
    """Dispatch to the scorer matching `row.benchmark`."""
    if row.benchmark == "gsm8k":
        return score_gsm8k(response, row.target)
    if row.benchmark == "mmlu":
        return score_mmlu(response, row.target)
    if row.benchmark == "bbh":
        return score_bbh(response, row.target, row.group)
    raise ValueError(f"unknown benchmark {row.benchmark!r}")

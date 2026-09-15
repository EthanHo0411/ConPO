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

from conpo import config

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_LETTER_RE = re.compile(r"\b([A-D])\b")
_WS_RE = re.compile(r"\s+")
_PAREN_LETTER_RE = re.compile(r"^\([A-Za-z]\)$")
BBH_SUFFIX = "Give only the final answer on the last line."


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


def bbh_suffix() -> str:
    """The instruction that closes every BBH prompt: RETENTION_SUFFIX when
    set to a non-empty string, else BBH_SUFFIX."""
    return config.env("RETENTION_SUFFIX", "") or BBH_SUFFIX


def retention_lenient() -> bool:
    """RETENTION_LENIENT env knob: unset or "" -> False (score_bbh stays
    byte-identical to the strict scorer); "1" -> True (the lenient BBH
    fallback in `score_bbh`). Any other value raises SystemExit, mirroring
    scripts/eval_retention.py's RETENTION_MAX_TOKENS / RETENTION_BBH_TASKS
    validation."""
    raw = config.env("RETENTION_LENIENT", "")
    if raw == "":
        return False
    if raw == "1":
        return True
    raise SystemExit(f"RETENTION_LENIENT must be '1' or unset, got {raw!r}")


def apply_bbh_suffix(prompt: str) -> str:
    """`prompt` with a trailing BBH_SUFFIX swapped for `bbh_suffix()`; the
    "\\n\\n" separator and everything before it stay. Unchanged when the
    prompt does not end with BBH_SUFFIX (rows written under a custom
    suffix). Lets scripts/eval_retention.py honour RETENTION_SUFFIX on
    rows scripts/fetch_data.py wrote with the default."""
    if not prompt.endswith(BBH_SUFFIX):
        return prompt
    return prompt[: -len(BBH_SUFFIX)] + bbh_suffix()


def rows_from_bbh(task: str, hf_rows: Sequence[Mapping[str, Any]]) -> list[RetentionRow]:
    """BBH test rows (fields "input", "target") for one task config to
    RetentionRows. group is the BBH task name. The prompt ends with
    `bbh_suffix()` after a "\\n\\n" separator."""
    rows = []
    suffix = bbh_suffix()
    for index, row in enumerate(hf_rows):
        prompt = f"{row['input']}\n\n{suffix}"
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


_PAREN_OPTION_RE = re.compile(r"\(([A-Z])\)")


def _lenient_paren_letter(response: str) -> str | None:
    """RETENTION_LENIENT paren-target fallback: the letter inside the LAST
    "(X)"-shaped option anywhere in `response` (not just its last line, as
    `extract_bbh_prediction`'s last-line heuristic would see), None if none
    is present. Covers chain-of-thought that commits to an option mid-
    response and then keeps talking without repeating it verbatim on the
    final line (e.g. "...so the ball is with (C)")."""
    matches = list(_PAREN_OPTION_RE.finditer(response))
    return matches[-1].group(1) if matches else None


# ponytail: fixed phrase lists, not an NLI model -- add a phrase or swap in a
# classifier if a new BBH task's phrasing slips past these.
_BBH_NEGATIVE_RE = re.compile(
    r"\b(?:does not tell the truth|doesn't tell the truth|is not telling the truth|is lying|lies|no)\b",
    re.IGNORECASE,
)
_BBH_POSITIVE_RE = re.compile(
    r"\b(?:tells the truth|is telling the truth|is truthful|yes)\b", re.IGNORECASE
)


def _lenient_yes_no(response: str) -> str | None:
    """RETENTION_LENIENT yes/no-target fallback: "no" or "yes" by whichever
    of `_BBH_NEGATIVE_RE` / `_BBH_POSITIVE_RE`'s last match starts later in
    `response`'s last 300 characters, None if neither phrase occurs there."""
    tail = response[-300:]
    neg = list(_BBH_NEGATIVE_RE.finditer(tail))
    pos = list(_BBH_POSITIVE_RE.finditer(tail))
    neg_start = neg[-1].start() if neg else -1
    pos_start = pos[-1].start() if pos else -1
    if neg_start == -1 and pos_start == -1:
        return None
    return "no" if neg_start > pos_start else "yes"


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
      is never accepted -- the model must have committed via a marker.
    - `target` made only of bracket/space characters: see
      `_bracket_suffix_match` (dyck_languages leniency).

    RETENTION_LENIENT=1 (see `retention_lenient`) adds one more fallback,
    tried only when `response` carries no `_extract_bbh_marker_prediction`
    commitment (a marker still wins over the lenient fallback, same as it
    wins over the plain last-line heuristic above): for a "(X)"-shaped
    target, `_lenient_paren_letter`; for a yes/no target, `_lenient_yes_no`.
    Unset or "", this function is byte-identical to the RETENTION_LENIENT-
    unaware version.
    """
    prediction = extract_bbh_prediction(response, group) or ""
    norm_target = _normalise_bbh_line(target)
    if prediction == norm_target:
        return True
    has_marker = _extract_bbh_marker_prediction(response) is not None
    if _PAREN_LETTER_RE.match(target.strip()):
        letter = norm_target.strip("()")
        if norm_target in prediction:
            return True
        if prediction == letter:
            return True
        if has_marker:
            tokens = prediction.split()
            if tokens and tokens[0] in (letter, f"({letter})"):
                return True
        elif retention_lenient():
            lenient_letter = _lenient_paren_letter(response)
            if lenient_letter is not None and lenient_letter.lower() == letter:
                return True
        return False
    if target and all(ch in "()[]{}<> \t" for ch in target):
        return _bracket_suffix_match(prediction, target)
    if retention_lenient() and not has_marker and norm_target in ("yes", "no"):
        decision = _lenient_yes_no(response)
        if decision is not None:
            return decision == norm_target
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

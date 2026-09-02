"""Corruption operators and fixtures for the IFEval "B" family group.

Twelve of the 25 classic IFEval constraint families (the other 13 live in
`ifeval_a.py`). Each family gets a `fixture(c, rng) -> str` that returns a
natural response satisfying the constraint, and an `operator(response, c,
rng) -> Iterator[str]` that yields small, targeted corruptions of a passing
response, most minimal edit first. Both call `conpo.verifiers.build(c)` to
read the checker instance's parsed kwargs (e.g. `_num_highlights`); the
verifier itself is the source of truth, these are heuristics.

Every operator below edits from one end of the response, then the other
(`_iter_two_directions`), so the smallest single-occurrence edit is always
tried before anything larger.

family id -> corruption strategy; collateral risk
----------------------------------------------------------------------------
detectable_format:number_highlighted_sections
    Strip the "*"/"**" wrapping off one highlighted span at a time, from the
    end then the start. Collateral: none among this group.
detectable_format:multiple_sections
    Delete one "{spliter} N" marker occurrence at a time (end, then start).
    Collateral: a co-constraint counting occurrences of the spliter word
    itself would also break; none of the other 11 families does.
detectable_format:json_format
    Drop the final character (unbalances the closing brace), or append a
    trailing word after the JSON body. Collateral: punctuation:no_comma
    would need the JSON body to hold no comma, which valid JSON with more
    than one field cannot guarantee.
detectable_format:title
    Remove the "<<"/">>" wrapper, or empty the text between the brackets.
    Collateral: negligible, the wrapper carries no words other families
    depend on.
combination:two_responses
    Delete the "******" separator (merges into one response), or insert a
    second separator right after it (creates an empty middle segment).
    Collateral: none among this group.
combination:repeat_prompt
    Drop the first character of the repeated prompt, or prepend "Sure. "
    before it. Collateral: breaks startend:quotation if the whole response
    is also quote-wrapped, since the prefix lands outside the opening quote.
startend:end_checker
    Drop the last character of the end phrase, or append a short trailer
    after it. Collateral: same quotation risk as repeat_prompt, on the
    closing side.
change_case:capital_word_frequency
    Flip one word's case at a time (an all-caps word to lower to reduce the
    count, a lowercase-bearing word to upper to raise it), end then start.
    Collateral: would collide with change_case:english_capital/lowercase if
    combined, since both target the whole response's case.
change_case:english_capital
    Lowercase one alphabetic character, end then start. Collateral: could
    break change_case:capital_word_frequency if combined and the flipped
    character was carrying that family's one required all-caps word.
change_case:english_lowercase
    Uppercase one alphabetic character, end then start. Collateral: mirrors
    english_capital's risk against capital_word_frequency.
punctuation:no_comma
    Insert a comma after the first word, or before the last word.
    Collateral: none among this group.
startend:quotation
    Drop the closing quote, or the opening quote. Collateral: breaks
    combination:repeat_prompt or startend:end_checker if either family's
    required edge sits at the same end of the response.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Iterator

from conpo.operators.registry import register
from conpo.verifiers import Constraint, build

# ---------------------------------------------------------------------------
# Shared fixture prose: deterministic given rng, comma-free, all-lowercase so
# it can be reused as-is by the case-sensitive families.
# ---------------------------------------------------------------------------

_FILLER: tuple[str, ...] = (
    "the system checks each condition before moving on",
    "every step gets logged for later review",
    "the pipeline runs the same way each time it starts",
    "each worker reports back once its task finishes",
    "the queue holds requests until a worker is free",
    "results are written to a shared file at the end",
    "a short summary follows once the run completes",
    "nothing here depends on any external service",
)

_MIN_WORDS = 34


def _filler(rng: random.Random, min_words: int = _MIN_WORDS) -> str:
    """Deterministic, comma-free, all-lowercase filler of at least `min_words` words."""
    pool = list(_FILLER)
    rng.shuffle(pool)
    picked: list[str] = []
    count = 0
    for sentence in pool:
        picked.append(sentence)
        count += len(sentence.split())
        if count >= min_words:
            break
    return ". ".join(picked) + "."


def _prose(rng: random.Random, min_words: int = _MIN_WORDS) -> str:
    """`_filler` with a capitalized opening letter, for case-insensitive fixtures."""
    text = _filler(rng, min_words)
    return text[0].upper() + text[1:]


def _iter_two_directions(text: str, step: Callable[[str, bool], str | None], cap: int = 6) -> Iterator[str]:
    """Yield successive single-occurrence edits: first stripping from the end, then the start."""
    for from_end in (True, False):
        current = text
        for _ in range(cap):
            nxt = step(current, from_end)
            if nxt is None:
                break
            yield nxt
            current = nxt


# ---------------------------------------------------------------------------
# detectable_format:number_highlighted_sections
# ---------------------------------------------------------------------------

_HIGHLIGHT_PATTERN = re.compile(r"\*\*[^\n\*]+\*\*|\*[^\n\*]+\*")


def _strip_highlight(text: str, from_end: bool) -> str | None:
    matches = list(_HIGHLIGHT_PATTERN.finditer(text))
    if not matches:
        return None
    m = matches[-1] if from_end else matches[0]
    return text[: m.start()] + m.group().strip("*") + text[m.end() :]


def _highlighted_sections_fixture(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    n = inst._num_highlights
    spans = " ".join(f"*point {i + 1}*" for i in range(n))
    return f"{_prose(rng)} {spans}".strip()


def _highlighted_sections_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    yield from _iter_two_directions(response, _strip_highlight)


register("detectable_format:number_highlighted_sections", _highlighted_sections_operator, _highlighted_sections_fixture)

# ---------------------------------------------------------------------------
# detectable_format:multiple_sections
# ---------------------------------------------------------------------------


def _multiple_sections_fixture(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    spliter = inst._section_spliter
    n = inst._num_sections
    parts = [f"{spliter} {i + 1}\n{_prose(rng, 20)}" for i in range(n)]
    return "\n".join(parts)


def _multiple_sections_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    pattern = re.compile(r"\s?" + inst._section_spliter + r"\s?\d+\s?")

    def step(text: str, from_end: bool) -> str | None:
        matches = list(pattern.finditer(text))
        if not matches:
            return None
        m = matches[-1] if from_end else matches[0]
        return text[: m.start()] + " " + text[m.end() :]

    yield from _iter_two_directions(response, step)


register("detectable_format:multiple_sections", _multiple_sections_operator, _multiple_sections_fixture)

# ---------------------------------------------------------------------------
# detectable_format:json_format
# ---------------------------------------------------------------------------


def _json_format_fixture(c: Constraint, rng: random.Random) -> str:
    body = {
        "summary": _filler(rng, 40),
        "status": "ok",
        "notes": ["checked inputs", "verified outputs", "logged results"],
    }
    return json.dumps(body)


def _json_format_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    if len(response) > 1:
        yield response[:-1]
    yield response + " trailing"


register("detectable_format:json_format", _json_format_operator, _json_format_fixture)

# ---------------------------------------------------------------------------
# detectable_format:title
# ---------------------------------------------------------------------------


def _title_fixture(c: Constraint, rng: random.Random) -> str:
    return f"{_prose(rng)} <<field notes>>"


def _title_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    if "<<" in response and ">>" in response:
        yield response.replace("<<", "").replace(">>", "")
    m = re.search(r"<<([^\n]+)>>", response)
    if m and m.group(1).strip():
        yield response[: m.start()] + "<<>>" + response[m.end() :]


register("detectable_format:title", _title_operator, _title_fixture)

# ---------------------------------------------------------------------------
# combination:two_responses
# ---------------------------------------------------------------------------


def _two_responses_fixture(c: Constraint, rng: random.Random) -> str:
    a = _prose(rng, 20)
    b = _prose(rng, 20)
    if b.strip() == a.strip():
        b = b + " Indeed."
    return f"{a}\n******\n{b}"


def _two_responses_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    idx = response.rfind("******")
    if idx == -1:
        return
    yield response[:idx] + "\n" + response[idx + 6 :]  # merge into one response
    yield response[: idx + 6] + "******" + response[idx + 6 :]  # empty middle segment


register("combination:two_responses", _two_responses_operator, _two_responses_fixture)

# ---------------------------------------------------------------------------
# combination:repeat_prompt
# ---------------------------------------------------------------------------


def _repeat_prompt_fixture(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    prompt = inst._prompt_to_repeat.strip()
    return f"{prompt} {_prose(rng)}"


def _repeat_prompt_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    if len(response) > 1:
        yield response[1:]  # drop the prompt's first character
    yield "Sure. " + response  # a lead-in before the repeat


register("combination:repeat_prompt", _repeat_prompt_operator, _repeat_prompt_fixture)

# ---------------------------------------------------------------------------
# startend:end_checker
# ---------------------------------------------------------------------------


def _end_checker_fixture(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    phrase = inst._end_phrase.strip()
    return f"{_prose(rng)} {phrase}"


def _end_checker_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    if len(response) > 1:
        yield response[:-1]  # drop the phrase's last character
    yield response + " (end)"  # trailing text after the required phrase


register("startend:end_checker", _end_checker_operator, _end_checker_fixture)

# ---------------------------------------------------------------------------
# change_case:capital_word_frequency
# ---------------------------------------------------------------------------


def _capital_word_frequency_fixture(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    n = inst._frequency if inst._comparison_relation == "at least" else max(inst._frequency - 1, 0)
    caps = " ".join(f"WORD{i}" for i in range(n))
    return f"{_prose(rng, 24)} {caps}".strip()


def _capital_word_frequency_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    at_least = inst._comparison_relation == "at least"

    def is_cap(tok: str) -> bool:
        return tok.isupper() and any(ch.isalpha() for ch in tok)

    def is_plain(tok: str) -> bool:
        return any(ch.islower() for ch in tok)

    predicate, transform = (is_cap, str.lower) if at_least else (is_plain, str.upper)

    def step(text: str, from_end: bool) -> str | None:
        matches = [m for m in re.finditer(r"\S+", text) if predicate(m.group())]
        if not matches:
            return None
        m = matches[-1] if from_end else matches[0]
        return text[: m.start()] + transform(m.group()) + text[m.end() :]

    yield from _iter_two_directions(response, step)


register("change_case:capital_word_frequency", _capital_word_frequency_operator, _capital_word_frequency_fixture)

# ---------------------------------------------------------------------------
# change_case:english_capital
# ---------------------------------------------------------------------------


def _english_capital_fixture(c: Constraint, rng: random.Random) -> str:
    return _filler(rng, 40).upper()


def _flip_one_letter(response: str, upper_to_lower: bool) -> Iterator[str]:
    idx = [i for i, ch in enumerate(response) if ch.isalpha()]
    if not idx:
        return
    for i in (idx[-1], idx[0]):
        ch = response[i].lower() if upper_to_lower else response[i].upper()
        yield response[:i] + ch + response[i + 1 :]


def _english_capital_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    yield from _flip_one_letter(response, upper_to_lower=True)


register("change_case:english_capital", _english_capital_operator, _english_capital_fixture)

# ---------------------------------------------------------------------------
# change_case:english_lowercase
# ---------------------------------------------------------------------------


def _english_lowercase_fixture(c: Constraint, rng: random.Random) -> str:
    return _filler(rng, 40)


def _english_lowercase_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    yield from _flip_one_letter(response, upper_to_lower=False)


register("change_case:english_lowercase", _english_lowercase_operator, _english_lowercase_fixture)

# ---------------------------------------------------------------------------
# punctuation:no_comma
# ---------------------------------------------------------------------------


def _no_comma_fixture(c: Constraint, rng: random.Random) -> str:
    return _prose(rng, 40)


def _no_comma_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    words = response.split(" ")
    if len(words) < 2:
        return
    first = list(words)
    first[0] = first[0] + ","
    yield " ".join(first)
    last = list(words)
    last[-1] = "," + last[-1]
    yield " ".join(last)


register("punctuation:no_comma", _no_comma_operator, _no_comma_fixture)

# ---------------------------------------------------------------------------
# startend:quotation
# ---------------------------------------------------------------------------


def _quotation_fixture(c: Constraint, rng: random.Random) -> str:
    return f'"{_prose(rng, 40)}"'


def _quotation_operator(response: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    if len(response) > 1 and response.endswith('"'):
        yield response[:-1]
    if len(response) > 1 and response.startswith('"'):
        yield response[1:]


register("startend:quotation", _quotation_operator, _quotation_fixture)

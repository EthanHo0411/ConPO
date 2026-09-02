"""ConPO verifier layer: a thin wrapper over the vendored IFEvalG registry.

Provides the 54-family registry split (IFEval 25 + IFBench-Train 29), the
held-out IFBench-test family ids, an immutable Constraint type, and the
verifier primitives ConPO's losses need: check(), verifier_vector(),
violation_set(), feasible(), description() and describe_failure().
"""

import json
import random
import re
import zlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import langdetect
import nltk

from conpo.vendor.ifevalg import instructions_registry, instructions_util
from conpo.vendor.ifevalg.instructions import Instruction

# The vendored language checkers call langdetect, whose PRNG is unseeded by
# default and makes change_case:* and language:* verdicts flaky. Pin it once
# here, the single import every verifier call goes through.
langdetect.DetectorFactory.seed = 0

# ---------------------------------------------------------------------------
# Family registry: 54 = 25 IFEval + 29 IFBench-Train, in registry order.
# ---------------------------------------------------------------------------

# The 25 keys that precede the "# New Constraints!" comment in
# conpo/vendor/ifevalg/instructions_registry.py's INSTRUCTION_DICT.
IFEVAL_FAMILIES: tuple[str, ...] = (
    "keywords:existence",
    "keywords:frequency",
    "keywords:forbidden_words",
    "keywords:letter_frequency",
    "language:response_language",
    "length_constraints:number_sentences",
    "length_constraints:number_paragraphs",
    "length_constraints:number_words",
    "length_constraints:nth_paragraph_first_word",
    "detectable_content:number_placeholders",
    "detectable_content:postscript",
    "detectable_format:number_bullet_lists",
    "detectable_format:constrained_response",
    "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections",
    "detectable_format:json_format",
    "detectable_format:title",
    "combination:two_responses",
    "combination:repeat_prompt",
    "startend:end_checker",
    "change_case:capital_word_frequency",
    "change_case:english_capital",
    "change_case:english_lowercase",
    "punctuation:no_comma",
    "startend:quotation",
)

# The 29 keys at and after the "# New Constraints!" comment (IFBench-Train).
IFTRAIN_FAMILIES: tuple[str, ...] = (
    "copy:repeat_phrase",
    "copy:copy",
    "new:copy_span_idx",
    "detectable_format:sentence_hyphens",
    "keywords:no_adjacent_consecutive",
    "detectable_format:square_brackets",
    "keywords:word_once",
    "keywords:word_count_different_numbers",
    "keywords:exclude_word_harder",
    "paragraphs:paragraphs",
    "paragraphs:paragraphs2",
    "first_word:first_word_sent",
    "first_word:first_word_answer",
    "last_word:last_word_sent",
    "last_word:last_word_answer",
    "detectable_format:bigram_wrapping",
    "copy:copying_simple",
    "copy:copying_multiple",
    "punctuation:punctuation_dot",
    "punctuation:punctuation_exclamation",
    "count:lowercase_counting",
    "letters:letter_counting",
    "letters:letter_counting2",
    "count:counting_composition",
    "count:count_unique",
    "count:count_increment_word",
    "keywords:palindrome",
    "keywords:keyword_specific_position",
    "keywords:start_end",
)

FAMILIES: tuple[str, ...] = IFEVAL_FAMILIES + IFTRAIN_FAMILIES

assert set(IFEVAL_FAMILIES) | set(IFTRAIN_FAMILIES) == set(instructions_registry.INSTRUCTION_DICT), (
    "IFEVAL_FAMILIES + IFTRAIN_FAMILIES must equal the vendored registry keys"
)
assert len(FAMILIES) == 54, f"expected 54 families, got {len(FAMILIES)}"

# The 58 IFBench TEST families, held out of ConPO training entirely. Read
# once from
# scratchpad/vendor_src/IFBench/ifbench/instructions_registry.py's
# INSTRUCTION_DICT (the block after the "# --- IFBench OOD verifiers ---"
# comment; that file also carries its own copy of the 25 classic IFEval
# keys, which duplicate IFEVAL_FAMILIES above and are excluded here).
HELDOUT_TEST_FAMILIES: tuple[str, ...] = (
    "count:word_count_range",
    "count:unique_word_count",
    "ratio:stop_words",
    "ratio:sentence_type",
    "ratio:sentence_balance",
    "count:conjunctions",
    "count:person_names",
    "ratio:overlap",
    "count:numbers",
    "words:alphabet",
    "words:vowel",
    "words:consonants",
    "sentence:alliteration_increment",
    "words:palindrome",
    "count:punctuation",
    "format:parentheses",
    "format:quotes",
    "words:prime_lengths",
    "format:options",
    "format:newline",
    "format:emoji",
    "ratio:sentence_words",
    "count:words_japanese",
    "words:start_verb",
    "words:repeats",
    "sentence:keyword",
    "count:pronouns",
    "words:odd_even_syllables",
    "words:last_first",
    "words:paragraph_last_first",
    "sentence:increment",
    "words:no_consecutive",
    "format:line_indent",
    "format:quote_unquote",
    "format:list",
    "format:thesis",
    "format:sub-bullets",
    "format:no_bullets_bullets",
    "custom:multiples",
    "custom:mcq_count_length",
    "custom:reverse_newline",
    "custom:word_reverse",
    "custom:character_reverse",
    "custom:sentence_alphabet",
    "custom:european_capitals_sort",
    "custom:csv_city",
    "custom:csv_special_character",
    "custom:csv_quotes",
    "custom:date_format_list",
    "count:keywords_multiple",
    "words:keywords_specific_position",
    "words:words_position",
    "repeat:repeat_change",
    "repeat:repeat_simple",
    "repeat:repeat_span",
    "format:title_case",
    "format:output_template",
    "format:no_whitespace",
)

assert len(HELDOUT_TEST_FAMILIES) == 58, f"expected 58 held-out families, got {len(HELDOUT_TEST_FAMILIES)}"
assert set(HELDOUT_TEST_FAMILIES).isdisjoint(FAMILIES), "held-out IFBench-test families must not overlap FAMILIES"

# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraint:
    """One (instruction_id, kwargs) pair, kwargs stored as an immutable mapping."""

    iid: str
    kwargs: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", MappingProxyType(dict(self.kwargs)))

    @classmethod
    def from_ground_truth(cls, iid: str, kwargs: Mapping[str, object] | None) -> "Constraint":
        """Build a Constraint from a dataset's raw ground-truth kwargs.

        Drops None-valued keys before storing. This matches the convention
        open-instruct itself uses to call build_description: ground_truth_utils.py
        (IFEvalVerifier.get_reward, `args = {k: v for k, v in args.items() if v
        is not None}` immediately before `instruction_instance.build_description(**args)`),
        also followed by stackelberg-if/probe_ifrepair.py:228
        (`inst.build_description(**{k: v for k, v in (kwargs or {}).items() if v
        is not None})`). Every vendored build_description signature defaults
        each optional parameter to None and re-samples when it sees None, so a
        missing key and an explicit None key are handled identically; dropping
        keeps the stored kwargs minimal without changing behaviour.
        """
        raw = kwargs or {}
        filtered = {k: v for k, v in raw.items() if v is not None}
        return cls(iid, filtered)


def _kwargs_seed(c: Constraint) -> int:
    """Deterministic RNG seed derived from the constraint's id and kwargs."""
    key = f"conpo:{c.iid}:{json.dumps(dict(c.kwargs), sort_keys=True, default=str)}"
    return zlib.crc32(key.encode())


def _build_with_description(c: Constraint) -> tuple[Instruction, str]:
    """Instantiate the checker and call build_description under a pinned seed.

    Several vendored classes sample a default from the global `random` module
    when a kwarg is None or missing; pinning and restoring random's state
    makes build() and description() deterministic and side-effect-free on the
    caller's RNG stream, mirroring stackelberg-if/probe_ifrepair.py:build_checkers.
    """
    cls = instructions_registry.INSTRUCTION_DICT[c.iid]
    inst = cls(c.iid)
    state = random.getstate()
    random.seed(_kwargs_seed(c))
    try:
        desc = inst.build_description(**dict(c.kwargs))
    finally:
        random.setstate(state)
    return inst, desc


def build(c: Constraint) -> Instruction:
    """Instantiate and configure the checker for `c`."""
    inst, _ = _build_with_description(c)
    return inst


def description(c: Constraint) -> str:
    """The requirement sentence for `c`."""
    _, desc = _build_with_description(c)
    return desc


# ---------------------------------------------------------------------------
# nltk data
# ---------------------------------------------------------------------------

_NLTK_DATA_DIR = Path(__file__).resolve().parent.parent / "nltk_data"
_nltk_ready = False


def ensure_nltk() -> None:
    """Download punkt/punkt_tab into a local nltk_data dir if not present.

    Lazy: called by check() so importing this module never touches the
    network. Idempotent and offline-safe once the data exists.
    """
    global _nltk_ready
    if _nltk_ready:
        return
    _NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = str(_NLTK_DATA_DIR)
    if data_dir not in nltk.data.path:
        nltk.data.path.insert(0, data_dir)
    for pkg, find_path in (("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab")):
        try:
            nltk.data.find(find_path)
        except LookupError:
            nltk.download(pkg, download_dir=data_dir, quiet=True)
    _nltk_ready = True


# ---------------------------------------------------------------------------
# check / verifier_vector / violation_set / feasible
# ---------------------------------------------------------------------------

CHECK_ERRORS: Counter = Counter()


def check(c: Constraint, response: str) -> bool:
    """Verifier verdict for one constraint, fail-closed on exceptions."""
    ensure_nltk()
    inst = build(c)
    try:
        return bool(inst.check_following(response))
    except Exception:
        CHECK_ERRORS[c.iid] += 1
        return False


def verifier_vector(constraints: tuple[Constraint, ...], response: str) -> tuple[bool, ...]:
    return tuple(check(c, response) for c in constraints)


def violation_set(constraints: tuple[Constraint, ...], response: str) -> frozenset[str]:
    return frozenset(c.iid for c, ok in zip(constraints, verifier_vector(constraints, response)) if not ok)


def feasible(constraints: tuple[Constraint, ...], response: str) -> bool:
    """F(x, y): the conjunctive reward, True only if every constraint passes."""
    return all(verifier_vector(constraints, response))


# ---------------------------------------------------------------------------
# describe_failure: requirement sentence plus a cheap observed-vs-required
# measurement, for repair prompts. Each _measure_* function reads the
# checker instance's private state (set by build_description) and returns
# the "X, required Y" tail; families with no natural measurement fall back
# to the plain requirement sentence.
# ---------------------------------------------------------------------------


def _count_star_paragraphs(text: str) -> int:
    """Paragraph count as ParagraphChecker/ParagraphBasicChecker count it."""
    paragraphs = re.split(r"\s?\*\*\*\s?", text)
    n = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip() and (i == 0 or i == len(paragraphs) - 1):
            n -= 1
    return n


def _count_dblnewline_paragraphs(text: str) -> int:
    """Paragraph count as ParagraphBasicChecker2 counts it (blank-line divider)."""
    paragraphs = re.split(r"\n\n", text)
    n = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip() and (i == 0 or i == len(paragraphs) - 1):
            n -= 1
    return n


def _measure_keyword_existence(inst: Instruction, response: str) -> str:
    missing = [kw for kw in inst._keywords if not re.search(kw, response, flags=re.IGNORECASE)]
    return f"missing keywords {missing}, required all of {list(inst._keywords)}"


def _measure_keyword_frequency(inst: Instruction, response: str) -> str:
    n = len(re.findall(inst._keyword, response, flags=re.IGNORECASE))
    return f"{n} occurrences of '{inst._keyword}', required {inst._comparison_relation} {inst._frequency}"


def _measure_forbidden_words(inst: Instruction, response: str) -> str:
    present = [w for w in inst._forbidden_words if re.search(r"\b" + w + r"\b", response, flags=re.IGNORECASE)]
    return f"forbidden words present {present}, required none of {list(inst._forbidden_words)}"


def _measure_letter_frequency(inst: Instruction, response: str) -> str:
    n = Counter(response.lower())[inst._letter]
    return f"letter '{inst._letter}' appears {n} times, required {inst._comparison_relation} {inst._frequency}"


def _measure_response_language(inst: Instruction, response: str) -> str:
    try:
        detected = langdetect.detect(response)
    except langdetect.LangDetectException as exc:
        return f"language detection failed ({exc}), required '{inst._language}'"
    return f"detected language '{detected}', required '{inst._language}'"


def _measure_number_sentences(inst: Instruction, response: str) -> str:
    n = instructions_util.count_sentences(response)
    return f"{n} sentences, required {inst._comparison_relation} {inst._num_sentences_threshold}"


def _measure_number_paragraphs(inst: Instruction, response: str) -> str:
    n = _count_star_paragraphs(response)
    return f"{n} paragraphs, required exactly {inst._num_paragraphs}"


def _measure_number_words(inst: Instruction, response: str) -> str:
    n = instructions_util.count_words(response)
    return f"{n} words, required {inst._comparison_relation} {inst._num_words}"


def _measure_nth_paragraph_first_word(inst: Instruction, response: str) -> str:
    paragraphs = re.split(r"\n\n", response)
    n = len(paragraphs)
    first = "<none>"
    if inst._nth_paragraph <= n and paragraphs[inst._nth_paragraph - 1].strip():
        word = paragraphs[inst._nth_paragraph - 1].strip().split()[0].lstrip("'\"")
        first = "".join(ch for ch in word if ch not in {".", ",", "?", "!", "'", '"'}).lower()
    return (
        f"{n} paragraphs, paragraph {inst._nth_paragraph} starts with '{first}', "
        f"required {inst._num_paragraphs} paragraphs with paragraph {inst._nth_paragraph} "
        f"starting '{inst._first_word}'"
    )


def _measure_number_placeholders(inst: Instruction, response: str) -> str:
    n = len(re.findall(r"\[.*?\]", response))
    return f"{n} placeholders, required at least {inst._num_placeholders}"


def _measure_postscript(inst: Instruction, response: str) -> str:
    found = inst._postscript_marker.lower() in response.lower()
    status = "found" if found else "not found"
    return f"postscript marker '{inst._postscript_marker}' {status}, required present at the end of the response"


def _measure_number_bullet_lists(inst: Instruction, response: str) -> str:
    bullets = re.findall(r"^\s*\*[^\*].*$", response, flags=re.MULTILINE)
    bullets2 = re.findall(r"^\s*-.*$", response, flags=re.MULTILINE)
    n = len(bullets) + len(bullets2)
    return f"{n} bullet points, required exactly {inst._num_bullets}"


def _measure_constrained_response(inst: Instruction, response: str) -> str:
    found = [opt for opt in inst._constrained_responses if opt in response.strip()]
    return f"matched options {found}, required one of {list(inst._constrained_responses)}"


def _measure_number_highlighted_sections(inst: Instruction, response: str) -> str:
    highlights = re.findall(r"\*[^\n\*]*\*", response)
    double = re.findall(r"\*\*[^\n\*]*\*\*", response)
    n = sum(1 for h in highlights if h.strip("*").strip())
    n += sum(1 for h in double if h.removeprefix("**").removesuffix("**").strip())
    return f"{n} highlighted sections, required at least {inst._num_highlights}"


def _measure_multiple_sections(inst: Instruction, response: str) -> str:
    pattern = r"\s?" + inst._section_spliter + r"\s?\d+\s?"
    n = len(re.split(pattern, response)) - 1
    return f"{n} sections marked with '{inst._section_spliter}', required at least {inst._num_sections}"


def _measure_json_format(inst: Instruction, response: str) -> str:
    value = (
        response.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
        return "parsed as valid JSON, required valid JSON"
    except ValueError as exc:
        return f"json parse error ({exc}), required the entire output to be valid JSON"


def _measure_title(inst: Instruction, response: str) -> str:
    titles = re.findall(r"<<[^\n]+>>", response)
    ok = any(t.lstrip("<").rstrip(">").strip() for t in titles)
    return f"title present={ok}, required a non-empty title wrapped in <<double angular brackets>>"


def _measure_two_responses(inst: Instruction, response: str) -> str:
    parts = [p for p in response.split("******") if p.strip()]
    return f"{len(parts)} non-empty response(s) split on ******, required exactly 2 distinct responses"


def _measure_repeat_prompt(inst: Instruction, response: str) -> str:
    ok = response.strip().lower().startswith(inst._prompt_to_repeat.strip().lower())
    return f"starts by repeating the prompt={ok}, required the prompt repeated verbatim first"


def _measure_end_checker(inst: Instruction, response: str) -> str:
    tail = response.strip().strip('"')
    tail = tail[-60:] if len(tail) > 60 else tail
    return f"response ends with '...{tail}', required ending exactly with '{inst._end_phrase}'"


def _measure_capital_word_frequency(inst: Instruction, response: str) -> str:
    words = nltk.word_tokenize(response)
    n = sum(1 for w in words if w.isupper())
    return f"{n} all-capital words, required {inst._comparison_relation} {inst._frequency}"


def _measure_english_capital(inst: Instruction, response: str) -> str:
    return f"is all-uppercase={response.isupper()}, required entire response in English, all capital letters"


def _measure_english_lowercase(inst: Instruction, response: str) -> str:
    return f"is all-lowercase={response.islower()}, required entire response in English, all lowercase letters"


def _measure_no_comma(inst: Instruction, response: str) -> str:
    n = len(re.findall(r",", response))
    return f"{n} commas, required 0"


def _measure_quotation(inst: Instruction, response: str) -> str:
    v = response.strip()
    ok = len(v) > 1 and v[0] == '"' and v[-1] == '"'
    return f"wrapped in double quotes={ok}, required the entire response wrapped in double quotation marks"


def _measure_repeat_phrase(inst: Instruction, response: str) -> str:
    first, last = inst._phrase.split()[0], inst._phrase.split()[-1]
    found = re.findall(rf"{first} .*? {last}", response)
    return f"{len(found)} matches of '{first} ... {last}', required exactly {inst._small_n}"


def _measure_copy(inst: Instruction, response: str) -> str:
    ok = response.strip().lower() == inst._prompt_to_repeat.strip().lower()
    return f"exact copy={ok}, response length {len(response)} vs prompt length {len(inst._prompt_to_repeat)}"


def _measure_copy_span_idx(inst: Instruction, response: str) -> str:
    expected = inst._prompt_to_repeat[inst._n_start : inst._n_end].strip()
    return (
        f"response='{response.strip()[:60]}', "
        f"required the exact span [{inst._n_start}:{inst._n_end}]='{expected[:60]}'"
    )


def _measure_sentence_hyphens(inst: Instruction, response: str) -> str:
    segments = len(response.split("-"))
    sentences = len(instructions_util.split_into_sentences(re.sub("-", " ", response)))
    return f"{segments} hyphen-joined segments vs {sentences} detected sentences, required equal, no internal spaces"


def _measure_no_adjacent_consecutive(inst: Instruction, response: str) -> str:
    words = response.split()
    for i in range(len(words) - 1):
        a, b = words[i][:1].lower(), words[i + 1][:1].lower()
        if a.isalpha() and b.isalpha() and ord(b) - ord(a) == 1:
            return f"adjacent words '{words[i]}' -> '{words[i + 1]}' start with consecutive letters, required none"
    return "no consecutive-letter adjacent pair found, requirement already satisfied"


def _measure_square_brackets(inst: Instruction, response: str) -> str:
    words = response.split()
    bad = [w for w in words if not (w.startswith("[") and w.endswith("]"))]
    return f"{len(bad)} of {len(words)} words not bracketed, required every word wrapped in [brackets]"


def _measure_word_once(inst: Instruction, response: str) -> str:
    n = len(re.findall(inst._keyword, response, flags=re.IGNORECASE))
    return f"{n} occurrences of '{inst._keyword}', required exactly 1"


def _measure_exclude_word_harder(inst: Instruction, response: str) -> str:
    n = response.count(" " + inst._keyword + " ")
    return f"{n} occurrences of ' {inst._keyword} ', required 0"


def _measure_paragraphs_basic(inst: Instruction, response: str) -> str:
    return f"{_count_star_paragraphs(response)} paragraphs (*** divider), required exactly 2"


def _measure_paragraphs_basic2(inst: Instruction, response: str) -> str:
    return f"{_count_dblnewline_paragraphs(response)} paragraphs (blank-line divider), required exactly 2"


def _measure_first_word_sent(inst: Instruction, response: str) -> str:
    sentences = [s for s in instructions_util.split_into_sentences(response) if s.strip()]
    bad = [s.split()[0] for s in sentences if s.split()[0].lower() != inst._first_word.lower()]
    return f"{len(bad)} of {len(sentences)} sentences do not start with '{inst._first_word}', required all to"


def _measure_first_word_answer(inst: Instruction, response: str) -> str:
    words = response.split()
    first = words[0] if words else "<empty>"
    return f"first word '{first}', required '{inst._first_word}'"


def _measure_last_word_sent(inst: Instruction, response: str) -> str:
    sentences = [s for s in instructions_util.split_into_sentences(response) if s.strip()]
    bad = []
    for s in sentences:
        w = re.sub(r"[^\w\s]", "", s.split()[-1])
        if w.lower() != inst._last_word.lower():
            bad.append(w)
    return f"{len(bad)} of {len(sentences)} sentences do not end with '{inst._last_word}', required all to"


def _measure_last_word_answer(inst: Instruction, response: str) -> str:
    words = response.split()
    last = re.sub(r"[^\w\s]", "", words[-1]) if words else "<empty>"
    return f"last word '{last}', required '{inst._last_word}'"


def _measure_bigram_wrapping(inst: Instruction, response: str) -> str:
    words = response.split()
    n_bad = 0
    for i in range(0, len(words) - 1, 2):
        if i + 1 < len(words) and not (words[i].startswith("<<") and words[i + 1].endswith(">>")):
            n_bad += 1
    return f"{n_bad} unwrapped bigrams, required 0"


def _measure_copying_multiple(inst: Instruction, response: str) -> str:
    parts = response.split("******")
    return f"{len(parts)} copies split on ******, required exactly {inst._N} exact copies of the prompt"


def _measure_punctuation_dot(inst: Instruction, response: str) -> str:
    n = len(re.findall(r"\.", response))
    return f"{n} dots, required 0"


def _measure_punctuation_exclamation(inst: Instruction, response: str) -> str:
    n = len(re.findall(r"!", response))
    return f"{n} exclamation marks, required 0"


def _measure_lowercase_counting(inst: Instruction, response: str) -> str:
    n = len(re.findall(r"\b[a-z]+\b", response))
    return f"{n} lowercase words, required at most {inst._N}"


def _measure_letter_counting(inst: Instruction, response: str) -> str:
    n = len(re.findall(r"[a-zA-Z]", response))
    return f"{n} letters, required {inst._relation} {inst._N}"


def _measure_counting_composition(inst: Instruction, response: str) -> str:
    paragraphs = re.split(r"\s?\*\*\*\s?", response)
    return (
        f"{len(paragraphs)} paragraphs, required 3 paragraphs of exactly {inst._n_sent} sentences "
        f"with exactly {inst._n_words} words each"
    )


def _measure_count_unique(inst: Instruction, response: str) -> str:
    words = nltk.word_tokenize(response)
    return f"{len(words)} words, {len(set(words))} unique, required all words unique"


def _measure_count_increment_word(inst: Instruction, response: str) -> str:
    n1 = len(re.findall(inst._keyword1, response, flags=re.IGNORECASE))
    n2 = len(re.findall(inst._keyword2, response, flags=re.IGNORECASE))
    return f"'{inst._keyword1}' x{n1} (required 1), '{inst._keyword2}' x{n2} (required 2)"


def _measure_palindrome(inst: Instruction, response: str) -> str:
    pals = [w for w in response.split() if w and w == w[::-1]]
    return f"{len(pals)} palindromic word(s) found {pals[:3]}, required at least 1"


def _measure_keyword_specific_position(inst: Instruction, response: str) -> str:
    sentences = instructions_util.split_into_sentences(response)
    if len(sentences) < inst._n:
        return f"only {len(sentences)} sentences, required sentence {inst._n} to hold '{inst._keyword}' as word {inst._m}"
    words = nltk.word_tokenize(sentences[inst._n - 1])
    actual = words[inst._m - 1] if len(words) >= inst._m else "<none>"
    return f"word {inst._m} of sentence {inst._n} is '{actual}', required '{inst._keyword}'"


def _measure_start_end(inst: Instruction, response: str) -> str:
    words = nltk.word_tokenize(response)
    if len(words) < 2:
        return f"only {len(words)} word(s), required first and last word identical"
    return f"first word '{words[0]}', last word '{words[-1]}', required identical"


_MEASUREMENTS = {
    "keywords:existence": _measure_keyword_existence,
    "keywords:frequency": _measure_keyword_frequency,
    "keywords:forbidden_words": _measure_forbidden_words,
    "keywords:letter_frequency": _measure_letter_frequency,
    "language:response_language": _measure_response_language,
    "length_constraints:number_sentences": _measure_number_sentences,
    "length_constraints:number_paragraphs": _measure_number_paragraphs,
    "length_constraints:number_words": _measure_number_words,
    "length_constraints:nth_paragraph_first_word": _measure_nth_paragraph_first_word,
    "detectable_content:number_placeholders": _measure_number_placeholders,
    "detectable_content:postscript": _measure_postscript,
    "detectable_format:number_bullet_lists": _measure_number_bullet_lists,
    "detectable_format:constrained_response": _measure_constrained_response,
    "detectable_format:number_highlighted_sections": _measure_number_highlighted_sections,
    "detectable_format:multiple_sections": _measure_multiple_sections,
    "detectable_format:json_format": _measure_json_format,
    "detectable_format:title": _measure_title,
    "combination:two_responses": _measure_two_responses,
    "combination:repeat_prompt": _measure_repeat_prompt,
    "startend:end_checker": _measure_end_checker,
    "change_case:capital_word_frequency": _measure_capital_word_frequency,
    "change_case:english_capital": _measure_english_capital,
    "change_case:english_lowercase": _measure_english_lowercase,
    "punctuation:no_comma": _measure_no_comma,
    "startend:quotation": _measure_quotation,
    "copy:repeat_phrase": _measure_repeat_phrase,
    "copy:copy": _measure_copy,
    "new:copy_span_idx": _measure_copy_span_idx,
    "detectable_format:sentence_hyphens": _measure_sentence_hyphens,
    "keywords:no_adjacent_consecutive": _measure_no_adjacent_consecutive,
    "detectable_format:square_brackets": _measure_square_brackets,
    "keywords:word_once": _measure_word_once,
    "keywords:word_count_different_numbers": _measure_keyword_frequency,
    "keywords:exclude_word_harder": _measure_exclude_word_harder,
    "paragraphs:paragraphs": _measure_paragraphs_basic,
    "paragraphs:paragraphs2": _measure_paragraphs_basic2,
    "first_word:first_word_sent": _measure_first_word_sent,
    "first_word:first_word_answer": _measure_first_word_answer,
    "last_word:last_word_sent": _measure_last_word_sent,
    "last_word:last_word_answer": _measure_last_word_answer,
    "detectable_format:bigram_wrapping": _measure_bigram_wrapping,
    "copy:copying_simple": _measure_copy,
    "copy:copying_multiple": _measure_copying_multiple,
    "punctuation:punctuation_dot": _measure_punctuation_dot,
    "punctuation:punctuation_exclamation": _measure_punctuation_exclamation,
    "count:lowercase_counting": _measure_lowercase_counting,
    "letters:letter_counting": _measure_letter_counting,
    "letters:letter_counting2": _measure_letter_frequency,
    "count:counting_composition": _measure_counting_composition,
    "count:count_unique": _measure_count_unique,
    "count:count_increment_word": _measure_count_increment_word,
    "keywords:palindrome": _measure_palindrome,
    "keywords:keyword_specific_position": _measure_keyword_specific_position,
    "keywords:start_end": _measure_start_end,
}

assert set(_MEASUREMENTS) == set(FAMILIES), "every family must have a measurement function"


def describe_failure(c: Constraint, response: str) -> str:
    """Requirement sentence plus an observed-vs-required measurement, for repair prompts."""
    inst, desc = _build_with_description(c)
    measure = _MEASUREMENTS.get(c.iid)
    if measure is None:
        return f"Not satisfied: {desc}"
    try:
        ensure_nltk()
        observed = measure(inst, response)
    except Exception:
        return f"Not satisfied: {desc}"
    if not observed:
        return f"Not satisfied: {desc}"
    return f"{desc} Observed {observed}."

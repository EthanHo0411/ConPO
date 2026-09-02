"""Programmatic corruption operators for group C of the IFBench-Train families.

Each operator yields small, span-local edits (most minimal first) that flip its
own family's verifier while leaving every other constraint on the prompt
untouched; `registry.exact` is the final judge, this module is only a source
of candidates. Each fixture builds a natural response from the Constraint's
own kwargs so it passes the family's verifier by construction.

Family id                              | corruption strategy -> collateral risk
--------------------------------------- | ---------------------------------------
copy:repeat_phrase                     | change a second word in one repeated span (or drop/duplicate a whole span) -> can shift length_constraints:number_words/number_sentences if a span is dropped or duplicated.
copy:copy                              | drop, duplicate, or append one token to the copied text -> breaks any other family requiring the same exact text (combination:repeat_prompt) or counting its words.
new:copy_span_idx                      | same token-level edit as copy:copy, applied to the required span -> same risk as copy:copy.
detectable_format:sentence_hyphens     | add/remove one space around a hyphen, or drop a hyphen -> can merge or split words, tripping keywords:no_adjacent_consecutive or a word-count family.
keywords:no_adjacent_consecutive       | insert one real word whose first letter is exactly one past its neighbour's -> can bump length_constraints:number_words or duplicate a tracked keyword.
detectable_format:square_brackets      | strip the brackets off one word -> no format collateral, but shifts nothing else since only bracket characters change.
keywords:word_once                     | duplicate or delete the sole keyword occurrence -> breaks any other family counting or forbidding the same word (word_count_different_numbers, exclude_word_harder); the fixture's own filler prose ("artist", "approach") is also an untested self-collision risk if the keyword is a substring of one of those words (e.g. keyword="art").
keywords:word_count_different_numbers  | add or remove exactly one occurrence at the threshold boundary -> same keyword-sharing risk as word_once, including the same self-collision risk against its own filler prose ("orchard", "rows") for a keyword such as "row".
keywords:exclude_word_harder           | insert the banned word as a standalone token -> can trip length_constraints:number_words or land inside another format's markers.
paragraphs:paragraphs                  | remove the `***` divider (merge) or insert a second one (split into three) -> breaks length_constraints:number_paragraphs, which uses the same divider convention.
paragraphs:paragraphs2                 | remove a blank-line divider (merge) or insert an extra one (split into three) -> breaks length_constraints:nth_paragraph_first_word or paragraphs:paragraphs if `***` is also present.
first_word:first_word_sent             | replace or drop the first word of the final sentence -> breaks last_word:last_word_sent if that sentence is a single word.
first_word:first_word_answer           | replace, drop, or push back the response's very first word -> breaks first_word:first_word_sent (same opening token).
last_word:last_word_sent               | replace or drop the last word of the final sentence -> breaks last_word:last_word_answer (same trailing token).
last_word:last_word_answer             | replace, drop, or append after the final word -> breaks last_word:last_word_sent and any trailing-anchor family.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator

from conpo.operators.registry import register
from conpo.vendor.ifevalg import instructions_util
from conpo.verifiers import Constraint, build

# ---------------------------------------------------------------------------
# copy:repeat_phrase
# ---------------------------------------------------------------------------

_REPEAT_PHRASE_ALTS = ("since", "before", "after", "while", "until", "unless")
_REPEAT_PHRASE_FILLERS = (
    "during the quiet afternoon, which many people still remember fondly to this day",
    "long before the festival began, drawing visitors from every nearby town and village",
    "once the harvest ended, leaving farmers with time to rest and reflect on the season",
    "after the meeting closed, giving the team a moment to gather their scattered notes",
)


def _fixture_repeat_phrase(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    words = inst._phrase.split()
    center = len(words) // 2
    banned = {w.lower() for w in words}
    alts = [w for w in _REPEAT_PHRASE_ALTS if w not in banned] or list(_REPEAT_PHRASE_ALTS)
    blocks = []
    for i in range(inst._small_n):
        mutated = words.copy()
        mutated[center] = alts[i % len(alts)]
        blocks.append(" ".join(mutated))
    parts = [f"{block}, {_REPEAT_PHRASE_FILLERS[i % len(_REPEAT_PHRASE_FILLERS)]}." for i, block in enumerate(blocks)]
    return " ".join(parts)


def _op_repeat_phrase(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(c)
    words = inst._phrase.split()
    first_word, last_word = words[0], words[-1]
    pattern = re.compile(rf"{re.escape(first_word)} .*? {re.escape(last_word)}")
    matches = list(pattern.finditer(response))
    if not matches:
        return
    last = matches[-1]
    span_words = last.group().split()
    if len(span_words) == len(words):
        for i, (a, b) in enumerate(zip(span_words, words)):
            if a == b and i not in (0, len(words) - 1):
                mutated = span_words.copy()
                mutated[i] = "then" if mutated[i] != "then" else "now"
                yield response[: last.start()] + " ".join(mutated) + response[last.end() :]
                break
    start, end = last.start(), last.end()
    yield response[:end] + " " + last.group() + response[end:]  # duplicate the span
    yield response[:start] + response[end:]  # drop the whole span


register("copy:repeat_phrase", _op_repeat_phrase, _fixture_repeat_phrase)


# ---------------------------------------------------------------------------
# copy:copy and new:copy_span_idx share one exact-text corruption strategy:
# the fixture must equal a target string exactly (case-insensitively), so any
# single-token insertion, deletion, or duplication breaks it.
# ---------------------------------------------------------------------------


def _fixture_copy(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    return inst._prompt_to_repeat


def _fixture_copy_span_idx(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    return inst._prompt_to_repeat[inst._n_start : inst._n_end].strip()


def _op_exact_text(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    words = response.split()
    if not words:
        return
    if len(words) > 1:
        yield " ".join(words[:-1])
    yield response.rstrip() + " " + words[-1]
    yield response.rstrip() + " indeed"


register("copy:copy", _op_exact_text, _fixture_copy)
register("new:copy_span_idx", _op_exact_text, _fixture_copy_span_idx)


# ---------------------------------------------------------------------------
# detectable_format:sentence_hyphens
# ---------------------------------------------------------------------------


def _fixture_sentence_hyphens(c: Constraint, rng: random.Random) -> str:
    return (
        "The weather changed quickly this morning.-Clouds gathered over the distant hills.-"
        "Everyone rushed indoors before the rain started falling hard.-By evening the sky cleared "
        "up completely again.-Stars appeared one by one across the quiet horizon."
    )


def _op_sentence_hyphens(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    idx = response.find("-")
    if idx == -1:
        return
    yield response[: idx + 1] + " " + response[idx + 1 :]
    yield response[:idx] + " " + response[idx:]
    yield response[:idx] + response[idx + 1 :]


register("detectable_format:sentence_hyphens", _op_sentence_hyphens, _fixture_sentence_hyphens)


# ---------------------------------------------------------------------------
# keywords:no_adjacent_consecutive
# ---------------------------------------------------------------------------

_LETTER_WORD = {
    "a": "an", "b": "by", "c": "can", "d": "do", "e": "even", "f": "for", "g": "go",
    "h": "had", "i": "in", "j": "just", "k": "kept", "l": "like", "m": "may", "n": "now",
    "o": "of", "p": "per", "q": "quite", "r": "right", "s": "so", "t": "to", "u": "until",
    "v": "very", "w": "with", "x": "extra", "y": "yet", "z": "zero",
}


def _fixture_no_adjacent_consecutive(c: Constraint, rng: random.Random) -> str:
    return (
        "The old river wanders through broad valleys covered with tall grass near several wooden "
        "bridges. Warm evenings bring gentle winds that drift across open meadows and rustle golden "
        "leaves gently before nightfall. Farmers work steadily in wide fields harvesting crops before "
        "the evening chill arrives, while their children gather near the bridge to watch fish swim "
        "slowly across clear water, resting quietly on wooden benches until the calm afternoon light "
        "fades over everything."
    )


def _op_no_adjacent_consecutive(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    matches = list(re.finditer(r"\S+", response))
    if not matches:
        return
    first = matches[0].group()[0].lower()
    if first.isalpha() and first < "z":
        insert = _LETTER_WORD[chr(ord(first) + 1)]
        pos = matches[0].end()
        yield response[:pos] + " " + insert + response[pos:]
    if len(matches) >= 2:
        before_last = matches[-2].group()[0].lower()
        if before_last.isalpha() and before_last < "z":
            insert = _LETTER_WORD[chr(ord(before_last) + 1)]
            pos = matches[-1].start()
            yield response[:pos] + insert + " " + response[pos:]
    for i in range(1, len(matches) - 2):
        a, b = matches[i], matches[i + 1]
        fa, fb = a.group()[0].lower(), b.group()[0].lower()
        if fa.isalpha() and fb.isalpha() and ord(fa) - ord(fb) == 1:
            yield response[: a.start()] + b.group() + response[a.end() : b.start()] + a.group() + response[b.end() :]
            break


register("keywords:no_adjacent_consecutive", _op_no_adjacent_consecutive, _fixture_no_adjacent_consecutive)


# ---------------------------------------------------------------------------
# detectable_format:square_brackets
# ---------------------------------------------------------------------------


def _fixture_square_brackets(c: Constraint, rng: random.Random) -> str:
    return (
        "[The] [weather] [today] [is] [absolutely] [wonderful] [and] [everyone] [in] [the] "
        "[neighborhood] [decided] [to] [spend] [the] [afternoon] [outside] [enjoying] [the] [warm] "
        "[sunshine] [while] [children] [played] [happily] [in] [the] [park] [nearby] [until] "
        "[dinner] [time] [arrived] [for] [every] [family.]"
    )


def _unwrap(response: str, matches: list[re.Match[str]], index: int) -> str:
    token = matches[index].group()
    if token.startswith("["):
        token = token[1:]
    if token.endswith("]"):
        token = token[:-1]
    start, end = matches[index].start(), matches[index].end()
    return response[:start] + token + response[end:]


def _op_square_brackets(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    matches = list(re.finditer(r"\S+", response))
    if not matches:
        return
    yield _unwrap(response, matches, len(matches) // 2)
    yield _unwrap(response, matches, 0)
    yield _unwrap(response, matches, len(matches) - 1)


register("detectable_format:square_brackets", _op_square_brackets, _fixture_square_brackets)


# ---------------------------------------------------------------------------
# keywords:word_once
# ---------------------------------------------------------------------------


def _fixture_word_once(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    kw = inst._keyword
    return (
        f"Every artist develops a {kw} approach to color and composition over years of dedicated "
        "practice. Museums across the world showcase paintings that reveal how technique evolves "
        "alongside cultural shifts and available materials. Students often study these historical "
        "works closely, learning to appreciate brushwork, perspective, and the patient discipline "
        "required to master any visual craft."
    )


def _collapse_junction_space(text: str, pos: int) -> str:
    """After deleting a span, collapse a double space at exactly `pos` (the
    deletion junction) to one -- never a global replace that could touch an
    unrelated double space elsewhere in the response."""
    if 0 < pos < len(text) and text[pos - 1 : pos + 1] == "  ":
        return text[: pos - 1] + text[pos:]
    return text


def _op_word_once(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(c)
    match = re.search(inst._keyword, response, flags=re.IGNORECASE)
    if match is None:
        return
    yield response[: match.end()] + " " + match.group() + response[match.end() :]
    deleted = response[: match.start()] + response[match.end() :]
    yield _collapse_junction_space(deleted, match.start())


register("keywords:word_once", _op_word_once, _fixture_word_once)


# ---------------------------------------------------------------------------
# keywords:word_count_different_numbers
# ---------------------------------------------------------------------------


def _fixture_word_count_different(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    kw = inst._keyword
    target = inst._frequency if inst._comparison_relation == "at least" else max(inst._frequency - 1, 0)
    kw_sentence = f"The orchard grew rows of {kw} trees that produced fruit every single autumn season."
    filler = (
        "Workers filled crates quickly, sorting the fresh produce by size and color before loading "
        "trucks bound for the regional market downtown. Local children often stopped to admire the "
        "blossoming rows on their way to school, chatting happily about the coming harvest festival."
    )
    sentences = [kw_sentence] * target
    return (" ".join(sentences) + (" " if sentences else "") + filler).strip()


def _op_word_count_different(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(c)
    matches = list(re.finditer(inst._keyword, response, flags=re.IGNORECASE))
    if inst._comparison_relation == "at least":
        if not matches:
            return
        first = matches[0]
        deleted = response[: first.start()] + response[first.end() :]
        yield _collapse_junction_space(deleted, first.start())
        if len(matches) > 1:
            last = matches[-1]
            deleted = response[: last.start()] + response[last.end() :]
            yield _collapse_junction_space(deleted, last.start())
    else:
        if matches:
            last = matches[-1]
            yield response[: last.end()] + " " + last.group() + response[last.end() :]
        else:
            yield response.rstrip() + f" {inst._keyword}."


register("keywords:word_count_different_numbers", _op_word_count_different, _fixture_word_count_different)


# ---------------------------------------------------------------------------
# keywords:exclude_word_harder
# ---------------------------------------------------------------------------


def _fixture_exclude_word_harder(c: Constraint, rng: random.Random) -> str:
    build(c)  # validates kwargs; the banned word is deliberately absent below
    return (
        "The market stalls overflowed with fresh produce every weekend, drawing crowds eager to fill "
        "their baskets with seasonal fruit and vegetables. Vendors called out prices while children "
        "weaved between the tables, and the smell of ripe peaches and citrus filled the warm morning "
        "air across the entire square."
    )


def _op_exclude_word_harder(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(c)
    # A space with a non-space char on each side, i.e. a genuine single-space
    # word boundary, never a paragraph divider ("\n\n") or a double space.
    spaces = [m.start() for m in re.finditer(r"(?<=\S) (?=\S)", response)]
    if not spaces:
        return
    mid = spaces[len(spaces) // 2]
    yield response[:mid] + " " + inst._keyword + response[mid:]
    first = spaces[0]
    yield response[:first] + " " + inst._keyword + response[first:]


register("keywords:exclude_word_harder", _op_exclude_word_harder, _fixture_exclude_word_harder)


# ---------------------------------------------------------------------------
# paragraphs:paragraphs (*** divider) and paragraphs:paragraphs2 (blank line)
# ---------------------------------------------------------------------------


def _fixture_paragraphs(c: Constraint, rng: random.Random) -> str:
    return (
        "First paragraph text describing the morning routine of a small coastal village where "
        "fishermen prepare their boats before dawn breaks over the harbor.\n\n***\n\nSecond paragraph "
        "text continuing the story as the market opens and traders arrange their stalls along the "
        "busy waterfront street."
    )


def _fixture_paragraphs2(c: Constraint, rng: random.Random) -> str:
    return (
        "First paragraph text describing the morning routine of a small coastal village where "
        "fishermen prepare their boats before dawn breaks over the harbor.\n\nSecond paragraph text "
        "continuing the story as the market opens and traders arrange their stalls along the busy "
        "waterfront street."
    )


def _op_paragraphs(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    if "***" not in response:
        return
    yield re.sub(r"\s?\*\*\*\s?", " ", response, count=1)
    mid = len(response) // 2
    space = response.find(" ", mid)
    if space != -1:
        yield response[:space] + "\n\n***\n\n" + response[space + 1 :]


def _op_paragraphs2(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    if "\n\n" not in response:
        return
    yield response.replace("\n\n", " ", 1)
    mid = len(response) // 2
    space = response.find(" ", mid)
    if space != -1:
        yield response[:space] + "\n\n" + response[space + 1 :]


register("paragraphs:paragraphs", _op_paragraphs, _fixture_paragraphs)
register("paragraphs:paragraphs2", _op_paragraphs2, _fixture_paragraphs2)


# ---------------------------------------------------------------------------
# first_word:first_word_sent and first_word:first_word_answer
# ---------------------------------------------------------------------------


def _fixture_first_word_sent(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    w = inst._first_word.capitalize()
    tails = (
        "the storm arrived earlier than forecasters expected, catching several boats still far from shore.",
        "the crew managed to secure the deck and radio for help before conditions worsened further.",
        "calm returned by midnight, and every sailor made it home safely to their relieved families.",
    )
    return " ".join(f"{w} {t}" for t in tails)


def _fixture_first_word_answer(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    w = inst._first_word.capitalize()
    return (
        f"{w} unlikely it seemed at first, the small research team eventually confirmed their initial "
        "hypothesis after months of careful experimentation and repeated trials across several "
        "independent laboratories worldwide, each contributing data that steadily strengthened the "
        "overall confidence in the final result."
    )


def _op_first_word_sent(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    sentences = [s for s in instructions_util.split_into_sentences(response) if s.strip()]
    if not sentences:
        return
    target = sentences[-1]
    idx = response.rfind(target)
    if idx == -1:
        return
    words = target.split()
    if not words:
        return
    yield response[:idx] + " ".join(["Meanwhile", *words[1:]]) + response[idx + len(target) :]
    yield response[:idx] + " ".join(words[1:]) + response[idx + len(target) :]


def _op_first_word_answer(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    match = re.match(r"(\s*)(\S+)(\s*)", response)
    if match is None or not match.group(2):
        return
    lead, sep, rest = match.group(1), match.group(3), response[match.end() :]
    yield lead + "Meanwhile" + sep + rest  # replace first word, everything after is untouched
    yield lead + rest  # drop first word and its trailing separator only
    yield "Well, " + response


register("first_word:first_word_sent", _op_first_word_sent, _fixture_first_word_sent)
register("first_word:first_word_answer", _op_first_word_answer, _fixture_first_word_answer)


# ---------------------------------------------------------------------------
# last_word:last_word_sent and last_word:last_word_answer
# ---------------------------------------------------------------------------


def _fixture_last_word_sent(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    w = inst._last_word
    heads = (
        "Reporters gathered outside the courthouse early",
        "Officials confirmed the announcement would happen",
        "Residents across the city waited anxiously to hear the news",
    )
    return " ".join(f"{h} {w}." for h in heads)


def _fixture_last_word_answer(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    w = inst._last_word
    return (
        "Reporters gathered outside the courthouse before sunrise, hoping to catch officials arriving "
        "for the long awaited announcement that had been delayed several times over the past two "
        f"weeks, finally scheduled for release sometime {w}"
    )


def _last_token_split(sentence: str) -> tuple[list[str], str]:
    """Split a sentence into (words, trailing-punctuation-of-last-word)."""
    words = sentence.split()
    if not words:
        return words, ""
    last = words[-1]
    core = re.sub(r"[^\w\s]", "", last)
    suffix = last[len(core) :] if last.startswith(core) else ""
    words[-1] = core
    return words, suffix


def _op_last_word_sent(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    sentences = [s for s in instructions_util.split_into_sentences(response) if s.strip()]
    if not sentences:
        return
    target = sentences[-1]
    idx = response.rfind(target)
    if idx == -1:
        return
    words, suffix = _last_token_split(target)
    if not words:
        return
    yield response[:idx] + " ".join([*words[:-1], "yesterday"]) + suffix + response[idx + len(target) :]
    yield response[:idx] + " ".join(words[:-1]) + suffix + response[idx + len(target) :]


def _op_last_word_answer(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    match = re.search(r"(\s*)(\S+)(\s*)$", response)
    if match is None or not match.group(2):
        return
    head, lead, trail = response[: match.start()], match.group(1), match.group(3)
    yield head + lead + "yesterday" + trail  # replace last word, everything before is untouched
    yield head + trail  # drop last word and the separator before it only
    yield response.rstrip() + " indeed"


register("last_word:last_word_sent", _op_last_word_sent, _fixture_last_word_sent)
register("last_word:last_word_answer", _op_last_word_answer, _fixture_last_word_answer)

"""Corruption operators and fixtures for the ifeval_a family group (12 families).

Ported from stackelberg-if/probe_ifrepair.py (lines 277-607) to the candidate
iterator contract in registry.py: each operator yields candidate corruptions
(most minimal edit first) instead of returning one string, and takes the
Constraint plus build(c) instead of a pre-built checker instance.

Corruption strategy per family and its collateral risk (which other family in
this repertoire the same edit is most likely to also break):

  keywords:existence                    replace one present keyword with a neutral word.
                                         risk: another constraint counting that same word (its own
                                         keywords:frequency, or a shared letter_frequency letter).
  keywords:frequency                    drop/add just enough occurrences of the keyword.
                                         risk: keywords:existence or forbidden_words tracking the same
                                         literal word; letter_frequency on a letter inside it.
  keywords:forbidden_words              insert one forbidden word into the response.
                                         risk: number_words if the insertion crosses a word-count edge;
                                         constrained_response if the forbidden word lands inside its
                                         matched option string.
  keywords:letter_frequency             add/remove occurrences of the target letter.
                                         risk: any other letter_frequency or change_case constraint
                                         counting the same letter class.
  length_constraints:number_sentences   merge two sentences at a boundary (comma-join, keep every
                                         word) to drop the count, or append filler sentences to
                                         raise it. risk: number_paragraphs or nth_paragraph_first_word
                                         if the merged/added boundary sits at a paragraph divider.
  length_constraints:number_paragraphs  merge two paragraphs by removing one *** divider, or add one.
                                         risk: nth_paragraph_first_word (paragraph indices shift),
                                         number_bullet_lists if bullets sit in the merged paragraph.
  length_constraints:number_words       truncate/extend by one word at a token boundary.
                                         risk: number_sentences if the cut removes terminal punctuation,
                                         keywords:existence/forbidden_words if the cut removes a
                                         tracked word.
  length_constraints:nth_paragraph_first_word  swap only the first word of the target paragraph.
                                         risk: keywords:existence/forbidden_words if that word happens
                                         to be a tracked keyword.
  detectable_content:number_placeholders  unwrap or delete one [bracket] placeholder.
                                         risk: none structural; the surrounding sentence is untouched.
  detectable_content:postscript         strip the postscript line, or rename its marker in place.
                                         risk: keywords/letter_frequency constraints counting text that
                                         lives inside the postscript line.
  detectable_format:number_bullet_lists  merge two bullets into one line, or unmark one bullet.
                                         risk: number_paragraphs if the merge crosses a *** divider,
                                         number_words if the merge changes the word count.
  detectable_format:constrained_response  alter the matched option string in place.
                                         risk: keywords/forbidden_words constraints anchored to the
                                         same phrase; number_words if the edit changes word count.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Iterator

from conpo.operators.registry import register
from conpo.verifiers import Constraint, build, ensure_nltk
from conpo.vendor.ifevalg import instructions_util

# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

_NEUTRAL_WORDS: tuple[str, ...] = ("thing", "item", "stuff", "matter")

_FILLER_SENTENCES: tuple[str, ...] = (
    "The garden stayed quiet through the whole afternoon.",
    "Local volunteers spent the morning tending the winding paths.",
    "Visitors often rest beneath the old oak tree near the gate.",
    "A gentle breeze moved slowly across the open field.",
    "Someone had left a small basket of tools by the bench.",
    "The librarian recommended a new book about local history.",
    "Children played near the fountain until the evening light faded.",
    "Every Tuesday the market opens early near the town square.",
)


def _match_case(word: str, like: str) -> str:
    """Case-match `word` to the casing of the substring it is replacing."""
    if like.isupper():
        return word.upper()
    if like[:1].isupper():
        return word.capitalize()
    return word


def _adapt_case(fragment: str, text: str) -> str:
    """Case-match `fragment` to the overall casing of `text`."""
    if text.isupper():
        return fragment.upper()
    if text.islower():
        return fragment.lower()
    return fragment


def _neutral_for(pattern: str) -> str | None:
    """A neutral word that the (unescaped, checker-style) `pattern` does not match."""
    for word in _NEUTRAL_WORDS:
        try:
            if not re.search(pattern, word, flags=re.IGNORECASE):
                return word
        except re.error:
            return None
    return None


def _filler_text(rng: random.Random, n_sentences: int = 5) -> str:
    """A shuffled selection of neutral filler sentences, ~40-70 words."""
    pool = list(_FILLER_SENTENCES)
    rng.shuffle(pool)
    return " ".join(pool[:n_sentences])


def _sentence_tokenizer():
    """The exact punkt tokenizer the vendored checker uses for sentence counts."""
    ensure_nltk()
    return instructions_util._get_sentence_tokenizer()


# ---------------------------------------------------------------------------
# keywords:existence
# ---------------------------------------------------------------------------


def _fixture_keyword_existence(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    order = list(inst._keywords)
    rng.shuffle(order)
    mention = "This paragraph also brings up " + ", ".join(order) + " for extra context."
    return _filler_text(rng) + " " + mention


def _op_keyword_existence(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    present = [kw for kw in inst._keywords if re.search(kw, text, flags=re.IGNORECASE)]
    if not present:
        return
    # most minimal first: drop the keyword with the fewest occurrences
    present.sort(key=lambda kw: len(re.findall(kw, text, flags=re.IGNORECASE)))
    for kw in present:
        repl = _neutral_for(kw)
        if repl is None:
            continue
        yield re.sub(kw, lambda m: _match_case(repl, m.group(0)), text, flags=re.IGNORECASE)


register("keywords:existence", _op_keyword_existence, _fixture_keyword_existence)

# ---------------------------------------------------------------------------
# keywords:frequency
# ---------------------------------------------------------------------------


def _fixture_keyword_frequency(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    keyword, freq, relation = inst._keyword, inst._frequency, inst._comparison_relation
    body = _filler_text(rng)
    count = len(re.findall(keyword, body, flags=re.IGNORECASE))
    if relation == "at least":
        need = max(0, freq - count)
        if need:
            body += " " + _adapt_case(" ".join([keyword] * need) + ".", body)
    else:
        # a literal replacement can itself contain the keyword (e.g. keyword="the"
        # inside "the topic"), reintroducing a match on every substitution and
        # looping forever; _neutral_for guarantees the replacement does not match
        # the keyword pattern, and the iteration cap is a hard backstop regardless.
        repl = _neutral_for(keyword) or _NEUTRAL_WORDS[0]
        remaining = 200
        while remaining and len(re.findall(keyword, body, flags=re.IGNORECASE)) >= freq:
            body = re.sub(keyword, repl, body, count=1, flags=re.IGNORECASE)
            remaining -= 1
    return body


def _op_keyword_frequency(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    keyword, freq, relation = inst._keyword, inst._frequency, inst._comparison_relation
    matches = list(re.finditer(keyword, text, flags=re.IGNORECASE))
    if relation == "at least":
        need = len(matches) - freq + 1
        if need < 1 or need > len(matches):
            return
        repl = _neutral_for(keyword)
        if repl is None:
            return
        out = text
        for m in reversed(matches[-need:]):
            out = out[: m.start()] + _match_case(repl, m.group(0)) + out[m.end() :]
        yield out
        if need < len(matches):
            out_all = text
            for m in reversed(matches):
                out_all = out_all[: m.start()] + _match_case(repl, m.group(0)) + out_all[m.end() :]
            yield out_all
    else:
        need = freq - len(matches)
        if need < 1 or need > 50:
            return
        out = text + " " + _adapt_case(" ".join([keyword] * need) + ".", text)
        if len(re.findall(keyword, out, flags=re.IGNORECASE)) >= freq:
            yield out
        out_extra = text + " " + _adapt_case(" ".join([keyword] * (need + 2)) + ".", text)
        if len(re.findall(keyword, out_extra, flags=re.IGNORECASE)) >= freq:
            yield out_extra


register("keywords:frequency", _op_keyword_frequency, _fixture_keyword_frequency)

# ---------------------------------------------------------------------------
# keywords:forbidden_words
# ---------------------------------------------------------------------------


def _fixture_forbidden_words(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    banned = [re.compile(r"\b" + word + r"\b", flags=re.IGNORECASE) for word in inst._forbidden_words]
    pool = [s for s in _FILLER_SENTENCES if not any(b.search(s) for b in banned)]
    if not pool:
        pool = list(_NEUTRAL_WORDS)
    rng.shuffle(pool)
    return " ".join(pool[: min(5, len(pool))])


def _op_forbidden_words(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    words = sorted(inst._forbidden_words, key=len)
    if not words:
        return
    # candidate 1: append the shortest forbidden word (minimal insertion)
    yield text + " " + _adapt_case(words[0], text) + "."
    # candidate 2: insert mid-sentence instead of appending
    idx = text.find(". ")
    if idx != -1:
        yield text[: idx + 2] + _adapt_case(words[0], text) + ". " + text[idx + 2 :]


register("keywords:forbidden_words", _op_forbidden_words, _fixture_forbidden_words)

# ---------------------------------------------------------------------------
# keywords:letter_frequency
# ---------------------------------------------------------------------------


def _trim_letter_to(body: str, letter: str, target: int) -> str:
    """Delete just enough occurrences of `letter` to bring its count down to `target`.

    Character-level, applied only while building a fixture (never on the
    corrupted side), so the boundary is exact and the later operator edit
    (add/remove one occurrence) stays a single-character change.
    """
    count = Counter(body.lower())[letter]
    to_drop = max(0, count - target)
    if not to_drop:
        return body
    out_chars = []
    dropped = 0
    for ch in body:
        if dropped < to_drop and ch.lower() == letter:
            dropped += 1
            continue
        out_chars.append(ch)
    return "".join(out_chars)


def _fixture_letter_frequency(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    letter, freq, relation = inst._letter, inst._frequency, inst._comparison_relation
    body = _filler_text(rng)
    if relation == "at least":
        # trim to exactly `freq` occurrences: no slack, so the operator only
        # needs to remove one to break the constraint.
        body = _trim_letter_to(body, letter, freq)
        count = Counter(body.lower())[letter]
        need = max(0, freq - count)
        if need:
            body += " " + _adapt_case(letter * need + ".", body)
    else:
        allowed = max(0, freq - 1)
        body = _trim_letter_to(body, letter, allowed)
    return body


def _op_letter_frequency(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    letter, freq, relation = inst._letter, inst._frequency, inst._comparison_relation
    count = Counter(text.lower())[letter]
    if relation == "at least":
        need = count - freq + 1
        if need < 1 or need > count:
            return
        # candidate 1: strip the letter from the end of the text
        chars = list(text)
        removed = 0
        for i in range(len(chars) - 1, -1, -1):
            if chars[i].lower() == letter:
                del chars[i]
                removed += 1
                if removed == need:
                    break
        yield "".join(chars)
        # candidate 2: strip the letter from the start instead
        chars2 = list(text)
        removed = 0
        for i in range(len(chars2)):
            if chars2[i].lower() == letter:
                del chars2[i]
                removed += 1
                if removed == need:
                    break
        yield "".join(chars2)
    else:
        need = freq - count
        if need < 1 or need > 200:
            return
        yield text + " " + _adapt_case(letter * need + ".", text)
        yield text + " " + _adapt_case(" ".join([letter] * need) + ".", text)


register("keywords:letter_frequency", _op_letter_frequency, _fixture_letter_frequency)

# ---------------------------------------------------------------------------
# length_constraints:number_sentences
# ---------------------------------------------------------------------------


def _fixture_number_sentences(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    threshold, relation = inst._num_sentences_threshold, inst._comparison_relation
    pool = list(_FILLER_SENTENCES)
    rng.shuffle(pool)
    n = max(threshold, 4) if relation == "at least" else max(threshold - 1, 1)
    return " ".join(pool[i % len(pool)] for i in range(n))


def _merge_sentence_boundary(left: str, right: str) -> str:
    """Join two sentences into one: drop `left`'s terminal punctuation, join with
    a comma, lowercase `right`'s first letter. Keeps every word, so the sentence
    tokenizer sees one sentence where it saw two, at the cost of two characters.
    """
    core = re.sub(r"[.!?]+\s*$", "", left.rstrip())
    return core + ", " + right[:1].lower() + right[1:]


def _op_number_sentences(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    threshold, relation = inst._num_sentences_threshold, inst._comparison_relation
    sentences = [s for s in _sentence_tokenizer().tokenize(text) if s.strip()]
    if relation == "at least":
        # merge sentence boundaries (word-preserving) instead of truncating:
        # dropping whole sentences removes their words wholesale, blowing past
        # the edit-distance budget when the fixture has any slack above threshold.
        need = len(sentences) - threshold + 1
        if need < 1 or need >= len(sentences):
            return
        # candidate 1: merge the last `need` boundaries
        merged = list(sentences)
        for _ in range(need):
            last = merged.pop()
            merged[-1] = _merge_sentence_boundary(merged[-1], last)
        yield " ".join(merged)
        # candidate 2: merge the first `need` boundaries instead
        merged2 = list(sentences)
        for _ in range(need):
            first = merged2.pop(0)
            merged2[0] = _merge_sentence_boundary(first, merged2[0])
        yield " ".join(merged2)
    else:
        need = threshold - len(sentences)
        if need < 1 or need > 50:
            return
        filler = "It goes on."
        yield text + " " + _adapt_case(" ".join([filler] * need), text)
        yield text + " " + _adapt_case(" ".join([filler] * (need + 1)), text)


register("length_constraints:number_sentences", _op_number_sentences, _fixture_number_sentences)

# ---------------------------------------------------------------------------
# length_constraints:number_paragraphs
# ---------------------------------------------------------------------------


def _fixture_number_paragraphs(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    num_paragraphs = max(inst._num_paragraphs, 1)
    pool = list(_FILLER_SENTENCES)
    rng.shuffle(pool)
    paras = []
    for i in range(num_paragraphs):
        a, b = pool[(2 * i) % len(pool)], pool[(2 * i + 1) % len(pool)]
        paras.append(a + " " + b)
    return "\n***\n".join(paras)


def _op_number_paragraphs(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    matches = list(re.finditer(r"\s?\*\*\*\s?", text))
    if matches:
        # candidate 1: merge the last two paragraphs (drop the last divider)
        m = matches[-1]
        yield text[: m.start()] + "\n" + text[m.end() :]
        # candidate 2: merge the first two paragraphs instead
        if len(matches) > 1:
            m0 = matches[0]
            yield text[: m0.start()] + "\n" + text[m0.end() :]
    # candidate: add one more paragraph
    yield text + "\n***\n" + _adapt_case("And one more paragraph follows.", text)


register("length_constraints:number_paragraphs", _op_number_paragraphs, _fixture_number_paragraphs)

# ---------------------------------------------------------------------------
# length_constraints:number_words
# ---------------------------------------------------------------------------

_WORD_POOL: tuple[str, ...] = (
    "the", "quiet", "garden", "stayed", "calm", "through", "afternoon", "while", "visitors",
    "wandered", "slowly", "past", "an", "old", "oak", "tree", "near", "open", "gate", "and",
    "small", "basket", "sat", "beside", "wooden", "bench", "every", "morning", "volunteers",
    "returned", "to", "tend", "narrow", "paths", "along", "fence", "line", "before", "noon",
)


def _fixture_number_words(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    num_words, relation = inst._num_words, inst._comparison_relation
    n = max(num_words, 1) if relation == "at least" else max(num_words - 1, 1)
    start = rng.randrange(len(_WORD_POOL))
    chosen = [_WORD_POOL[(start + i) % len(_WORD_POOL)] for i in range(n)]
    return " ".join(chosen) + "."


def _op_number_words(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    words = list(re.finditer(r"\w+", text))
    num, relation = inst._num_words, inst._comparison_relation
    if relation == "at least":
        keep = num - 1
        if keep < 1 or keep > len(words) - 1:
            return
        yield text[: words[keep - 1].end()]
        if keep - 2 >= 1:
            yield text[: words[keep - 2].end()]
    else:
        need = num - len(words)
        if need < 1 or need > 500:
            return
        yield text + " " + _adapt_case(("more " * need).strip() + ".", text)
        yield text + " " + _adapt_case(("more " * (need + 3)).strip() + ".", text)


register("length_constraints:number_words", _op_number_words, _fixture_number_words)

# ---------------------------------------------------------------------------
# length_constraints:nth_paragraph_first_word
# ---------------------------------------------------------------------------


def _fixture_nth_paragraph_first_word(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    num_paragraphs, nth, first_word = inst._num_paragraphs, inst._nth_paragraph, inst._first_word
    pool = list(_FILLER_SENTENCES)
    rng.shuffle(pool)
    paras = []
    for i in range(num_paragraphs):
        a, b = pool[(2 * i) % len(pool)], pool[(2 * i + 1) % len(pool)]
        para = a + " " + b
        if i == nth - 1:
            words = para.split()
            words[0] = first_word.capitalize()
            para = " ".join(words)
        paras.append(para)
    return "\n\n".join(paras)


def _op_nth_paragraph_first_word(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    paras = text.split("\n\n")
    i = inst._nth_paragraph - 1
    if i < 0 or i >= len(paras) or not paras[i].strip():
        return
    m = re.search(r"\w+", paras[i])
    if m is None:
        return
    for repl in ("meanwhile", "however", "instead"):
        if repl == inst._first_word:
            continue
        new_paras = list(paras)
        new_paras[i] = paras[i][: m.start()] + _match_case(repl, m.group()) + paras[i][m.end() :]
        yield "\n\n".join(new_paras)


register(
    "length_constraints:nth_paragraph_first_word",
    _op_nth_paragraph_first_word,
    _fixture_nth_paragraph_first_word,
)

# ---------------------------------------------------------------------------
# detectable_content:number_placeholders
# ---------------------------------------------------------------------------

_PLACEHOLDER_NAMES: tuple[str, ...] = ("name", "date", "location", "time", "address", "phone", "email", "city")


def _fixture_number_placeholders(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    n = max(inst._num_placeholders, 0)
    names = list(_PLACEHOLDER_NAMES)
    rng.shuffle(names)
    chosen = [names[i % len(names)] for i in range(n)]
    body = _filler_text(rng)
    mention = "Please fill in " + ", ".join(f"[{w}]" for w in chosen) + " before sending." if chosen else ""
    return (body + " " + mention).strip()


def _op_number_placeholders(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    matches = list(re.finditer(r"\[.*?\]", text))
    need = len(matches) - inst._num_placeholders + 1
    if need < 1 or need > len(matches):
        return
    # candidate 1: unwrap just enough placeholders (minimal, keeps the words)
    out = text
    for m in reversed(matches[-need:]):
        out = out[: m.start()] + m.group()[1:-1] + out[m.end() :]
    yield out
    # candidate 2: delete the placeholder text entirely
    out_delete = text
    for m in reversed(matches[-need:]):
        out_delete = out_delete[: m.start()] + out_delete[m.end() :]
    yield out_delete


register("detectable_content:number_placeholders", _op_number_placeholders, _fixture_number_placeholders)

# ---------------------------------------------------------------------------
# detectable_content:postscript
# ---------------------------------------------------------------------------


def _fixture_postscript(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    marker = inst._postscript_marker
    body = _filler_text(rng)
    return body + "\n\n" + marker + " Thanks for reading this note."


def _postscript_pattern(marker: str) -> str:
    if marker == "P.P.S":
        return r"\s*p\.\s?p\.\s?s.*$"
    if marker == "P.S.":
        return r"\s*p\.\s?s\..*$"
    return r"\s*" + re.escape(marker.lower()) + r".*$"


def _op_postscript(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    marker = inst._postscript_marker
    pattern = _postscript_pattern(marker)
    # candidate 1: rename just the marker token, keep the trailing sentence (most minimal)
    if marker in text:
        yield text.replace(marker, "Note:", 1)
    # candidate 2: strip the whole postscript line (fallback, larger edit)
    out = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)
    if out != text:
        yield out


register("detectable_content:postscript", _op_postscript, _fixture_postscript)

# ---------------------------------------------------------------------------
# detectable_format:number_bullet_lists
# ---------------------------------------------------------------------------

_BULLET_LINE = re.compile(r"\s*(\*[^\*]|-)")
_BULLET_ITEMS: tuple[str, ...] = (
    "Check the schedule",
    "Confirm the address",
    "Bring a copy of the form",
    "Notify the front desk",
    "Review the checklist",
    "Pack the necessary documents",
)


def _fixture_number_bullet_lists(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    n = max(inst._num_bullets, 0)
    items = list(_BULLET_ITEMS)
    rng.shuffle(items)
    chosen = [items[i % len(items)] for i in range(n)]
    intro = "Here is a short checklist to review carefully before the appointment tomorrow:"
    outro = "Please confirm everything on this list before you leave the office today."
    bullets = "\n".join(f"* {t}." for t in chosen)
    parts = [intro] + ([bullets] if bullets else []) + [outro]
    return "\n".join(parts)


def _op_number_bullet_lists(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if _BULLET_LINE.match(ln)]
    if len(idx) >= 2:
        # candidate 1: merge the last two bullets into one line
        i, j = idx[-1], idx[-2]
        merged = list(lines)
        body = re.sub(r"^\s*(\*|-)\s*", "", merged[i])
        merged[j] = merged[j].rstrip() + " and " + body
        del merged[i]
        yield "\n".join(merged)
        # candidate 2: strip the marker off one bullet, turning it into prose
        stripped = list(lines)
        stripped[idx[-1]] = re.sub(r"^\s*(\*|-)\s*", "", stripped[idx[-1]])
        yield "\n".join(stripped)
    elif len(idx) == 1:
        stripped = list(lines)
        stripped[idx[0]] = re.sub(r"^\s*(\*|-)\s*", "", stripped[idx[0]])
        yield "\n".join(stripped)


register("detectable_format:number_bullet_lists", _op_number_bullet_lists, _fixture_number_bullet_lists)

# ---------------------------------------------------------------------------
# detectable_format:constrained_response
# ---------------------------------------------------------------------------


def _fixture_constrained_response(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    options = list(inst._constrained_responses)
    choice = options[rng.randrange(len(options))]
    return choice + " " + _filler_text(rng)


def _op_constrained_response(text: str, constraint: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(constraint)
    options = list(inst._constrained_responses)
    present = [opt for opt in options if opt in text]
    if not present:
        return
    # candidate 1: swap "answer" for "reply" inside each matched option
    out = text
    for opt in present:
        out = out.replace(opt, opt.replace("answer", "reply"))
    if out != text:
        yield out
    # candidate 2: drop the trailing period so the exact option no longer matches
    out_trim = text
    for opt in present:
        if opt.endswith("."):
            out_trim = out_trim.replace(opt, opt[:-1])
    if out_trim != text:
        yield out_trim


register("detectable_format:constrained_response", _op_constrained_response, _fixture_constrained_response)

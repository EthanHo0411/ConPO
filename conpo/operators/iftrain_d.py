"""Programmatic corruption operators for group D of the IFBench-Train families.

Each operator yields small, span-local edits (most minimal first) that flip its
own family's verifier while leaving every other constraint on the prompt
untouched; `registry.exact` is the final judge, this module is only a source
of candidates. Each fixture builds a response from the Constraint's own kwargs
so it passes the family's verifier by construction.

Family id                              | corruption strategy -> collateral risk
--------------------------------------- | ---------------------------------------
detectable_format:bigram_wrapping      | strip "<<" or ">>" off one wrapped bigram -> no structural collateral, only the two bracket characters change.
copy:copying_simple                    | drop, duplicate, or append a token to the copied text -> breaks any other family requiring the same exact text (combination:repeat_prompt) or counting its words.
copy:copying_multiple                  | token-edit inside one copy, break the "******" separator, or duplicate a whole copy -> breaks combination:repeat_prompt or a word-count family reading the same text.
punctuation:punctuation_dot            | insert one "." at the start, middle, or end -> breaks punctuation:punctuation_exclamation only if the inserted text also contains "!" (it does not).
punctuation:punctuation_exclamation    | insert one "!" at the start, middle, or end -> breaks punctuation:punctuation_dot only if the inserted text also contains "." (it does not).
count:lowercase_counting               | lowercase one Title-Cased word -> breaks change_case-style families tracking capitalization of that same word.
letters:letter_counting                | add/remove exactly one letter at the boundary -> breaks any other letters/keywords family counting total or specific letters.
letters:letter_counting2               | add/remove one occurrence of the target letter at the boundary -> breaks any other letter-frequency family on the same letter.
count:counting_composition             | insert/drop one filler word in the first sentence, merge the first "***" divider, or duplicate the first sentence -> breaks paragraphs/length_constraints families reading the same paragraph or sentence boundary.
count:count_unique                     | duplicate an existing word (or its trailing punctuation) -> no structural collateral, the duplicate is additive only.
count:count_increment_word             | add/remove one occurrence of keyword1 or keyword2 at its boundary count -> breaks any other keyword-counting family tracking the same word.
keywords:palindrome                    | replace, delete, or punctuate the sole palindrome token -> breaks any family requiring that exact token elsewhere (a shared copy/keyword family).
keywords:keyword_specific_position     | replace, delete, or shift the keyword at its required sentence position -> breaks a keyword-existence family expecting that literal word.
keywords:start_end                     | replace the first or last word, or append trailing punctuation -> breaks first_word/last_word families anchored to the same token.
"""

from __future__ import annotations

import itertools
import random
import re
from collections.abc import Iterator

from conpo.operators.registry import register
from conpo.verifiers import Constraint, build
from conpo.vendor.ifevalg import instructions_util

# ---------------------------------------------------------------------------
# detectable_format:bigram_wrapping
# ---------------------------------------------------------------------------

_BIGRAM_WORDS: tuple[str, ...] = (
    "This", "report", "covers", "recent", "weather", "patterns", "across", "the", "northern", "region",
    "during", "early", "spring", "months", "while", "farmers", "prepare", "fields", "for", "planting",
    "season", "ahead", "of", "expected", "rainfall", "increases", "later", "this", "year", "according",
    "to", "regional", "climate", "analysts", "who", "study", "long", "term", "trends", "carefully",
)


def _fixture_bigram_wrapping(c: Constraint, rng: random.Random) -> str:
    build(c)
    words = list(_BIGRAM_WORDS)
    if len(words) % 2:
        words.append("today")
    pairs = [f"<<{words[i]} {words[i + 1]}>>" for i in range(0, len(words), 2)]
    return " ".join(pairs)


def _op_bigram_wrapping(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    words = response.split()
    wrapped = [i for i in range(0, len(words) - 1, 2) if words[i].startswith("<<") and words[i + 1].endswith(">>")]
    if not wrapped:
        return
    for i in (wrapped[0], wrapped[len(wrapped) // 2], wrapped[-1]):
        mutated = words.copy()
        mutated[i] = mutated[i][2:]
        yield " ".join(mutated)
    i = wrapped[0]
    mutated2 = words.copy()
    mutated2[i + 1] = mutated2[i + 1][:-2]
    yield " ".join(mutated2)


register("detectable_format:bigram_wrapping", _op_bigram_wrapping, _fixture_bigram_wrapping)


# ---------------------------------------------------------------------------
# copy:copying_simple and copy:copying_multiple share the same exact-text
# semantics as group C's copy:copy, reimplemented here independently.
# ---------------------------------------------------------------------------


def _fixture_copying_simple(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    return inst._prompt_to_repeat


def _op_copying_simple(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    words = response.split()
    if not words:
        return
    if len(words) > 1:
        yield " ".join(words[:-1])
    yield response.rstrip() + " " + words[-1]
    yield response.rstrip() + " please"


register("copy:copying_simple", _op_copying_simple, _fixture_copying_simple)


def _fixture_copying_multiple(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    return "******".join([inst._prompt_to_repeat] * inst._N)


def _op_copying_multiple(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    parts = response.split("******")
    if not parts:
        return
    last_words = parts[-1].split()
    if len(last_words) > 1:
        yield "******".join([*parts[:-1], " ".join(last_words[:-1])])
    idx = response.find("******")
    if idx != -1:
        yield response[:idx] + "*****" + response[idx + 6 :]
    yield response + "******" + parts[-1]


register("copy:copying_multiple", _op_copying_multiple, _fixture_copying_multiple)


# ---------------------------------------------------------------------------
# punctuation:punctuation_dot and punctuation:punctuation_exclamation
# ---------------------------------------------------------------------------

_NO_DOT_NO_BANG_FILLER = (
    "The morning market opens early and vendors arrange fresh produce along the busy street while "
    "customers wander slowly between the stalls comparing prices and chatting with old friends before "
    "the midday heat sends everyone searching for shade beneath the wide canvas awnings"
)


def _fixture_punctuation_dot(c: Constraint, rng: random.Random) -> str:
    build(c)
    return _NO_DOT_NO_BANG_FILLER


def _op_punctuation_dot(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    words = response.split()
    if not words:
        return
    yield response.rstrip() + "."
    mid = len(words) // 2
    yield " ".join([*words[:mid], words[mid] + ".", *words[mid + 1 :]])
    yield "." + response


register("punctuation:punctuation_dot", _op_punctuation_dot, _fixture_punctuation_dot)


def _fixture_punctuation_exclamation(c: Constraint, rng: random.Random) -> str:
    build(c)
    return _NO_DOT_NO_BANG_FILLER


def _op_punctuation_exclamation(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    words = response.split()
    if not words:
        return
    yield response.rstrip() + "!"
    mid = len(words) // 2
    yield " ".join([*words[:mid], words[mid] + "!", *words[mid + 1 :]])
    yield "!" + response


register("punctuation:punctuation_exclamation", _op_punctuation_exclamation, _fixture_punctuation_exclamation)


# ---------------------------------------------------------------------------
# count:lowercase_counting
# ---------------------------------------------------------------------------

_LOWERCASE_COUNTING_WORDS: tuple[str, ...] = (
    "mountain", "river", "forest", "harbor", "meadow", "canyon", "valley", "island", "desert", "garden",
    "orchard", "village", "bridge", "tower", "market", "library", "museum", "theater", "station", "chapel",
    "quarry", "vineyard", "pasture", "coastline", "wetland", "plateau", "glacier", "reef", "lagoon", "delta",
    "prairie", "savanna", "tundra", "highland", "lowland", "estuary", "peninsula", "archipelago", "basin", "ridge",
)


def _fixture_lowercase_counting(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    n = inst._N
    words = list(_LOWERCASE_COUNTING_WORDS)
    keep_lower = min(n, max(len(words) - 5, 0))
    positions = set(rng.sample(range(len(words)), keep_lower)) if keep_lower else set()
    out = [w if i in positions else w.capitalize() for i, w in enumerate(words)]
    return " ".join(out) + "."


def _op_lowercase_counting(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    words = response.split()
    idxs = [i for i, w in enumerate(words) if w[:1].isupper()]
    if not idxs:
        return
    for idx in (idxs[0], idxs[len(idxs) // 2], idxs[-1]):
        mutated = words.copy()
        mutated[idx] = mutated[idx][0].lower() + mutated[idx][1:]
        yield " ".join(mutated)


register("count:lowercase_counting", _op_lowercase_counting, _fixture_lowercase_counting)


# ---------------------------------------------------------------------------
# letters:letter_counting (total [a-zA-Z] letters) and letters:letter_counting2
# (LetterFrequencyChecker, a single target letter). Both fixtures land exactly
# on the pass/fail boundary so the operator's minimal add/remove-one edit is
# guaranteed to flip the target.
# ---------------------------------------------------------------------------

_LETTER_TOTAL_FILLER = (
    "A gentle rain fell across the quiet valley while farmers gathered the last of the autumn harvest "
    "before the cold winds arrived from the northern hills and covered every field in a thin layer of "
    "frost that lingered until the sun rose slowly over the distant mountains and warmed the frozen "
    "ground enough for the morning birds to return and sing again"
)


def _fixture_letter_counting(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    n, relation = inst._N, inst._relation
    target = n if relation == "at least" else max(n - 1, 0)
    if target <= 0:
        return "the quiet valley rested"
    count = 0
    cut = len(_LETTER_TOTAL_FILLER)
    for i, ch in enumerate(_LETTER_TOTAL_FILLER):
        if ch.isalpha():
            count += 1
            if count == target:
                cut = i + 1
                break
    text = _LETTER_TOTAL_FILLER[:cut]
    count = sum(1 for ch in text if ch.isalpha())
    if count < target:
        text += "x" * (target - count)
    return text


def _op_letter_counting(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(c)
    n, relation = inst._N, inst._relation
    total = len(re.findall(r"[a-zA-Z]", response))
    if relation == "at least":
        need = total - n + 1
        if need < 1 or need > total:
            return
        chars = list(response)
        removed = 0
        for i in range(len(chars) - 1, -1, -1):
            if chars[i].isalpha():
                del chars[i]
                removed += 1
                if removed == need:
                    break
        yield "".join(chars)
        chars2 = list(response)
        idxs = [i for i, ch in enumerate(chars2) if ch.isalpha()][:need]
        for i in reversed(idxs):
            del chars2[i]
        yield "".join(chars2)
    else:
        need = n - total
        if need < 1:
            return
        yield response + ("x" * need)
        yield ("x" * need) + response


register("letters:letter_counting", _op_letter_counting, _fixture_letter_counting)


_LETTER2_FILLER = (
    "The old lighthouse keeper walked along the rocky shoreline every quiet morning before the fishing "
    "boats returned home with their heavy nets full of silver fish caught overnight in the calm northern "
    "waters near the harbor town where generations of sailors had lived and worked since the village was "
    "first founded many years ago beside the cold restless sea"
)


def _fixture_letter_counting2(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    letter, freq, relation = inst._letter, inst._frequency, inst._comparison_relation
    target = freq if relation == "at least" else max(freq - 1, 0)
    body = list(_LETTER2_FILLER)
    count = sum(1 for ch in body if ch.lower() == letter)
    if count > target:
        to_drop = count - target
        dropped = 0
        for i, ch in enumerate(body):
            if dropped < to_drop and ch.lower() == letter:
                body[i] = ""
                dropped += 1
    text = "".join(body)
    count = sum(1 for ch in text if ch.lower() == letter)
    if count < target:
        text += " " + letter * (target - count)
    return text


def _op_letter_counting2(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(c)
    letter, freq, relation = inst._letter, inst._frequency, inst._comparison_relation
    count = sum(1 for ch in response.lower() if ch == letter)
    if relation == "at least":
        need = count - freq + 1
        if need < 1 or need > count:
            return
        chars = list(response)
        removed = 0
        for i in range(len(chars) - 1, -1, -1):
            if chars[i].lower() == letter:
                del chars[i]
                removed += 1
                if removed == need:
                    break
        yield "".join(chars)
        chars2 = list(response)
        idxs = [i for i, ch in enumerate(chars2) if ch.lower() == letter][:need]
        for i in reversed(idxs):
            del chars2[i]
        yield "".join(chars2)
    else:
        need = freq - count
        if need < 1:
            return
        yield response + (letter * need)
        yield (letter * need) + response


register("letters:letter_counting2", _op_letter_counting2, _fixture_letter_counting2)


# ---------------------------------------------------------------------------
# count:counting_composition. Highest collateral risk of the group: every
# edit either shifts a sentence's word count, a paragraph's sentence count, or
# the paragraph count itself, so four independent strategies are offered.
# ---------------------------------------------------------------------------

_COMPOSITION_POOL: tuple[str, ...] = (
    "quiet", "gardens", "bloom", "softly", "near", "old", "stone", "walls", "every", "spring",
    "children", "laugh", "loudly", "beside", "calm", "pond", "today", "farmers", "gather", "fresh",
    "crops", "before", "dusk", "arrives", "slowly", "across", "wide", "fields", "birds", "sing",
)


def _fixture_counting_composition(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    n_sent, n_words = inst._n_sent, inst._n_words
    content_words = max(n_words - 1, 0)
    pool = itertools.cycle(_COMPOSITION_POOL)
    paragraphs = []
    for _ in range(3):
        sentences = []
        for _ in range(n_sent):
            chunk = [next(pool) for _ in range(content_words)]
            if chunk:
                chunk[0] = chunk[0].capitalize()
            sentences.append(" ".join(chunk) + ".")
        paragraphs.append(" ".join(sentences))
    return "\n\n***\n\n".join(paragraphs)


def _op_counting_composition(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    dot = response.find(".")
    if dot == -1:
        return
    head, tail = response[: dot + 1], response[dot + 1 :]
    # candidate 1: insert an extra word into the first sentence (breaks its word count)
    yield head[:-1] + " extra." + tail
    # candidate 2: drop the last content word of the first sentence (breaks its word count)
    words_in_first = head[:-1].split()
    if len(words_in_first) > 1:
        yield " ".join(words_in_first[:-1]) + "." + tail
    # candidate 3: merge the first "***" divider (breaks the paragraph/sentence count)
    star = response.find("***")
    if star != -1:
        start, end = star, star + 3
        if start > 0 and response[start - 1] in " \n\t":
            start -= 1
        if end < len(response) and response[end] in " \n\t":
            end += 1
        yield response[:start] + " " + response[end:]
    # candidate 4: duplicate the first sentence in place (breaks the sentence count)
    yield head + " " + head + tail


register("count:counting_composition", _op_counting_composition, _fixture_counting_composition)


# ---------------------------------------------------------------------------
# count:count_unique
# ---------------------------------------------------------------------------

_UNIQUE_WORDS: tuple[str, ...] = (
    "wandering", "travelers", "often", "discover", "hidden", "paths", "winding", "through", "quiet", "forests",
    "beyond", "ancient", "stone", "bridges", "where", "gentle", "streams", "reflect", "golden", "autumn",
    "light", "while", "curious", "birds", "circle", "overhead", "searching", "for", "warm", "shelter",
    "before", "evening", "arrives", "across", "distant", "hills", "local", "guides", "share", "stories",
    "about", "forgotten", "villages", "nestled", "among", "rolling", "valleys", "surrounded", "by", "tall",
    "pine", "trees", "and", "scattered", "wildflowers", "growing", "freely", "near", "narrow", "trails",
    "that", "lead", "toward", "breathtaking", "views", "atop", "rugged", "cliffs", "overlooking", "calm", "lakes",
)


def _fixture_count_unique(c: Constraint, rng: random.Random) -> str:
    build(c)
    return " ".join(_UNIQUE_WORDS) + "."


def _op_count_unique(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    words = response.split()
    if len(words) < 2:
        return
    yield " ".join([*words, words[0]])
    mid = len(words) // 2
    yield " ".join([*words[: mid + 1], words[mid], *words[mid + 1 :]])
    yield " ".join([words[-1], *words])


register("count:count_unique", _op_count_unique, _fixture_count_unique)


# ---------------------------------------------------------------------------
# count:count_increment_word
# ---------------------------------------------------------------------------

_INCREMENT_FILLER: tuple[str, ...] = (
    "walked", "slowly", "through", "the", "quiet", "market", "square", "admiring", "colorful", "stalls",
    "filled", "with", "fresh", "produce", "while", "local", "musicians", "played", "gentle", "tunes",
    "nearby", "and", "children", "chased", "each", "other", "around", "the", "old", "stone",
    "fountain", "before", "their", "parents", "called", "them", "back", "for", "dinner",
)


def _fixture_count_increment_word(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    k1, k2 = inst._keyword1, inst._keyword2
    filler = [w for w in _INCREMENT_FILLER if k1.lower() not in w.lower() and k2.lower() not in w.lower()]
    third = max(len(filler) // 3, 1)
    words = [k1, *filler[:third], k2, *filler[third : 2 * third], k2, *filler[2 * third :]]
    return " ".join(words) + "."


def _op_count_increment_word(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(c)
    k1, k2 = inst._keyword1, inst._keyword2
    m1 = list(re.finditer(re.escape(k1), response, flags=re.IGNORECASE))
    m2 = list(re.finditer(re.escape(k2), response, flags=re.IGNORECASE))
    if m1:
        first = m1[0]
        yield response[: first.end()] + " " + first.group() + response[first.end() :]
    if m2:
        last = m2[-1]
        yield (response[: last.start()] + response[last.end() :]).replace("  ", " ")
    if m2:
        last = m2[-1]
        yield response[: last.end()] + " " + last.group() + response[last.end() :]
    if m1:
        first = m1[0]
        yield (response[: first.start()] + response[first.end() :]).replace("  ", " ")


register("count:count_increment_word", _op_count_increment_word, _fixture_count_increment_word)


# ---------------------------------------------------------------------------
# keywords:palindrome
# ---------------------------------------------------------------------------

_PALINDROME_FILLER = (
    "The old fishing boats rested quietly near the wooden dock while the tide stayed level with the "
    "worn stone steps leading up toward the village. Local workers repaired torn nets beside stacked "
    "crates as gulls circled overhead searching for scraps left behind by the returning crews. Visitors "
    "walking along the harbor often paused to watch the calm water before continuing toward the market"
)


def _fixture_palindrome(c: Constraint, rng: random.Random) -> str:
    build(c)
    return _PALINDROME_FILLER


def _op_palindrome(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    words = response.split()
    idx = next((i for i, w in enumerate(words) if len(w) > 1 and w == w[::-1]), None)
    if idx is None:
        return
    mutated = words.copy()
    mutated[idx] = "steady"
    yield " ".join(mutated)
    mutated2 = words.copy()
    del mutated2[idx]
    yield " ".join(mutated2)
    mutated3 = words.copy()
    mutated3[idx] = mutated3[idx] + ","
    yield " ".join(mutated3)


register("keywords:palindrome", _op_palindrome, _fixture_palindrome)


# ---------------------------------------------------------------------------
# keywords:keyword_specific_position
# ---------------------------------------------------------------------------

_KSP_FILLER_POOL: tuple[str, ...] = (
    "gentle", "wind", "moved", "slowly", "across", "open", "fields", "each", "quiet", "evening",
    "while", "birds", "sang", "softly", "near", "old", "stone", "bridge", "before", "sunset", "faded",
)


def _fixture_keyword_specific_position(c: Constraint, rng: random.Random) -> str:
    inst = build(c)
    keyword, n, m = inst._keyword, inst._n, inst._m
    pool = itertools.cycle(_KSP_FILLER_POOL)
    lead_sentences = ["A short quiet moment passed calmly." for _ in range(n - 1)]
    target_words = [next(pool) for _ in range(m - 1)]
    target_words.append(keyword)
    target_words.extend(next(pool) for _ in range(3))
    if m > 1:
        target_words[0] = target_words[0].capitalize()
    target_sentence = " ".join(target_words) + "."
    return " ".join([*lead_sentences, target_sentence])


def _keyword_specific_position_span(response: str, n: int, m: int) -> tuple[int, int] | None:
    """Locate word m-1 of sentence n-1's character span in `response`, mirroring
    KeywordSpecificPositionChecker's own split_into_sentences + word_tokenize walk
    exactly, so the span found is the one the checker actually reads."""
    sentences = instructions_util.split_into_sentences(response)
    if len(sentences) < n:
        return None
    cursor = 0
    sent_start = -1
    sentence_text = ""
    for i in range(n):
        idx = response.find(sentences[i], cursor)
        if idx == -1:
            return None
        sent_start, sentence_text = idx, sentences[i]
        cursor = idx + len(sentence_text)
    words = instructions_util.nltk.word_tokenize(sentence_text)
    if len(words) < m:
        return None
    word_cursor = 0
    for j, word in enumerate(words):
        local_idx = sentence_text.find(word, word_cursor)
        if local_idx == -1:
            return None
        if j == m - 1:
            return sent_start + local_idx, sent_start + local_idx + len(word)
        word_cursor = local_idx + len(word)
    return None


def _op_keyword_specific_position(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    inst = build(c)
    keyword, n, m = inst._keyword, inst._n, inst._m
    span = _keyword_specific_position_span(response, n, m)
    if span is None:
        idx = response.find(keyword)
        if idx == -1:
            return
        span = (idx, idx + len(keyword))
    start, end = span
    yield response[:start] + "neutral" + response[end:]
    del_end = end
    if start > 0 and end < len(response) and response[start - 1] == " " and response[end] == " ":
        del_end += 1  # eat the trailing separator only, leaving one space at start-1
    yield response[:start] + response[del_end:]
    yield response[:start] + "extra " + response[start:]


register("keywords:keyword_specific_position", _op_keyword_specific_position, _fixture_keyword_specific_position)


# ---------------------------------------------------------------------------
# keywords:start_end
# ---------------------------------------------------------------------------


def _fixture_start_end(c: Constraint, rng: random.Random) -> str:
    build(c)
    return (
        "Morning arrived quietly across the harbor as fishing boats drifted slowly past the old "
        "stone pier while gulls circled overhead searching for scraps left behind by the early "
        "crews returning home this calm autumn morning"
    )


def _op_start_end(response: str, c: Constraint, rng: random.Random) -> Iterator[str]:
    words = response.split()
    if len(words) < 2:
        return
    yield " ".join([*words[:-1], "evening"])
    yield " ".join(["Evening", *words[1:]])
    yield response.rstrip() + "."


register("keywords:start_end", _op_start_end, _fixture_start_end)

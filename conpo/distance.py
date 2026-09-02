"""Token-level normalised edit distance between responses (ConPO.pdf section 2.1).

d_Y(y, y') is specified only as "token-level edit distance normalised by
response length"; the paper does not fix which tokeniser. This module splits
on whitespace rather than the policy model's subword tokeniser, for two
reasons: the distance gate is CPU code shared by forward.py (repair
admission) and reverse.py (corruption admission), and it must not pull in a
loaded model on paths that run without one; and calibrating delta (DEV_SET,
conpo/config.py) against whitespace tokens keeps the calibration stable
whether or not a tokeniser is even available. The alternative -- distance
over the policy's own subword tokens, which tracks what the model actually
pays for length -- is a straightforward swap here (Levenshtein.distance
takes any sequence of hashables) if delta ever needs recalibrating against
it.
"""

from rapidfuzz.distance import Levenshtein

from conpo import config


def tokens(text: str) -> list[str]:
    """Whitespace-separated tokens."""
    return text.split()


def token_distance(a: str, b: str) -> float:
    """Levenshtein distance between `tokens(a)` and `tokens(b)`, normalised
    by `max(len(tokens(a)), 1)`.

    `a` is the ORIGINAL response: section 2.1 normalises d_Y by response
    length, and `a` is that reference length. `b` is the candidate (a
    repair proposal or a corruption).
    """
    a_tokens = tokens(a)
    b_tokens = tokens(b)
    distance = Levenshtein.distance(a_tokens, b_tokens)
    return distance / max(len(a_tokens), 1)


def within_delta(a: str, b: str, delta: float = config.DELTA) -> bool:
    """True if `token_distance(a, b) <= delta`."""
    return token_distance(a, b) <= delta

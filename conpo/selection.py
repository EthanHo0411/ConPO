"""Transfer-aware active selection over constraint families (ConPO.pdf section 2.6).

FamilyStats tracks, per constraint family c: the first-pass satisfaction rate
p_c (EMA), the admitted-edge count n_c, and the transfer statistic T_c (EMA of
the cosine between the response-loss gradient on a probe batch of family c and
the clean-objective gradient). From these it derives reward-uncertainty U_c,
coverage C_c, and the acquisition score a_c. mu(c | x) turns acquisition into a
sampling distribution over the families present on a prompt, with an
exploration floor and an exclusion for families that have no corruption
operator at all (conpo.operators.registry.NO_OPERATOR).
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

import torch

from conpo import config
from conpo.operators.registry import NO_OPERATOR


class FamilyStats:
    """Per-family p_c, n_c, T_c and the acquisition score built from them."""

    def __init__(self) -> None:
        self._p: dict[str, float] = {}
        self._n: dict[str, int] = {}
        self._t: dict[str, float] = {}
        self._last_probe_update: dict[str, int] = {}

    def update_pass_rates(self, observations: Mapping[str, Sequence[bool]]) -> None:
        """Fold one first-pass group's verdicts into p_c for every family present.

        Every family in `observations` is used, including a group whose rate is
        0.0 or 1.0: p_c is initialised to that first observed rate, and every
        later observation blends in with decay config.EMA_DECAY_P_C.
        """
        decay = config.EMA_DECAY_P_C
        for family, verdicts in observations.items():
            rate = sum(verdicts) / len(verdicts)
            if family in self._p:
                self._p[family] = decay * self._p[family] + (1 - decay) * rate
            else:
                self._p[family] = rate

    def record_admission(self, family: str) -> None:
        """Count one more admitted edge for `family`."""
        self._n[family] = self._n.get(family, 0) + 1

    def record_probe(self, family: str, cosine: float, update: int) -> None:
        """Fold one probe's gradient cosine into T_c and stamp the probe update.

        T_c starts at 0.0 for a family that has never been probed; the first
        probe blends that 0.0 with `cosine` using the same EMA as every later
        probe (decay config.EMA_DECAY_T_C).
        """
        decay = config.EMA_DECAY_T_C
        prior = self._t.get(family, 0.0)
        self._t[family] = decay * prior + (1 - decay) * cosine
        self._last_probe_update[family] = update

    def p_c(self, family: str) -> float:
        return self._p.get(family, 0.0)

    def n_c(self, family: str) -> int:
        return self._n.get(family, 0)

    def t_c(self, family: str) -> float:
        return self._t.get(family, 0.0)

    def last_probe_update(self, family: str) -> int | None:
        """Optimiser update at which `family` was last probed, or None if never."""
        return self._last_probe_update.get(family)

    def U(self, family: str) -> float:
        """Reward uncertainty: 4 p (1 - p), maximal at p = 0.5, zero at p in {0, 1}."""
        p = self.p_c(family)
        return 4.0 * p * (1.0 - p)

    def C(self, family: str) -> float:
        """Coverage: 1 / sqrt(n_c + 1), monotone decreasing in n_c."""
        return 1.0 / math.sqrt(self.n_c(family) + 1)

    def acquisition(self, family: str) -> float:
        """a_c = lambda_T T_c + lambda_U U_c + lambda_C C_c."""
        return (
            config.LAMBDA_T * self.t_c(family)
            + config.LAMBDA_U * self.U(family)
            + config.LAMBDA_C * self.C(family)
        )

    def state_dict(self) -> dict[str, dict[str, float | int | None]]:
        """Every field needed to resume FamilyStats exactly (RUN_RESUME,
        conpo/trainer.py's checkpoint persistence). Unlike snapshot(), which
        derives U_c/C_c/acquisition for logging, this round-trips unchanged
        through load_state_dict."""
        return {"p": dict(self._p), "n": dict(self._n), "t": dict(self._t), "last_probe_update": dict(self._last_probe_update)}

    def load_state_dict(self, state: Mapping[str, Mapping[str, float | int | None]]) -> None:
        """Inverse of state_dict; replaces every field in place."""
        self._p = dict(state["p"])
        self._n = dict(state["n"])
        self._t = dict(state["t"])
        self._last_probe_update = dict(state["last_probe_update"])

    def snapshot(self) -> dict[str, dict[str, float | int | None]]:
        """One dict per family seen so far, for logging."""
        families = set(self._p) | set(self._n) | set(self._t) | set(self._last_probe_update)
        return {
            family: {
                "p_c": self.p_c(family),
                "n_c": self.n_c(family),
                "T_c": self.t_c(family),
                "U_c": self.U(family),
                "C_c": self.C(family),
                "acquisition": self.acquisition(family),
                "last_probe_update": self.last_probe_update(family),
            }
            for family in sorted(families)
        }


def mu(stats: FamilyStats, families_on_prompt: Sequence[str]) -> dict[str, float]:
    """mu(c | x) over the families present on one prompt (ConPO.pdf section 2.6).

    (1 - EPS_A) * softmax(a / ETA) over the eligible families, plus a uniform
    floor of EPS_A / |eligible|. Families in NO_OPERATOR (no corruption
    operator exists for them) are excluded before C(x) is formed.
    """
    eligible = [family for family in families_on_prompt if family not in NO_OPERATOR]
    if not eligible:
        raise ValueError("no family with an operator is present on this prompt")
    scores = [stats.acquisition(family) / config.ETA for family in eligible]
    peak = max(scores)
    weights = [math.exp(score - peak) for score in scores]
    total = sum(weights)
    softmax = [weight / total for weight in weights]
    floor = config.EPS_A / len(eligible)
    return {family: (1 - config.EPS_A) * share + floor for family, share in zip(eligible, softmax)}


def sample_family(stats: FamilyStats, families_on_prompt: Sequence[str], rng: random.Random) -> str:
    """Draw one family from mu(. | x)."""
    probabilities = mu(stats, families_on_prompt)
    families = list(probabilities.keys())
    weights = list(probabilities.values())
    return rng.choices(families, weights=weights, k=1)[0]


def grad_cosine(grad_a: Sequence[torch.Tensor], grad_b: Sequence[torch.Tensor]) -> float:
    """Cosine similarity between two gradients, each given as a list of tensors."""
    flat_a = torch.cat([g.reshape(-1) for g in grad_a])
    flat_b = torch.cat([g.reshape(-1) for g in grad_b])
    return torch.nn.functional.cosine_similarity(flat_a.unsqueeze(0), flat_b.unsqueeze(0)).item()


def probe_due(stats: FamilyStats, family: str, current_update: int, period: int = config.PROBE_PERIOD_S) -> bool:
    """Whether `family` is due a probe refresh: never probed, or `period` updates have elapsed."""
    last = stats.last_probe_update(family)
    if last is None:
        return True
    return current_update - last >= period


def next_probe_family(stats: FamilyStats, families: Sequence[str], current_update: int) -> str | None:
    """The family in `families` least recently probed (never-probed counts as oldest).

    Round-robin: repeated calls interleaved with record_probe() cycle through
    `families` in order. `current_update` is accepted for parity with
    probe_due() but the ordering itself comes entirely from last_probe_update.
    """
    if not families:
        return None
    return min(
        families,
        key=lambda family: stats.last_probe_update(family) if stats.last_probe_update(family) is not None else -1,
    )

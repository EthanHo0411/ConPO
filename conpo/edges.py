"""Counterfactual edges and their buffers (ConPO.pdf sections 2.2-2.4, 4.1).

An Edge is one admitted counterfactual pair e = (x, y-, y+, A) (section 2.4).
"forward" edges come from a monotone repair trajectory (section 2.2:
y_t -> y_{t+1}); "reverse" edges come from corrupting a feasible response
(section 2.3: y+ -> corrupted y-). Both directions feed the same buffer B
(section 2.4) and are trained with the same response-level loss.

EdgeBuffer implements B (section 4.1: capacity 50,000, hard age cap 500
optimiser updates): a FIFO buffer over capacity, with edges older than
age_cap retired before every draw. FeasibleBuffer implements B_F (sections
2.2-2.3): feasible responses awaiting reverse-direction corruption.
"""

import json
import random
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from conpo import config

_DIRECTIONS = frozenset({"forward", "reverse"})


@dataclass(frozen=True)
class Edge:
    """One admitted counterfactual edge.

    direction: "forward" (repair trajectory step, section 2.2) or "reverse"
    (corruption, section 2.3). step is the repair-trajectory index for
    forward edges (0 for reverse edges, which are single-shot).
    logp_ref_plus/minus and len_plus/minus cache log pi_ref(y | x) and |y|
    so q_theta^(alpha) (section 2.4) never has to re-run the reference
    model. created_update is the optimiser update at which the edge was
    admitted; age(e) (section 2.4) is current_update - created_update.
    meta holds anything else a producer wants to keep (e.g. the operator
    or repair-step reason) without widening this schema.
    """

    prompt_id: str
    prompt: str
    y_plus: str
    y_minus: str
    families: frozenset[str]
    direction: str
    step: int
    logp_ref_plus: float
    logp_ref_minus: float
    len_plus: int
    len_minus: int
    created_update: int
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(_DIRECTIONS)}, got {self.direction!r}")
        object.__setattr__(self, "families", frozenset(self.families))
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))

    def to_json(self) -> dict[str, object]:
        """JSON-safe representation (families as a sorted list)."""
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "y_plus": self.y_plus,
            "y_minus": self.y_minus,
            "families": sorted(self.families),
            "direction": self.direction,
            "step": self.step,
            "logp_ref_plus": self.logp_ref_plus,
            "logp_ref_minus": self.logp_ref_minus,
            "len_plus": self.len_plus,
            "len_minus": self.len_minus,
            "created_update": self.created_update,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "Edge":
        return cls(
            prompt_id=data["prompt_id"],
            prompt=data["prompt"],
            y_plus=data["y_plus"],
            y_minus=data["y_minus"],
            families=frozenset(data["families"]),
            direction=data["direction"],
            step=data["step"],
            logp_ref_plus=data["logp_ref_plus"],
            logp_ref_minus=data["logp_ref_minus"],
            len_plus=data["len_plus"],
            len_minus=data["len_minus"],
            created_update=data["created_update"],
            meta=data.get("meta", {}),
        )


class EdgeBuffer:
    """FIFO counterfactual-edge buffer B (section 4.1: capacity 50,000,
    hard age cap of 500 optimiser updates).

    Age-cap retirement is lazy: push() and sample() both drop edges whose
    age exceeds age_cap, measured against the highest current_update either
    has observed so far. This assumes edges are pushed in non-decreasing
    created_update order, true of a training loop that constructs edges as
    optimiser updates advance, so only the buffer's front (oldest edges)
    is ever checked.
    """

    def __init__(self, capacity: int = config.BUFFER_SIZE, age_cap: int = config.BUFFER_AGE_CAP) -> None:
        self.capacity = capacity
        self.age_cap = age_cap
        self._edges: deque[Edge] = deque()
        self._current_update = 0

    def __len__(self) -> int:
        return len(self._edges)

    def age(self, edge: Edge, current_update: int) -> int:
        return current_update - edge.created_update

    def _retire(self, current_update: int) -> None:
        self._current_update = max(self._current_update, current_update)
        while self._edges and self.age(self._edges[0], self._current_update) > self.age_cap:
            self._edges.popleft()

    def push(self, edge: Edge) -> None:
        self._retire(edge.created_update)
        self._edges.append(edge)
        while len(self._edges) > self.capacity:
            self._edges.popleft()

    def sample(self, n: int, current_update: int, rng: random.Random, balanced: bool = True) -> list[Edge]:
        """Draw at most `n` edges, retiring stale ones first.

        When `balanced` and both directions are present, allocates n // 2 to
        each side, giving any remainder (n is odd) to whichever side has
        more edges available; a side short of its allocation yields what it
        has and the rest goes to the other side.
        """
        self._retire(current_update)
        forward = [e for e in self._edges if e.direction == "forward"]
        reverse = [e for e in self._edges if e.direction == "reverse"]
        if not balanced or not forward or not reverse:
            pool = list(self._edges) if not balanced else (forward or reverse)
            return rng.sample(pool, min(n, len(pool)))

        base, remainder = divmod(n, 2)
        forward_target = base + (remainder if len(forward) >= len(reverse) else 0)
        reverse_target = base + (remainder if len(reverse) > len(forward) else 0)
        take_forward = min(forward_target, len(forward))
        take_reverse = min(reverse_target, len(reverse))
        shortfall = n - take_forward - take_reverse
        if shortfall > 0:
            take_forward = min(len(forward), take_forward + shortfall)
            shortfall = n - take_forward - take_reverse
        if shortfall > 0:
            take_reverse = min(len(reverse), take_reverse + shortfall)

        sampled = rng.sample(forward, take_forward) + rng.sample(reverse, take_reverse)
        rng.shuffle(sampled)
        return sampled

    def sample_where(
        self, predicate: Callable[[Edge], bool], n: int, current_update: int, rng: random.Random
    ) -> list[Edge]:
        """Draw at most `n` edges satisfying `predicate`, retiring stale ones first.

        One pass over the buffer via reservoir sampling, instead of
        materialising the whole buffer (`sample(len(buffer))`) and then
        filtering it down -- callers that only want single-family edges or
        one family's edges (compute_loss's L_prompt draw, the T_c probe's
        per-family draw) no longer pay for a full-buffer copy each call.
        Uniform-random without replacement over the matching edges, same as
        sample()'s own guarantee.
        """
        self._retire(current_update)
        reservoir: list[Edge] = []
        seen = 0
        for edge in self._edges:
            if not predicate(edge):
                continue
            seen += 1
            if len(reservoir) < n:
                reservoir.append(edge)
            else:
                j = rng.randint(0, seen - 1)
                if j < n:
                    reservoir[j] = edge
        return reservoir

    def counts_by_direction(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._edges:
            counts[e.direction] = counts.get(e.direction, 0) + 1
        return counts

    def counts_by_family(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self._edges:
            for f in e.families:
                counts[f] = counts.get(f, 0) + 1
        return counts

    def save(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for e in self._edges:
                f.write(json.dumps(e.to_json()) + "\n")

    def load(self, path: str | Path) -> None:
        """Replace this buffer's contents with the edges saved at `path`."""
        self._edges.clear()
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._edges.append(Edge.from_json(json.loads(line)))
        self._current_update = max((e.created_update for e in self._edges), default=0)
        while len(self._edges) > self.capacity:
            self._edges.popleft()


class FeasibleBuffer:
    """Feasible-response buffer B_F (sections 2.2-2.3): responses drawn for
    reverse-direction corruption. Caps how many responses accumulate per
    prompt, FIFO within a prompt, so one prompt cannot dominate sampling.
    """

    def __init__(self, per_prompt_cap: int = 8) -> None:
        self.per_prompt_cap = per_prompt_cap
        self._by_prompt: dict[str, list[str]] = {}

    def add(self, prompt_id: str, response: str) -> None:
        bucket = self._by_prompt.setdefault(prompt_id, [])
        bucket.append(response)
        if len(bucket) > self.per_prompt_cap:
            del bucket[0]

    def state_dict(self) -> dict[str, list[str]]:
        """JSON-safe snapshot of every prompt's response bucket (RUN_RESUME,
        conpo/trainer.py's checkpoint persistence)."""
        return {k: list(v) for k, v in self._by_prompt.items()}

    def load_state_dict(self, state: Mapping[str, list[str]]) -> None:
        """Inverse of state_dict; replaces this buffer's contents in place."""
        self._by_prompt = {k: list(v) for k, v in state.items()}

    def sample_prompt(self, rng: random.Random) -> tuple[str, str]:
        prompt_id = rng.choice(list(self._by_prompt.keys()))
        response = rng.choice(self._by_prompt[prompt_id])
        return prompt_id, response

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_prompt.values())

    def drop(self, prompt_id: str) -> None:
        self._by_prompt.pop(prompt_id, None)

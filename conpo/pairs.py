"""DPO baseline pair construction (locked decision 2026-09-02(c)): the three
fixed pair sources trained by scripts/train_dpo.py.

`Pair` is the on-disk unit every arm's pairs file holds. build_vf_pairs
(vf_dpo, the AutoIF / Tulu 3 shape): independently sample n responses per
prompt from the initial policy, keep prompts with at least one verifier-
passing and one verifier-failing response, pair one random passing with one
random failing. build_relax_pairs (relax_dpo, the Tulu 3 If-Persona-Pref
shape): for each prompt and each constraint index, sample chosen candidates
from the full prompt and rejected candidates from that constraint's
counterfactual_prompts x_minus, keeping only pairs where the chosen response
satisfies every constraint and the rejected response fails specifically the
deleted one. build_edge_pairs (edge_dpo): replays a ConPO run's dumped
conpo.edges.Edge buffer as pairs, isolating ConPO's contrastive objective
from its counterfactual-construction machinery.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from conpo import config
from conpo.data import PoolRow, TokenMeter
from conpo.edges import Edge
from conpo.engine import Engine
from conpo.prompts import counterfactual_prompts
from conpo.verifiers import check, feasible


@dataclass(frozen=True)
class Pair:
    """One DPO training pair. source: "vf" | "relax" | "edge"."""

    prompt_id: str
    prompt: str
    chosen: str
    rejected: str
    source: str
    meta: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "source": self.source,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "Pair":
        return cls(
            prompt_id=data["prompt_id"],
            prompt=data["prompt"],
            chosen=data["chosen"],
            rejected=data["rejected"],
            source=data["source"],
            meta=dict(data.get("meta", {})),
        )


def save_pairs(pairs: Sequence[Pair], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p.to_json()) + "\n")


def load_pairs(path: str | Path) -> list[Pair]:
    pairs: list[Pair] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(Pair.from_json(json.loads(line)))
    return pairs


# ---------------------------------------------------------------------------
# vf_dpo
# ---------------------------------------------------------------------------


def build_vf_pairs(
    rows: Sequence[PoolRow],
    engine: Engine,
    n_samples: int,
    rng: random.Random,
    meter: TokenMeter,
    pairs_per_prompt: int = 1,
) -> list[Pair]:
    """Verifier-filtered pairs from independently sampled responses.

    `n_samples` responses are drawn per prompt at config.TEMPERATURE /
    config.TOP_P / config.MAX_COMPLETION_TOKENS, batched into one
    `engine.generate` call. Every sampled token is charged to `meter`,
    admitted or not. A prompt with no passing response or no failing
    response yields no pair.
    """
    if not rows:
        return []
    messages_batch = [[{"role": "user", "content": r.prompt}] for r in rows]
    seed = rng.randint(0, 2**31 - 1)
    responses_batch = engine.generate(
        messages_batch, n_samples, config.MAX_COMPLETION_TOKENS, config.TEMPERATURE, config.TOP_P, seed
    )
    meter.add(engine.count_tokens([r for group in responses_batch for r in group]))

    pairs: list[Pair] = []
    for row, responses in zip(rows, responses_batch):
        passing = [r for r in responses if feasible(row.constraints, r)]
        failing = [r for r in responses if not feasible(row.constraints, r)]
        if not passing or not failing:
            continue
        for _ in range(pairs_per_prompt):
            pairs.append(
                Pair(
                    prompt_id=row.key,
                    prompt=row.prompt,
                    chosen=rng.choice(passing),
                    rejected=rng.choice(failing),
                    source="vf",
                    meta={},
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# relax_dpo
# ---------------------------------------------------------------------------


def build_relax_pairs(
    rows: Sequence[PoolRow],
    engine: Engine,
    n_samples: int,
    rng: random.Random,
    meter: TokenMeter,
) -> list[Pair]:
    """Relaxation-pair DPO pairs (Tulu 3 If-Persona-Pref shape).

    For each prompt and each constraint index, x_minus (conpo.prompts.
    counterfactual_prompts) drops that one constraint sentence. Chosen
    candidates are sampled from the full prompt x, rejected candidates from
    x_minus. A pair is admitted only when a chosen candidate passes every
    constraint in `row.constraints` (feasible, the full conjunctive check)
    and a rejected candidate fails the deleted constraint specifically --
    both checked against `row.constraints`, never a partial tuple. At most
    one pair per (row, constraint index); the deleted family is recorded in
    `meta`.
    """
    pairs: list[Pair] = []
    for row in rows:
        for target_index, target in enumerate(row.constraints):
            x_plus, x_minus = counterfactual_prompts(row, target_index)
            seed = rng.randint(0, 2**31 - 1)
            chosen_candidates, rejected_candidates = engine.generate(
                [[{"role": "user", "content": x_plus}], [{"role": "user", "content": x_minus}]],
                n_samples,
                config.MAX_COMPLETION_TOKENS,
                config.TEMPERATURE,
                config.TOP_P,
                seed,
            )
            meter.add(engine.count_tokens(chosen_candidates + rejected_candidates))

            chosen_ok = [c for c in chosen_candidates if feasible(row.constraints, c)]
            rejected_ok = [c for c in rejected_candidates if not check(target, c)]
            if not chosen_ok or not rejected_ok:
                continue
            pairs.append(
                Pair(
                    prompt_id=row.key,
                    prompt=x_plus,
                    chosen=rng.choice(chosen_ok),
                    rejected=rng.choice(rejected_ok),
                    source="relax",
                    meta={"deleted_family": target.iid, "deleted_index": target_index},
                )
            )
    return pairs


# ---------------------------------------------------------------------------
# edge_dpo
# ---------------------------------------------------------------------------


def build_edge_pairs(edges_jsonl_path: str | Path) -> list[Pair]:
    """Replay a ConPO run's dumped edge buffer (conpo.edges.Edge) as pairs."""
    pairs: list[Pair] = []
    with open(edges_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            edge = Edge.from_json(json.loads(line))
            pairs.append(
                Pair(
                    prompt_id=edge.prompt_id,
                    prompt=edge.prompt,
                    chosen=edge.y_plus,
                    rejected=edge.y_minus,
                    source="edge",
                    meta={"direction": edge.direction, "families": sorted(edge.families)},
                )
            )
    return pairs

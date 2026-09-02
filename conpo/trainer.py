"""ConPOTrainer: GRPOTrainer subclass implementing ConPO (ConPO.pdf sections 2.2-2.7).

`_generate_and_score_completions` runs the stock GRPO clean-group generation
via `super()`, then (per `arm`) drives forward repair (conpo/forward.py) and
reverse corruption (conpo/reverse.py) over that group's completions, caches
reference log-probs on every newly admitted edge, pushes them into
`edge_buffer`, and writes one mechanism-log record for the step.
`compute_loss` adds L_resp and L_prompt (conpo/losses.py) on top of TRL's own
GRPO objective, sampled from `edge_buffer`, plus an optional gradient-alignment
probe (conpo/selection.py section 2.6).
"""

from __future__ import annotations

import dataclasses
import json
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from trl import GRPOTrainer

from conpo import config, forward, losses, prompts, reverse, selection
from conpo.arms import ArmSpec, Overrides
from conpo.data import PoolRow, TokenMeter
from conpo.edges import Edge, EdgeBuffer, FeasibleBuffer
from conpo.engine import Engine
from conpo.judge import strip_think
from conpo.selection import FamilyStats
from conpo.verifiers import feasible, verifier_vector


def _completion_text(completion: object) -> str:
    """Response text from either a plain string or a TRL conversational
    completion (`[{"role": "assistant", "content": ...}]`)."""
    if isinstance(completion, list):
        return str(completion[-1]["content"])
    return str(completion)


def score_response(
    model: torch.nn.Module, tokenizer: object, prompt_text: str, response: str
) -> tuple[torch.Tensor, int]:
    """log pi(response | prompt_text) and its completion-token length.

    One forward pass on `model` as it stands: adapters enabled or disabled,
    and gradient tracking or not, are entirely the caller's responsibility
    (wrap the call in `torch.no_grad()` / `model.disable_adapter()` for a
    reference score; call plainly for a policy score that needs a gradient).
    Used identically for y_plus and y_minus.
    """
    rendered_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=False,
        **config.POLICY_CHAT_TEMPLATE_KWARGS,
    )
    device = next(model.parameters()).device
    prompt_ids = tokenizer(rendered_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.cat([torch.zeros_like(prompt_ids), torch.ones_like(response_ids)], dim=1)
    logp = losses.sequence_logprobs(model, input_ids, attention_mask, completion_mask)[0]
    length = int(losses.completion_lengths(completion_mask)[0].item())
    return logp, length


class ConPOTrainer(GRPOTrainer):
    """GRPOTrainer subclass implementing ConPO.

    TRL folds `beta * KL(pi_theta || pi_ref)` into its own clean-objective
    loss (trl's `GRPOTrainer._compute_loss`), so `compute_loss` below never
    adds a separate KL term itself: the effective KL weight actually applied
    is `LAMBDA_G * KL_BETA` (config.py), and the paper's objective
    `L = L_resp + lambda_P L_prompt + lambda_G L_clean + beta L_KL` is
    realised exactly only at `LAMBDA_G = 1`.

    pi_ref is the frozen initial policy: with a zero-initialised LoRA
    adapter (peft's default init), the adapter's output equals the base
    model's, so every `logp_ref_*` cached on an Edge is computed once, at
    admission, under `accelerator.unwrap_model(model).disable_adapter()`
    (see `score_response`), and never recomputed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        args: object,
        train_dataset: object,
        processing_class: object,
        arm: ArmSpec,
        rows_by_key: Mapping[str, PoolRow],
        edge_buffer: EdgeBuffer,
        feasible_buffer: FeasibleBuffer,
        stats: FamilyStats,
        judge: object,
        engine_factory: Callable[["ConPOTrainer"], Engine],
        meter: TokenMeter,
        mechanism_log_path: str | Path,
        seed: int = 0,
        overrides: Overrides | None = None,
        probe_enabled: bool = True,
        **kwargs: object,
    ) -> None:
        self.arm = arm
        self.overrides = overrides if overrides is not None else Overrides()
        self.rows_by_key: dict[str, PoolRow] = dict(rows_by_key)
        self.edge_buffer = edge_buffer
        self.feasible_buffer = feasible_buffer
        self.stats = stats
        self.judge = judge
        self.meter = meter
        self.mechanism_log_path = Path(mechanism_log_path)
        self.mechanism_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.probe_enabled = probe_enabled
        self.last_loss_stats: dict[str, float] = {}

        reward_func = self.reward_conjunctive if arm.reward == "conjunctive" else self.reward_additive
        super().__init__(
            model=model,
            reward_funcs=reward_func,
            args=args,
            train_dataset=train_dataset,
            processing_class=processing_class,
            **kwargs,
        )
        self.engine: Engine = engine_factory(self)

    # -----------------------------------------------------------------
    # Reward functions (the arm decides which one is passed to GRPOTrainer)
    # -----------------------------------------------------------------

    def reward_conjunctive(self, completions: Sequence[object], key: Sequence[str], **_: object) -> list[float]:
        """F(x, y) (ConPO.pdf's conjunctive reward): 1.0 iff every constraint on the row passes, else 0.0."""
        scores = []
        for completion, k in zip(completions, key):
            row = self.rows_by_key[k]
            text = _completion_text(completion)
            scores.append(1.0 if feasible(row.constraints, text) else 0.0)
        return scores

    def reward_additive(self, completions: Sequence[object], key: Sequence[str], **_: object) -> list[float]:
        """open-instruct IFEvalVerifier semantics: mean per-constraint 0/1 verdict; empty completion scores 0."""
        scores = []
        for completion, k in zip(completions, key):
            row = self.rows_by_key[k]
            text = _completion_text(completion)
            if not text.strip() or not row.constraints:
                scores.append(0.0)
                continue
            verdicts = verifier_vector(row.constraints, text)
            scores.append(sum(verdicts) / len(verdicts))
        return scores

    # -----------------------------------------------------------------
    # Counterfactual construction
    # -----------------------------------------------------------------

    def _generate_and_score_completions(self, inputs: list[dict[str, object]]) -> dict[str, object]:
        output = super()._generate_and_score_completions(inputs)

        keys = [str(row["key"]) for row in inputs]
        completions_text = [
            strip_think(t)
            for t in self.processing_class.batch_decode(output["completion_ids"], skip_special_tokens=True)
        ]
        self.meter.add(self.engine.count_tokens(completions_text))

        current_update = self.state.global_step
        n = self.num_generations if self.model.training else self.num_generations_eval
        advantages = output["advantages"]

        forward_reasons: Counter[str] = Counter()
        reverse_reasons: Counter[str] = Counter()
        new_edges: list[Edge] = []
        degenerate_groups = 0
        total_groups = 0

        for start in range(0, len(inputs), n):
            end = start + n
            key = keys[start]
            row = self.rows_by_key[key]
            group_completions = completions_text[start:end]
            group_advantages = advantages[start:end]
            total_groups += 1
            if torch.allclose(group_advantages, torch.zeros_like(group_advantages)):
                degenerate_groups += 1

            observations: dict[str, list[bool]] = {c.iid: [] for c in row.constraints}
            failed_responses: list[str] = []
            for text in group_completions:
                verdicts = verifier_vector(row.constraints, text)
                for c, ok in zip(row.constraints, verdicts):
                    observations[c.iid].append(ok)
                if verdicts and all(verdicts):
                    self.feasible_buffer.add(key, text)
                elif verdicts:
                    failed_responses.append(text)
            if observations:
                self.stats.update_pass_rates(observations)

            if self.arm.forward and failed_responses:
                result = forward.repair_failed_group(
                    row,
                    failed_responses,
                    self.engine,
                    self.judge,
                    self.rng,
                    self.meter,
                    current_update,
                    veto=not self.overrides.no_veto,
                )
                new_edges.extend(result.edges)
                forward_reasons.update(result.reasons)
                for endpoint in result.feasible_endpoints:
                    self.feasible_buffer.add(key, endpoint)

        if self.arm.reverse and len(self.feasible_buffer) > 0:
            results = reverse.reverse_batch(
                self.rows_by_key,
                self.feasible_buffer,
                config.REVERSE_DRAWS_PER_STEP,
                self.stats,
                self.judge,
                self.rng,
                current_update,
                selection=self.overrides.selection,
                exactness=not self.overrides.no_exactness,
                distance_gate=not self.overrides.no_distance,
                veto=not self.overrides.no_veto,
            )
            for r in results:
                reverse_reasons[r.reason] += 1
                if r.edge is not None:
                    new_edges.append(r.edge)

        distances: list[float] = []
        admitted_by_direction: Counter[str] = Counter()
        unwrapped = self.accelerator.unwrap_model(self.model)
        with torch.no_grad(), unwrapped.disable_adapter():
            for edge in new_edges:
                logp_plus, len_plus = score_response(unwrapped, self.processing_class, edge.prompt, edge.y_plus)
                logp_minus, len_minus = score_response(unwrapped, self.processing_class, edge.prompt, edge.y_minus)
                scored_edge = dataclasses.replace(
                    edge,
                    logp_ref_plus=logp_plus.item(),
                    logp_ref_minus=logp_minus.item(),
                    len_plus=len_plus,
                    len_minus=len_minus,
                )
                self.edge_buffer.push(scored_edge)
                admitted_by_direction[scored_edge.direction] += 1
                distances.append(float(scored_edge.meta.get("distance", 0.0)))

        record = {
            "step": current_update,
            "admitted_by_direction": dict(admitted_by_direction),
            "veto_discarded_by_direction": {
                "forward": forward_reasons.get("veto", 0),
                "reverse": reverse_reasons.get("veto", 0),
            },
            "reasons_by_direction": {"forward": dict(forward_reasons), "reverse": dict(reverse_reasons)},
            "mean_edit_distance": (sum(distances) / len(distances)) if distances else 0.0,
            "degenerate_group_rate": (degenerate_groups / total_groups) if total_groups else 0.0,
            "edges_per_million_tokens": (len(new_edges) / self.meter.total * 1e6) if self.meter.total else 0.0,
            "edge_buffer_size": len(self.edge_buffer),
            "feasible_buffer_size": len(self.feasible_buffer),
            "family_stats": self.stats.snapshot(),
        }
        with open(self.mechanism_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        if self.meter.exhausted():
            self.control.should_training_stop = True

        return output

    # -----------------------------------------------------------------
    # Losses
    # -----------------------------------------------------------------

    def _score_edge_pairs(self, model: torch.nn.Module, edges: Sequence[Edge]) -> tuple[torch.Tensor, torch.Tensor]:
        """Delta_theta(e) (ConPO.pdf section 2.4) and age(e) for a batch of edges.

        One policy forward pass per response (2 per edge, gradient-tracked);
        logp_ref_*/len_* come from each edge's cache, filled once at
        admission by `_generate_and_score_completions`.

        ponytail: one edge at a time (2*len(edges) separate forward calls,
        batch size 1 each) instead of padding the batch into one call; add
        batched scoring if per-step throughput on the real GPU run measures
        as a bottleneck.
        """
        logp_plus, logp_minus, ref_plus, ref_minus, len_plus, len_minus, ages = [], [], [], [], [], [], []
        for edge in edges:
            lp_plus, _ = score_response(model, self.processing_class, edge.prompt, edge.y_plus)
            lp_minus, _ = score_response(model, self.processing_class, edge.prompt, edge.y_minus)
            logp_plus.append(lp_plus)
            logp_minus.append(lp_minus)
            ref_plus.append(edge.logp_ref_plus)
            ref_minus.append(edge.logp_ref_minus)
            len_plus.append(edge.len_plus)
            len_minus.append(edge.len_minus)
            ages.append(self.edge_buffer.age(edge, self.state.global_step))
        device = logp_plus[0].device
        q_plus = losses.q_theta(
            torch.stack(logp_plus),
            torch.tensor(ref_plus, dtype=torch.float32, device=device),
            torch.tensor(len_plus, dtype=torch.long, device=device),
        )
        q_minus = losses.q_theta(
            torch.stack(logp_minus),
            torch.tensor(ref_minus, dtype=torch.float32, device=device),
            torch.tensor(len_minus, dtype=torch.long, device=device),
        )
        ages_t = torch.tensor(ages, dtype=torch.long, device=device)
        return q_plus - q_minus, ages_t

    def _prompt_margins(self, model: torch.nn.Module, edges: Sequence[Edge]) -> tuple[torch.Tensor, torch.Tensor]:
        """m_plus = m_theta(x^+A), m_minus = m_theta(x^-A) (ConPO.pdf section 2.5) for single-family edges.

        x^+A is `edge.prompt` itself (counterfactual_prompts leaves it
        unchanged), so its reference log-probs are already cached on the
        edge; x^-A's reference log-probs are computed fresh here, under
        `disable_adapter()`, since no edge caches them.
        """
        unwrapped = self.accelerator.unwrap_model(model)
        m_plus_terms: list[torch.Tensor] = []
        m_minus_terms: list[torch.Tensor] = []
        for edge in edges:
            row = self.rows_by_key[edge.prompt_id]
            family = next(iter(edge.families))
            target_index = next(i for i, c in enumerate(row.constraints) if c.iid == family)
            x_plus, x_minus = prompts.counterfactual_prompts(row, target_index)

            lp_plus_xplus, _ = score_response(model, self.processing_class, x_plus, edge.y_plus)
            lp_minus_xplus, _ = score_response(model, self.processing_class, x_plus, edge.y_minus)
            ref_plus_xplus = torch.tensor(edge.logp_ref_plus, dtype=lp_plus_xplus.dtype, device=lp_plus_xplus.device)
            ref_minus_xplus = torch.tensor(
                edge.logp_ref_minus, dtype=lp_minus_xplus.dtype, device=lp_minus_xplus.device
            )
            q_plus_xplus = losses.q_theta(
                lp_plus_xplus, ref_plus_xplus, torch.tensor(edge.len_plus, device=lp_plus_xplus.device)
            )
            q_minus_xplus = losses.q_theta(
                lp_minus_xplus, ref_minus_xplus, torch.tensor(edge.len_minus, device=lp_minus_xplus.device)
            )
            m_plus_terms.append(q_plus_xplus - q_minus_xplus)

            lp_plus_xminus, len_p_xminus = score_response(model, self.processing_class, x_minus, edge.y_plus)
            lp_minus_xminus, len_m_xminus = score_response(model, self.processing_class, x_minus, edge.y_minus)
            with torch.no_grad(), unwrapped.disable_adapter():
                ref_plus_xminus, _ = score_response(unwrapped, self.processing_class, x_minus, edge.y_plus)
                ref_minus_xminus, _ = score_response(unwrapped, self.processing_class, x_minus, edge.y_minus)
            q_plus_xminus = losses.q_theta(
                lp_plus_xminus, ref_plus_xminus.detach(), torch.tensor(len_p_xminus, device=lp_plus_xminus.device)
            )
            q_minus_xminus = losses.q_theta(
                lp_minus_xminus, ref_minus_xminus.detach(), torch.tensor(len_m_xminus, device=lp_minus_xminus.device)
            )
            m_minus_terms.append(q_plus_xminus - q_minus_xminus)

        return torch.stack(m_plus_terms), torch.stack(m_minus_terms)

    def _run_probe(self, model: torch.nn.Module, clean: torch.Tensor, current_update: int) -> None:
        """T_c gradient-alignment probe (ConPO.pdf section 2.6): cosine between
        L_resp's gradient on a probe batch of the due family and L_clean's
        gradient on the current inputs, w.r.t. the LoRA parameters only.

        `torch.autograd.grad` (not `.backward()`) never touches `.grad`
        attributes, so this has no optimiser side effects; `retain_graph`
        keeps both graphs alive for the real `total.backward()` still to come.
        """
        families = sorted(self.edge_buffer.counts_by_family())
        if not families:
            return
        family = selection.next_probe_family(self.stats, families, current_update)
        if family is None or not selection.probe_due(self.stats, family, current_update, config.PROBE_PERIOD_S):
            return
        pool = self.edge_buffer.sample(len(self.edge_buffer), current_update, self.rng, balanced=False)
        probe_edges = [e for e in pool if family in e.families][: config.PROBE_SIZE]
        if not probe_edges:
            return
        lora_params = [p for name, p in model.named_parameters() if "lora_" in name and p.requires_grad]
        if not lora_params:
            return

        delta, ages = self._score_edge_pairs(model, probe_edges)
        probe_loss, _ = losses.resp_loss(delta, ages, tau=config.TAU, gamma=config.GAMMA, kappa=config.KAPPA)
        probe_grad = torch.autograd.grad(probe_loss, lora_params, retain_graph=True, allow_unused=True)
        clean_grad = torch.autograd.grad(clean, lora_params, retain_graph=True, allow_unused=True)
        probe_grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(probe_grad, lora_params)]
        clean_grad = [g if g is not None else torch.zeros_like(p) for g, p in zip(clean_grad, lora_params)]
        cosine = selection.grad_cosine(probe_grad, clean_grad)
        self.stats.record_probe(family, cosine, current_update)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, object],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if return_outputs:
            raise ValueError("ConPOTrainer does not support return_outputs")

        clean = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
        total = config.LAMBDA_G * clean
        current_update = self.state.global_step
        stats: dict[str, float] = {"clean_loss": clean.item()}

        if self.arm.resp_loss and len(self.edge_buffer) > 0:
            resp_edges = self.edge_buffer.sample(config.EDGE_PAIRS_PER_STEP, current_update, self.rng, balanced=True)
            if resp_edges:
                delta, ages = self._score_edge_pairs(model, resp_edges)
                loss_resp, resp_stats = losses.resp_loss(
                    delta, ages, tau=config.TAU, gamma=config.GAMMA, kappa=config.KAPPA
                )
                total = total + loss_resp
                stats["resp_loss"] = loss_resp.item()
                stats["resp_mean_age"] = ages.float().mean().item()
                stats.update({f"resp_{k}": v for k, v in resp_stats.items()})

                margins_by_direction_family: dict[tuple[str, str], list[float]] = {}
                for edge, margin in zip(resp_edges, delta.detach().tolist()):
                    family = next(iter(edge.families)) if len(edge.families) == 1 else "multi"
                    margins_by_direction_family.setdefault((edge.direction, family), []).append(margin)
                for (direction, family), values in margins_by_direction_family.items():
                    stats[f"resp_margin_{direction}_{family}"] = sum(values) / len(values)

        if self.arm.prompt_loss and self.overrides.lambda_p > 0 and len(self.edge_buffer) > 0:
            pool = self.edge_buffer.sample(len(self.edge_buffer), current_update, self.rng, balanced=False)
            prompt_edges = [e for e in pool if len(e.families) == 1][: config.PROMPT_PAIRS_PER_STEP]
            if prompt_edges:
                m_plus, m_minus = self._prompt_margins(model, prompt_edges)
                loss_prompt, prompt_stats = losses.prompt_loss(
                    m_plus, m_minus, tau_p=config.TAU_P, lambda_0=config.LAMBDA_0
                )
                total = total + self.overrides.lambda_p * loss_prompt
                stats["prompt_loss"] = loss_prompt.item()
                stats.update({f"prompt_{k}": v for k, v in prompt_stats.items()})

        if self.probe_enabled and len(self.edge_buffer) > 0:
            self._run_probe(model, clean, current_update)

        self.last_loss_stats = stats
        mode = "train" if model.training else "eval"
        for name, value in stats.items():
            self._metrics[mode][name].append(value)

        return total

"""Diagnosis-only instrumentation for the per-token log-prob blow-up.

See /home/dev/project/ethan/.cache/tmp/claude-1001/-home-dev-project-ethan-research/
84bb2180-5c56-4110-9294-50df603afb3c/scratchpad/brief_diag_logp.md for the
full brief. This script does not edit conpo/ or scripts/train.py: it
monkeypatches three methods at import time, then reuses scripts.train.main()
for construction (least code -- see the module docstring below for why the
in-process route was picked over a subprocess + sitecustomize patch module).

Hooks (TRL 1.9.2, trl/trainer/grpo_trainer.py in the venv):
  - GRPOTrainer._generate_and_score_completions (class defined at line 2260,
    returns at ~2837): captures prompt_ids/prompt_mask/completion_ids/
    completion_mask/old_per_token_logps/ref_per_token_logps/
    sampling_per_token_logps for the whole generation-batch (one call per
    optimiser step here, since GRPOConfig.steps_per_generation defaults to
    gradient_accumulation_steps -- verified by reading _prepare_inputs,
    line 1555). ConPOTrainer._generate_and_score_completions (conpo/
    trainer.py:432) calls super() for this and does not touch `output`
    itself, so patching the TRL class method also covers ConPO's override.
  - GRPOTrainer._get_per_token_logps_and_entropies (line 1435): every call
    TRL makes to get per-token log-probs from a model, including the
    no-grad old/ref calls inside _generate_and_score_completions (lines
    2565, 2628, 2644) and the with-grad policy call inside _compute_loss
    (line 3001, called from GRPOTrainer.compute_loss, called from
    ConPOTrainer.compute_loss's `clean = super().compute_loss(...)` at
    conpo/trainer.py:1255). ConPO's own pair-loss forwards
    (_score_edge_pairs, _prompt_margins, flush_pending_veto's ref scoring)
    never call this method: they use score_response ->
    losses.sequence_logprobs instead (conpo/trainer.py:139), a separate
    code path, batch size 1 per call. This is verified at runtime too: see
    "call_batch_sizes" in the summary output.
  - ConPOTrainer.compute_loss (conpo/trainer.py:1245): wrapped only to mark
    which _get_per_token_logps_and_entropies call within one micro-batch is
    the real with-grad policy forward, since conpo/trainer.py:1323's T_c
    probe (_run_probe -> _resample_clean_loss -> super().compute_loss) can
    trigger a second, unrelated with-grad call on a subsample of the same
    micro-batch. Only the first grad-tracked call after entering
    ConPOTrainer.compute_loss is recorded.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path("/home/dev/project/ethan/ConPO")
SRC_CHECKPOINT = REPO / "runs/full_s42/checkpoint-75"
OUT_DIR = Path("/node-local-cache/cache/conpo-runs/diag_full_ckpt75")
DIAG_OUT = Path(
    os.environ.get(
        "DIAG_OUT_DIR",
        "/home/dev/project/ethan/.cache/tmp/claude-1001/"
        "-home-dev-project-ethan-research/"
        "84bb2180-5c56-4110-9294-50df603afb3c/scratchpad/diag_logp_out",
    )
)
NATS_THRESHOLD = 5.0


def _copy_checkpoint() -> None:
    """Copies the NFS checkpoint-75 under OUT_DIR/checkpoint-75 (node-local),
    skipping the copy if a prior attempt already finished it (retries must
    not re-copy 1.5 GB every time). Never touches full_s42/ifrlvr_s42.
    """
    dst = OUT_DIR / "checkpoint-75"
    if (dst / "trainer_state.json").exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SRC_CHECKPOINT, dst)


# ---------------------------------------------------------------------------
# Instrumentation state (module-level: the monkeypatches close over this)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {
    "cycle": None,
    "cycle_count": 0,
    "capturing_clean": True,
    "call_log": [],  # (batch_size, requires_grad) for every _get_per_token_logps_and_entropies call
    "rows": [],
    "unmatched": 0,
    "tokenizer": None,
    "max_completion_tokens": None,
}


def _install_patches() -> None:
    import trl.trainer.grpo_trainer as grpo_mod

    import conpo.trainer as conpo_trainer_mod

    orig_generate = grpo_mod.GRPOTrainer._generate_and_score_completions
    orig_logps = grpo_mod.GRPOTrainer._get_per_token_logps_and_entropies
    orig_compute_loss = conpo_trainer_mod.ConPOTrainer.compute_loss

    def patched_generate(self: Any, inputs: list[dict[str, Any]]) -> dict[str, Any]:
        output = orig_generate(self, inputs)
        keys = [str(x.get("key", "")) for x in inputs]
        _state["cycle_count"] += 1
        prompt_ids = output["prompt_ids"].detach().cpu()
        completion_ids = output["completion_ids"].detach().cpu()
        _state["cycle"] = {
            "cycle": _state["cycle_count"],
            "optimizer_step": int(self.state.global_step) + 1,
            "keys": keys,
            "prompt_mask": output["prompt_mask"].detach().cpu(),
            "completion_mask": output["completion_mask"].detach().cpu(),
            "completion_ids": completion_ids,
            "prompt_completion_ids": torch.cat([prompt_ids, completion_ids], dim=1),
            "old": _to_cpu(output.get("old_per_token_logps")),
            "ref": _to_cpu(output.get("ref_per_token_logps")),
            "sampling": _to_cpu(output.get("sampling_per_token_logps")),
        }
        return output

    def patched_compute_loss(
        self: Any,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        _state["capturing_clean"] = True
        _state["tokenizer"] = self.processing_class
        _state["max_completion_tokens"] = getattr(self.args, "max_completion_length", None)
        return orig_compute_loss(
            self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )

    def patched_logps(
        self: Any, model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        out = orig_logps(self, model, input_ids, attention_mask, *args, **kwargs)
        per_token_logps = out[0]
        requires_grad = bool(per_token_logps.requires_grad)
        _state["call_log"].append((int(input_ids.shape[0]), requires_grad))
        if requires_grad and _state["capturing_clean"]:
            _state["capturing_clean"] = False
            _record_rows(input_ids.detach().cpu(), per_token_logps.detach().cpu())
        return out

    grpo_mod.GRPOTrainer._generate_and_score_completions = patched_generate
    grpo_mod.GRPOTrainer._get_per_token_logps_and_entropies = patched_logps
    conpo_trainer_mod.ConPOTrainer.compute_loss = patched_compute_loss


def _to_cpu(t: torch.Tensor | None) -> torch.Tensor | None:
    return t.detach().cpu() if t is not None else None


def _max_in(gap: torch.Tensor, mask: torch.Tensor) -> tuple[int, float]:
    """argmax of `gap` restricted to positions where `mask` is True. Returns
    (-1, 0.0) if no position is masked in.
    """
    if not bool(mask.any()):
        return -1, 0.0
    masked = torch.where(mask, gap, torch.full_like(gap, -1.0))
    pos = int(torch.argmax(masked))
    return pos, float(gap[pos])


def _record_rows(mb_input_ids: torch.Tensor, mb_policy_logps: torch.Tensor) -> None:
    """Matches every row of one compute_loss micro-batch back to its row in
    the current generation cycle (row order differs: _prepare_inputs shuffles
    the cycle before splitting into micro-batches, so matching is by exact
    input_ids content, not position) and records the four log-probs.
    """
    cycle = _state["cycle"]
    if cycle is None or cycle["old"] is None or cycle["ref"] is None or cycle["sampling"] is None:
        return
    full_ids = cycle["prompt_completion_ids"]
    n_completion = mb_policy_logps.shape[1]
    max_cap = _state["max_completion_tokens"]
    tok = _state["tokenizer"]

    for row_in_mb in range(mb_input_ids.shape[0]):
        row_ids = mb_input_ids[row_in_mb]
        if row_ids.shape[0] != full_ids.shape[1]:
            _state["unmatched"] += 1
            continue
        match = (full_ids == row_ids.unsqueeze(0)).all(dim=1)
        hits = match.nonzero(as_tuple=True)[0]
        if hits.numel() == 0:
            _state["unmatched"] += 1
            continue
        cycle_row = int(hits[0])

        comp_mask = cycle["completion_mask"][cycle_row][:n_completion]
        length = int(comp_mask.sum().item())
        if length == 0:
            continue
        valid = comp_mask.bool()

        vllm = cycle["sampling"][cycle_row][:n_completion]
        old = cycle["old"][cycle_row][:n_completion]
        ref = cycle["ref"][cycle_row][:n_completion]
        policy = mb_policy_logps[row_in_mb]
        comp_ids = cycle["completion_ids"][cycle_row][:n_completion]

        gap_vllm_old = (vllm - old).abs()
        gap_ref_policy = (ref - policy).abs()

        pos_a, val_a = _max_in(gap_vllm_old, valid)
        pos_b, val_b = _max_in(gap_ref_policy, valid)

        idx = torch.arange(n_completion)
        first3 = valid & (idx < 3)
        last3 = valid & (idx >= max(length - 3, 0))
        _, val_a_first3 = _max_in(gap_vllm_old, first3)
        _, val_a_last3 = _max_in(gap_vllm_old, last3)
        _, val_b_first3 = _max_in(gap_ref_policy, first3)
        _, val_b_last3 = _max_in(gap_ref_policy, last3)

        main_pos, max_gap = (pos_a, val_a) if val_a >= val_b else (pos_b, val_b)
        four = {
            "vllm": float(vllm[main_pos]),
            "old": float(old[main_pos]),
            "ref": float(ref[main_pos]),
            "policy": float(policy[main_pos]),
        }
        deviation = {k: sum(abs(v - four[other]) for other in four if other != k) for k, v in four.items()}
        outlier = max(deviation, key=lambda k: deviation[k])

        prompt_len = int(cycle["prompt_mask"][cycle_row].sum().item())
        left_pad = int(cycle["prompt_mask"].shape[1] - prompt_len)
        token_id = int(comp_ids[main_pos])
        window = range(max(main_pos - 3, 0), min(main_pos + 4, length))
        neighbor_ids = [int(comp_ids[i]) for i in window]

        _state["rows"].append(
            {
                "optimizer_step": cycle["optimizer_step"],
                "key": cycle["keys"][cycle_row],
                "cycle_row_index": cycle_row,
                "microbatch_row_index": row_in_mb,
                "prompt_length": prompt_len,
                "left_padding": left_pad,
                "completion_length": length,
                "hit_cap": bool(max_cap is not None and length >= max_cap),
                "padded_seq_length": int(full_ids.shape[1]),
                "gap_vllm_old_max": val_a,
                "gap_vllm_old_pos": pos_a,
                "gap_vllm_old_first3_max": val_a_first3,
                "gap_vllm_old_last3_max": val_a_last3,
                "gap_ref_policy_max": val_b,
                "gap_ref_policy_pos": pos_b,
                "gap_ref_policy_first3_max": val_b_first3,
                "gap_ref_policy_last3_max": val_b_last3,
                "max_gap": max_gap,
                "outlier_component": outlier,
                "outlier_position": main_pos,
                "outlier_token_id": token_id,
                "outlier_token_text": tok.decode([token_id]),
                "neighbor_token_ids": neighbor_ids,
                "neighbor_tokens_text": [tok.decode([i]) for i in neighbor_ids],
                "logp_vllm": four["vllm"],
                "logp_old": four["old"],
                "logp_ref": four["ref"],
                "logp_policy": four["policy"],
            }
        )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _write_outputs() -> dict[str, Any]:
    DIAG_OUT.mkdir(parents=True, exist_ok=True)
    rows = _state["rows"]
    jsonl_path = DIAG_OUT / "diag_logp.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(row["optimizer_step"], []).append(row)

    per_step_summary = {}
    for step, step_rows in sorted(by_step.items()):
        outliers = [r for r in step_rows if r["max_gap"] > NATS_THRESHOLD]
        buckets = {"first_token": 0, "inside": 0, "last_token": 0, "at_cap": 0}
        for r in outliers:
            pos = r["outlier_position"]
            if r["hit_cap"] and pos == r["completion_length"] - 1:
                buckets["at_cap"] += 1
            elif pos == 0:
                buckets["first_token"] += 1
            elif pos == r["completion_length"] - 1:
                buckets["last_token"] += 1
            else:
                buckets["inside"] += 1
        outlier_component_counts: dict[str, int] = {}
        for r in outliers:
            outlier_component_counts[r["outlier_component"]] = outlier_component_counts.get(r["outlier_component"], 0) + 1
        per_step_summary[step] = {
            "sequences": len(step_rows),
            "outlier_sequences": len(outliers),
            "position_histogram": buckets,
            "outlier_component_counts": outlier_component_counts,
            "outlier_prompt_lengths": sorted({r["prompt_length"] for r in outliers}),
            "outlier_left_paddings": sorted({r["left_padding"] for r in outliers}),
            "outlier_hit_cap_count": sum(1 for r in outliers if r["hit_cap"]),
            "outlier_cycle_row_indices": [r["cycle_row_index"] for r in outliers],
        }

    call_log = _state["call_log"]
    call_batch_sizes: dict[str, int] = {}
    for size, grad in call_log:
        tag = f"{size}_{'grad' if grad else 'nograd'}"
        call_batch_sizes[tag] = call_batch_sizes.get(tag, 0) + 1

    top_outliers = sorted(rows, key=lambda r: -r["max_gap"])[:3]

    summary = {
        "steps_run": sorted(by_step.keys()),
        "sequences_per_step": {step: len(v) for step, v in by_step.items()},
        "nats_threshold": NATS_THRESHOLD,
        "per_step": per_step_summary,
        "call_log_batch_sizes": call_batch_sizes,
        "total_get_per_token_logps_calls": len(call_log),
        "unmatched_rows": _state["unmatched"],
        "top_outlier_examples": top_outliers,
    }
    (DIAG_OUT / "diag_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("RUN_ARM", "full")
    os.environ.setdefault("RUN_SEED", "42")
    os.environ.setdefault("RUN_STEPS", "78")
    os.environ.setdefault("RUN_RESUME", "1")
    os.environ.setdefault("RUN_OUT_DIR", str(OUT_DIR))
    os.environ.setdefault("RUN_USE_VLLM", "1")
    os.environ.setdefault("RUN_SUBSET", str(REPO / "subsets/train_15000_seed42.keys"))

    _copy_checkpoint()
    _install_patches()

    sys.argv = [sys.argv[0]]  # train.main() parses argv itself (--tiny only)
    # `scripts/` has no __init__.py, so this is a sibling-module import, not a
    # package import: it relies on sys.path[0] being scripts/ itself, true for
    # `python /path/to/scripts/diag_logp.py` (Python sets sys.path[0] to the
    # running script's own directory). Also strips a leading '' from sys.path
    # first: under some invocations sys.path[0] is '' (cwd) instead, and one
    # cwd this repo is checked out under (research/handoff) holds a
    # same-named `conpo` symlink to the ConPO repo ROOT (not the `conpo`
    # package), which shadows the real editable-installed package and breaks
    # `from conpo.data import ...` inside train.py (verified while writing
    # this script). Belt-and-suspenders: harmless when sys.path[0] is already
    # scripts/.
    if sys.path and sys.path[0] == "":
        sys.path.pop(0)
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import train as train_script

    try:
        train_script.main()
    finally:
        summary = _write_outputs()
        print(f"diag: {len(_state['rows'])} sequence rows, steps {summary['steps_run']}")
        print(f"diag: outputs under {DIAG_OUT}")


if __name__ == "__main__":
    main()

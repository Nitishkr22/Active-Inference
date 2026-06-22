"""
evaluate_v5.py — Closed-loop AIF evaluation for WorldModelV5.

Two inference modes:
  "history"  — same as V4: accumulate obs/action history, call forward_filter
               each step with a window.  Fair apples-to-apples comparison with V4c.
  "online"   — V5-new: cache GRU hidden state, call forward_step_online each step.
               O(1) per step — true real-time mode.

Outputs:
  - Per-trial CSV
  - JSON summary (includes mean_ms_per_step for timing comparison)
"""

from __future__ import annotations

import json
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from model_v5 import WorldModelV5, ModelV5Config
from simulator import TinyIndoorEnv, Pose, StepResult
from aif_planner_v3 import (
    AIFPlannerV3, AIFPlannerConfig, IDX_TO_HEADING,
    most_likely_state_from_belief, build_reference_action_sequence, masked_joint_from_belief,
)
from aif_efe_planner_v2 import EFEPlannerV2, EFEPlannerConfig
from aif_efe_planner_v3 import EFEPlannerV3


# ============================================================
# Config
# ============================================================

@dataclass
class EvalConfig:
    checkpoint_path:   str = "./checkpoints_v5f/best_model_v5.pt"
    model_config_json: str = "./checkpoints_v5f/model_config.json"

    num_trials: int = 100
    max_steps:  int = 80
    seed:       int = 42

    # history mode settings
    history_keep:            int = 64
    recent_true_states_keep: int = 20
    warmup_actions:          Tuple[int, ...] = (2, 3, 3, 2)

    # "history" — same window-based inference as V4
    # "online"  — cached GRU, single-step forward_step_online
    inference_mode: str = "online"

    controller: str = "pure_efe"

    output_csv:     str = "./v5f_eval_results.csv"
    output_summary: str = "./v5f_eval_summary.json"

    efe_cfg: EFEPlannerConfig = field(
        default_factory=lambda: EFEPlannerConfig(
            horizon=5,
            max_candidates=243,
            allow_backward=False,
            w_risk=6.0,
            w_terminal_risk=18.0,
            w_min_risk=8.0,
            w_ambiguity=0.05,
            w_collision=14.0,
            w_info_gain=0.25,
            w_context_uncertainty=0.15,
            use_context_uncertainty=True,
            preference_precision=1.00,
            discount=0.90,
            reference_prefix_penalty=0.0,
            adaptive_precision=True,
            adaptive_entropy_threshold=0.10,
            adaptive_risk_min_scale=0.50,
            adaptive_epistemic_boost=6.0,
        )
    )


# ============================================================
# Model loading
# ============================================================

def build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_config_json: Optional[str] = None,
) -> WorldModelV5:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_cfg" in ckpt:
        model_cfg = ModelV5Config(**ckpt["model_cfg"])
    elif model_config_json and Path(model_config_json).exists():
        with open(model_config_json) as f:
            model_cfg = ModelV5Config(**json.load(f))
        print(f"Loaded model_config from JSON: {model_config_json}")
    else:
        raise ValueError("No model_cfg in checkpoint and model_config_json not found.")

    model = WorldModelV5(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded checkpoint : {checkpoint_path}")
    print(f"use_predictive_prior: {model_cfg.use_predictive_prior}")
    print(f"context_dim: {model_cfg.context_dim}")
    return model


# ============================================================
# Environment helpers  (identical to V4)
# ============================================================

def reset_env(env, start_pose, goal_pos):
    pose = Pose(row=start_pose[0], col=start_pose[1], heading=start_pose[2])
    return env.reset(start_pose=pose, goal_pos=goal_pos, use_goal=True)


def step_env(env, action):
    result: StepResult = env.step(action)
    return result.obs, float(result.reward), bool(result.done), result.info


def get_true_pose(info):
    return int(info["row"]), int(info["col"]), str(info["heading"])


def build_reachable_mask(env, num_rows, num_cols):
    mask = torch.zeros(num_rows, num_cols, dtype=torch.float32)
    for r in range(num_rows):
        for c in range(num_cols):
            if 0 <= r < env.rows and 0 <= c < env.cols:
                mask[r, c] = 0.0 if env._is_blocking(r, c) else 1.0
    return mask


def shortest_path_distances(env, goal_rc, num_rows, num_cols, unreachable_cost=50.0):
    out = torch.full((num_rows, num_cols), unreachable_cost, dtype=torch.float32)
    for r in range(num_rows):
        for c in range(num_cols):
            if 0 <= r < env.rows and 0 <= c < env.cols and not env._is_blocking(r, c):
                d = env.shortest_path_length((r, c), goal_rc)
                if d is not None:
                    out[r, c] = float(d)
    return out


def get_free_cells(env):
    return [(r, c) for r in range(env.rows) for c in range(env.cols) if not env._is_blocking(r, c)]


def sample_start_goal_pairs(env, num_trials, seed):
    rng = np.random.default_rng(seed)
    free_cells = get_free_cells(env)
    headings   = ["N", "E", "S", "W"]
    pairs, tries = [], 0
    while len(pairs) < num_trials and tries < num_trials * 100:
        tries += 1
        start_rc = free_cells[int(rng.integers(0, len(free_cells)))]
        goal_rc  = free_cells[int(rng.integers(0, len(free_cells)))]
        if start_rc == goal_rc:
            continue
        d = env.shortest_path_length(start_rc, goal_rc)
        if d is None or d < 3:
            continue
        heading = headings[int(rng.integers(0, 4))]
        pairs.append(((start_rc[0], start_rc[1], heading), goal_rc))
    if len(pairs) < num_trials:
        raise RuntimeError(f"Could only sample {len(pairs)} valid pairs.")
    return pairs


# ============================================================
# Tensor / belief helpers
# ============================================================

def build_obs_tensor(obs_history, device):
    obs_np = np.stack(obs_history, axis=0).astype(np.float32)
    return torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(2).to(device)


def build_act_tensor(action_history, device):
    if not action_history:
        return None
    return torch.tensor(action_history, dtype=torch.long, device=device).unsqueeze(0)


def belief_entropy_masked(row_probs, col_probs, reachable_mask):
    joint, valid_mass = masked_joint_from_belief(row_probs, col_probs, reachable_mask)
    if valid_mass <= 1e-8:
        return 10.0
    eps = 1e-8
    return float((-(joint * joint.clamp_min(eps).log()).sum()).item())


def infer_belief_history(model, obs_history, action_history, device):
    """Batch history inference — same as V4 planner.infer_current_belief."""
    obs_t = build_obs_tensor(obs_history, device)
    act_t = build_act_tensor(action_history, device)
    out = model.forward_filter(obs_t, act_t, skip_recon=True)
    return {
        "row_probs":      out["row_probs_seq"][:, -1, :],
        "col_probs":      out["col_probs_seq"][:, -1, :],
        "heading_probs":  out["heading_probs_seq"][:, -1, :],
        "context":        out["context_seq"][:, -1, :],
        "context_logvar": out["context_logvar_seq"][:, -1, :],
    }


# ============================================================
# Single trial
# ============================================================

def run_single_trial(
    trial_id: int,
    cfg: EvalConfig,
    model: WorldModelV5,
    efe_planner: EFEPlannerV3,
    device: torch.device,
    start_pose: Tuple[int, int, str],
    goal_pos: Tuple[int, int],
) -> Dict[str, Any]:

    env            = TinyIndoorEnv(seed=cfg.seed + trial_id)
    num_rows       = model.cfg.num_row_classes
    num_cols       = model.cfg.num_col_classes
    reachable_mask = build_reachable_mask(env, num_rows, num_cols).to(device)
    dist_t         = shortest_path_distances(env, goal_pos, num_rows, num_cols).to(device)
    shortest_dist  = env.shortest_path_length((start_pose[0], start_pose[1]), goal_pos)

    obs, info   = reset_env(env, start_pose, goal_pos)
    true_pose   = get_true_pose(info)

    obs_history:         List[np.ndarray]           = [np.asarray(obs, dtype=np.float32)]
    action_history:      List[int]                  = []
    recent_true_states:  List[Tuple[int, int, str]] = [true_pose]

    # Online mode state
    h_state:      Optional[torch.Tensor]           = None
    prev_belief:  Optional[Dict[str, torch.Tensor]] = None
    last_action:  Optional[torch.Tensor]            = None   # tensor([a])

    reached_goal      = False
    total_collisions  = 0
    belief_errors     = 0
    inference_times_ms: List[float] = []
    entropy_values:     List[float]  = []
    context_std_values: List[float]  = []
    efe_count = fallback_count = empty_fallback_count = 0

    # ---- Warm-up ----
    for a in cfg.warmup_actions:
        obs, _, done, info = step_env(env, a)
        true_pose = get_true_pose(info)
        obs_history.append(np.asarray(obs, dtype=np.float32))
        action_history.append(a)
        recent_true_states.append(true_pose)
        if info["collision"]:
            total_collisions += 1
        if done or info["reached_goal"]:
            reached_goal = True
            break

        # Online mode: step through warmup frames to warm up GRU state
        if cfg.inference_mode == "online":
            obs_t_wu = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            act_wu   = torch.tensor([a], dtype=torch.long, device=device)
            with torch.no_grad():
                prev_belief, h_state = model.forward_step_online(obs_t_wu, last_action, h_state, prev_belief)
            last_action = act_wu

    if reached_goal:
        return _make_result(
            trial_id, cfg, start_pose, goal_pos, recent_true_states,
            reached_goal=True, planning_steps=0, total_collisions=total_collisions,
            shortest_dist=shortest_dist, belief_errors=0,
            entropy_values=[], context_std_values=[], inference_times_ms=[],
            efe_count=0, fallback_count=0, empty_fallback_count=0,
        )

    # ---- Planning loop ----
    for _ in range(cfg.max_steps):

        # --- Inference ---
        t_inf_start = time.perf_counter()

        if cfg.inference_mode == "history":
            with torch.no_grad():
                belief = infer_belief_history(model, obs_history, action_history, device)

        elif cfg.inference_mode == "online":
            # obs_history[-1] is the latest observation (already stored)
            obs_np = obs_history[-1]
            obs_t  = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0).to(device)   # [1,1,H,W]
            with torch.no_grad():
                belief, h_state = model.forward_step_online(obs_t, last_action, h_state, prev_belief)
            prev_belief = belief

        else:
            raise ValueError(f"Unknown inference_mode: {cfg.inference_mode!r}")

        t_inf_end = time.perf_counter()
        inference_times_ms.append((t_inf_end - t_inf_start) * 1000.0)

        # --- Belief metrics ---
        belief_ent = belief_entropy_masked(belief["row_probs"][0], belief["col_probs"][0], reachable_mask)
        entropy_values.append(belief_ent)

        if "context_logvar" in belief:
            context_std_values.append(torch.exp(0.5 * belief["context_logvar"]).mean().item())

        pred_r, pred_c, pred_h = most_likely_state_from_belief(
            belief["row_probs"][0], belief["col_probs"][0],
            belief["heading_probs"][0], reachable_mask,
        )
        if (pred_r, pred_c, IDX_TO_HEADING[pred_h]) != recent_true_states[-1]:
            belief_errors += 1

        # --- EFE planning ---
        # Score sequences directly using belief dict — planner doesn't re-call forward_filter
        with torch.no_grad():
            scored = efe_planner.score_action_sequences_pure(
                belief=belief,
                recent_true_states=recent_true_states,
            )

        best_seq = scored.get("best_sequence") or []
        if not best_seq:
            best_seq = [0]
            empty_fallback_count += 1
        else:
            efe_count += 1

        action = int(best_seq[0])

        obs, _, done, info = step_env(env, action)
        true_pose = get_true_pose(info)

        obs_history.append(np.asarray(obs, dtype=np.float32))
        action_history.append(action)
        recent_true_states.append(true_pose)

        if cfg.inference_mode == "history":
            if len(obs_history) > cfg.history_keep:
                obs_history    = obs_history[-cfg.history_keep:]
            if len(action_history) > cfg.history_keep - 1:
                action_history = action_history[-(cfg.history_keep - 1):]
        elif cfg.inference_mode == "online":
            last_action = torch.tensor([action], dtype=torch.long, device=device)

        if len(recent_true_states) > cfg.recent_true_states_keep:
            recent_true_states = recent_true_states[-cfg.recent_true_states_keep:]

        if info["collision"]:
            total_collisions += 1
        if done or info["reached_goal"]:
            reached_goal = True
            break

    return _make_result(
        trial_id, cfg, start_pose, goal_pos, recent_true_states,
        reached_goal=reached_goal,
        planning_steps=len(inference_times_ms),
        total_collisions=total_collisions,
        shortest_dist=shortest_dist,
        belief_errors=belief_errors,
        entropy_values=entropy_values,
        context_std_values=context_std_values,
        inference_times_ms=inference_times_ms,
        efe_count=efe_count,
        fallback_count=fallback_count,
        empty_fallback_count=empty_fallback_count,
    )


def _make_result(
    trial_id, cfg, start_pose, goal_pos, recent_true_states,
    reached_goal, planning_steps, total_collisions,
    shortest_dist, belief_errors, entropy_values,
    context_std_values, inference_times_ms,
    efe_count, fallback_count, empty_fallback_count,
) -> Dict[str, Any]:
    path_eff = (
        float(shortest_dist) / float(planning_steps)
        if (reached_goal and shortest_dist and planning_steps > 0) else 0.0
    )
    final = recent_true_states[-1]
    return {
        "trial":                trial_id,
        "inference_mode":       cfg.inference_mode,
        "controller":           cfg.controller,
        "success":              int(reached_goal),
        "timeout":              int(not reached_goal),
        "steps":                planning_steps,
        "collisions":           total_collisions,
        "start_row":            start_pose[0],
        "start_col":            start_pose[1],
        "start_heading":        start_pose[2],
        "goal_row":             goal_pos[0],
        "goal_col":             goal_pos[1],
        "final_row":            final[0],
        "final_col":            final[1],
        "final_heading":        final[2],
        "shortest_dist":        shortest_dist if shortest_dist is not None else -1,
        "path_efficiency":      path_eff,
        "belief_error_rate":    belief_errors / max(planning_steps, 1),
        "mean_entropy":         float(np.mean(entropy_values)) if entropy_values else 0.0,
        "mean_context_std":     float(np.mean(context_std_values)) if context_std_values else 0.0,
        "mean_ms_per_step":     float(np.mean(inference_times_ms)) if inference_times_ms else 0.0,
        "efe_count":            efe_count,
        "fallback_count":       fallback_count,
        "empty_fallback_count": empty_fallback_count,
    }


# ============================================================
# Summary
# ============================================================

def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n         = len(results)
    successes = [r for r in results if r["success"] == 1]
    failures  = [r for r in results if r["success"] == 0]

    def mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "num_trials":             n,
        "inference_mode":         results[0]["inference_mode"] if results else "?",
        "success_rate":           len(successes) / max(n, 1),
        "timeout_rate":           len(failures)  / max(n, 1),
        "avg_steps_successful":   mean([r["steps"] for r in successes]),
        "avg_steps_all":          mean([r["steps"] for r in results]),
        "avg_collisions":         mean([r["collisions"] for r in results]),
        "collision_rate":         mean([float(r["collisions"] > 0) for r in results]),
        "mean_path_efficiency":   mean([r["path_efficiency"] for r in successes]),
        "mean_belief_error_rate": mean([r["belief_error_rate"] for r in results]),
        "mean_entropy":           mean([r["mean_entropy"] for r in results]),
        "mean_context_std":       mean([r["mean_context_std"] for r in results]),
        "mean_ms_per_step":       mean([r["mean_ms_per_step"] for r in results]),
        "total_efe":              sum(r["efe_count"]            for r in results),
        "total_fallback":         sum(r["fallback_count"]       for r in results),
        "total_empty_fallback":   sum(r["empty_fallback_count"] for r in results),
    }

    print()
    print("=" * 80)
    print("AIF V5 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Inference mode          : {summary['inference_mode']}")
    print(f"  Trials                  : {summary['num_trials']}")
    print(f"  Success rate            : {summary['success_rate']:.4f}")
    print(f"  Timeout rate            : {summary['timeout_rate']:.4f}")
    print(f"  Avg steps (success)     : {summary['avg_steps_successful']:.2f}")
    print(f"  Avg steps (all)         : {summary['avg_steps_all']:.2f}")
    print(f"  Avg collisions          : {summary['avg_collisions']:.4f}")
    print(f"  Collision rate          : {summary['collision_rate']:.4f}")
    print(f"  Mean path efficiency    : {summary['mean_path_efficiency']:.4f}")
    print(f"  Mean belief error rate  : {summary['mean_belief_error_rate']:.4f}")
    print(f"  Mean belief entropy     : {summary['mean_entropy']:.4f}")
    print(f"  Mean context std        : {summary['mean_context_std']:.4f}")
    print(f"  Mean ms/step (inference): {summary['mean_ms_per_step']:.1f} ms")

    if failures:
        print(f"\n  First 10 failures:")
        for r in failures[:10]:
            print(
                f"    trial={r['trial']:03d} | "
                f"start=({r['start_row']},{r['start_col']},{r['start_heading']}) | "
                f"goal=({r['goal_row']},{r['goal_col']}) | "
                f"steps={r['steps']} | coll={r['collisions']} | "
                f"ms/step={r['mean_ms_per_step']:.1f}"
            )

    return summary


def save_csv(results, path):
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved per-trial CSV : {path}")


def save_summary_json(summary, path):
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON  : {path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg    = EvalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device      : {device}")
    print(f"Inference mode    : {cfg.inference_mode}")
    print(f"Controller        : {cfg.controller}")
    print(f"Num trials        : {cfg.num_trials}")
    print(f"Max steps         : {cfg.max_steps}")

    model = build_model_from_checkpoint(cfg.checkpoint_path, device, cfg.model_config_json)

    template_env = TinyIndoorEnv(seed=cfg.seed)
    pairs        = sample_start_goal_pairs(template_env, cfg.num_trials, cfg.seed)

    # Build EFE planner once (reachable_mask and dist_t are rebuilt per trial)
    # We use trial 0's env as template for planner construction;
    # each trial rebuilds dist_t internally so goal-specific costs are correct.
    # The planner is constructed fresh per trial in run_single_trial for simplicity.
    efe_planner_template_env = TinyIndoorEnv(seed=cfg.seed)
    template_reachable_mask  = build_reachable_mask(efe_planner_template_env, model.cfg.num_row_classes, model.cfg.num_col_classes).to(device)
    template_dist            = torch.zeros(model.cfg.num_row_classes, model.cfg.num_col_classes, device=device)

    efe_planner = EFEPlannerV3(
        model=model,
        dist_t=template_dist,
        reachable_mask=template_reachable_mask,
        cfg=cfg.efe_cfg,
    )

    results: List[Dict[str, Any]] = []
    t_start = time.time()

    for i, (start_pose, goal_pos) in enumerate(pairs):
        # Rebuild planner with correct goal-specific distances for this trial
        trial_env  = TinyIndoorEnv(seed=cfg.seed + i)
        dist_t_i   = shortest_path_distances(trial_env, goal_pos, model.cfg.num_row_classes, model.cfg.num_col_classes).to(device)
        reach_i    = build_reachable_mask(trial_env, model.cfg.num_row_classes, model.cfg.num_col_classes).to(device)
        efe_planner_i = EFEPlannerV3(model=model, dist_t=dist_t_i, reachable_mask=reach_i, cfg=cfg.efe_cfg)

        result = run_single_trial(
            trial_id=i,
            cfg=cfg,
            model=model,
            efe_planner=efe_planner_i,
            device=device,
            start_pose=start_pose,
            goal_pos=goal_pos,
        )
        results.append(result)

        print(
            f"[{i+1:03d}/{cfg.num_trials:03d}] "
            f"{'OK' if result['success'] else 'FAIL'} | "
            f"steps={result['steps']:02d} | "
            f"coll={result['collisions']} | "
            f"efe={result['efe_count']} | "
            f"ent={result['mean_entropy']:.3f} | "
            f"ctx_std={result['mean_context_std']:.3f} | "
            f"ms/step={result['mean_ms_per_step']:.1f} | "
            f"start=({result['start_row']},{result['start_col']},{result['start_heading']}) | "
            f"goal=({result['goal_row']},{result['goal_col']})"
        )

    elapsed = time.time() - t_start
    summary = summarize_results(results)
    summary["total_wall_clock_seconds"] = round(elapsed, 2)

    save_csv(results, cfg.output_csv)
    save_summary_json(summary, cfg.output_summary)
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

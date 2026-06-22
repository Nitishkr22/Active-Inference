# evaluate_v5_noisy.py
#
# Noise stress-test for WorldModelV5 + EFEPlannerV3 (batched GPU).
# Uses online inference (forward_step_online, O(1) per step with GRU caching).
#
# Noise conditions swept:
#   0. baseline  — no added noise (σ=0)
#   1. gauss_low — Gaussian σ=0.05
#   2. gauss_med — Gaussian σ=0.10
#   3. gauss_hi  — Gaussian σ=0.20
#   4. mask_40   — 40% random pixel masking
#   5. combined  — Gaussian σ=0.10 + 30% masking
#
# Start/goal pairs and distance maps are pre-computed ONCE and reused across
# all noise conditions so results are directly comparable.

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
    AIFPlannerConfig,
    IDX_TO_HEADING,
    most_likely_state_from_belief,
    masked_joint_from_belief,
)
from aif_efe_planner_v2 import EFEPlannerConfig
from aif_efe_planner_v3 import EFEPlannerV3


# ============================================================
# Noise application
# ============================================================

def apply_noise(
    obs: np.ndarray,
    gauss_sigma: float,
    mask_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    out = obs.copy()
    if gauss_sigma > 0.0:
        out += rng.normal(0.0, gauss_sigma, size=out.shape).astype(np.float32)
    if mask_fraction > 0.0:
        mask = rng.random(size=out.shape) < mask_fraction
        out[mask] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ============================================================
# Config
# ============================================================

@dataclass
class NoiseCondition:
    name:          str
    gauss_sigma:   float
    mask_fraction: float

    def label(self) -> str:
        parts = []
        if self.gauss_sigma > 0:
            parts.append(f"σ={self.gauss_sigma:.2f}")
        if self.mask_fraction > 0:
            parts.append(f"mask={int(self.mask_fraction*100)}%")
        return ", ".join(parts) if parts else "clean"


NOISE_CONDITIONS: List[NoiseCondition] = [
    NoiseCondition("baseline",  gauss_sigma=0.00, mask_fraction=0.00),
    NoiseCondition("gauss_low", gauss_sigma=0.05, mask_fraction=0.00),
    NoiseCondition("gauss_med", gauss_sigma=0.10, mask_fraction=0.00),
    NoiseCondition("gauss_hi",  gauss_sigma=0.20, mask_fraction=0.00),
    NoiseCondition("mask_40",   gauss_sigma=0.00, mask_fraction=0.40),
    NoiseCondition("combined",  gauss_sigma=0.10, mask_fraction=0.30),
]


@dataclass
class SweepConfig:
    checkpoint_path:   str = "./checkpoints_v5f/best_model_v5.pt"
    model_config_json: str = "./checkpoints_v5f/model_config.json"

    num_trials:     int   = 25
    max_steps:      int   = 80
    seed:           int   = 42
    warmup_actions: Tuple[int, ...] = (2, 3, 3, 2)

    output_dir: str = "./noise_sweep_outputs_v5f"

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

def build_model(cfg: SweepConfig, device: torch.device) -> WorldModelV5:
    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    if "model_cfg" in ckpt:
        model_cfg = ModelV5Config(**ckpt["model_cfg"])
    elif Path(cfg.model_config_json).exists():
        with open(cfg.model_config_json) as f:
            model_cfg = ModelV5Config(**json.load(f))
    else:
        raise ValueError("No model_cfg in checkpoint and model_config_json not found.")
    model = WorldModelV5(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded: {cfg.checkpoint_path}  use_predictive_prior={model_cfg.use_predictive_prior}")
    return model


# ============================================================
# Environment helpers
# ============================================================

def get_free_cells(env: TinyIndoorEnv) -> List[Tuple[int, int]]:
    return [(r, c) for r in range(env.rows) for c in range(env.cols) if not env._is_blocking(r, c)]


def sample_pairs(
    env: TinyIndoorEnv, num_trials: int, seed: int
) -> List[Tuple[Tuple[int, int, str], Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    free_cells = get_free_cells(env)
    headings = ["N", "E", "S", "W"]
    pairs, tries = [], 0
    while len(pairs) < num_trials and tries < num_trials * 200:
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
        raise RuntimeError(f"Only sampled {len(pairs)} valid pairs.")
    return pairs


def build_reachable_mask(env: TinyIndoorEnv, num_rows: int, num_cols: int) -> torch.Tensor:
    mask = torch.zeros(num_rows, num_cols, dtype=torch.float32)
    for r in range(num_rows):
        for c in range(num_cols):
            if 0 <= r < env.rows and 0 <= c < env.cols:
                mask[r, c] = 0.0 if env._is_blocking(r, c) else 1.0
    return mask


def build_dist_map(
    env: TinyIndoorEnv, goal_rc: Tuple[int, int], num_rows: int, num_cols: int
) -> torch.Tensor:
    out = torch.full((num_rows, num_cols), 50.0, dtype=torch.float32)
    for r in range(num_rows):
        for c in range(num_cols):
            if 0 <= r < env.rows and 0 <= c < env.cols and not env._is_blocking(r, c):
                d = env.shortest_path_length((r, c), goal_rc)
                if d is not None:
                    out[r, c] = float(d)
    return out


def belief_entropy(
    row_probs: torch.Tensor,
    col_probs: torch.Tensor,
    reachable_mask: torch.Tensor,
) -> float:
    joint, valid_mass = masked_joint_from_belief(row_probs, col_probs, reachable_mask)
    if valid_mass <= 1e-8:
        return 10.0
    eps = 1e-8
    return float((-(joint * joint.clamp_min(eps).log()).sum()).item())


# ============================================================
# Single trial under a noise condition (online inference)
# ============================================================

def run_trial(
    trial_id:       int,
    start_pose:     Tuple[int, int, str],
    goal_pos:       Tuple[int, int],
    dist_t:         torch.Tensor,
    reachable_mask: torch.Tensor,
    model:          WorldModelV5,
    efe_planner:    EFEPlannerV3,
    noise:          NoiseCondition,
    cfg:            SweepConfig,
    device:         torch.device,
) -> Dict[str, Any]:
    noise_rng = np.random.default_rng(cfg.seed + trial_id * 1000 + hash(noise.name) % 10000)
    env = TinyIndoorEnv(seed=cfg.seed + trial_id)

    pose = Pose(row=start_pose[0], col=start_pose[1], heading=start_pose[2])
    obs_raw, info = env.reset(start_pose=pose, goal_pos=goal_pos, use_goal=True)
    shortest_dist = env.shortest_path_length((start_pose[0], start_pose[1]), goal_pos)

    def corrupt(o: np.ndarray) -> np.ndarray:
        return apply_noise(np.asarray(o, dtype=np.float32), noise.gauss_sigma, noise.mask_fraction, noise_rng)

    true_poses: List[Tuple[int, int, str]] = [(int(info["row"]), int(info["col"]), str(info["heading"]))]

    reached_goal     = False
    total_collisions = 0
    belief_errors    = 0
    entropy_vals:    List[float] = []
    ctx_std_vals:    List[float] = []
    planning_steps   = 0

    # Online GRU state
    h_state:     Optional[torch.Tensor]            = None
    prev_belief: Optional[Dict[str, torch.Tensor]] = None
    last_action: Optional[torch.Tensor]            = None

    # Seed the GRU with the first observation (no action yet)
    first_obs_t = torch.from_numpy(corrupt(np.asarray(obs_raw, dtype=np.float32))).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        prev_belief, h_state = model.forward_step_online(first_obs_t, last_action, h_state, prev_belief)

    # Warm-up
    for a in cfg.warmup_actions:
        result = env.step(a)
        obs_wu = corrupt(np.asarray(result.obs, dtype=np.float32))
        act_t  = torch.tensor([a], dtype=torch.long, device=device)
        obs_t  = torch.from_numpy(obs_wu).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            prev_belief, h_state = model.forward_step_online(obs_t, act_t, h_state, prev_belief)
        last_action = act_t
        true_poses.append((int(result.info["row"]), int(result.info["col"]), str(result.info["heading"])))
        if result.info["collision"]:
            total_collisions += 1
        if result.done or result.info["reached_goal"]:
            reached_goal = True
            break

    if reached_goal:
        return _result(trial_id, noise.name, start_pose, goal_pos, true_poses, True,
                       0, total_collisions, shortest_dist, 0, [], [])

    # Planning loop
    for _ in range(cfg.max_steps):
        # belief dict from last forward_step_online call
        belief = prev_belief

        ent = belief_entropy(belief["row_probs"][0], belief["col_probs"][0], reachable_mask)
        entropy_vals.append(ent)

        if "context_logvar" in belief:
            ctx_std_vals.append(torch.exp(0.5 * belief["context_logvar"]).mean().item())

        pred_r, pred_c, pred_h = most_likely_state_from_belief(
            belief["row_probs"][0], belief["col_probs"][0],
            belief["heading_probs"][0], reachable_mask,
        )
        if (pred_r, pred_c, IDX_TO_HEADING[pred_h]) != true_poses[-1]:
            belief_errors += 1

        with torch.no_grad():
            scored = efe_planner.score_action_sequences_pure(
                belief=belief, recent_true_states=true_poses,
            )
        best_seq = scored.get("best_sequence") or [0]
        action = int(best_seq[0])

        result = env.step(action)
        obs_noisy = corrupt(np.asarray(result.obs, dtype=np.float32))
        obs_t     = torch.from_numpy(obs_noisy).unsqueeze(0).unsqueeze(0).to(device)
        act_t     = torch.tensor([action], dtype=torch.long, device=device)

        with torch.no_grad():
            prev_belief, h_state = model.forward_step_online(obs_t, act_t, h_state, prev_belief)
        last_action = act_t

        true_poses.append((int(result.info["row"]), int(result.info["col"]), str(result.info["heading"])))
        if len(true_poses) > 20:
            true_poses = true_poses[-20:]

        planning_steps += 1
        if result.info["collision"]:
            total_collisions += 1
        if result.done or result.info["reached_goal"]:
            reached_goal = True
            break

    return _result(trial_id, noise.name, start_pose, goal_pos, true_poses,
                   reached_goal, planning_steps, total_collisions, shortest_dist,
                   belief_errors, entropy_vals, ctx_std_vals)


def _result(
    trial_id, noise_name, start_pose, goal_pos, true_poses,
    reached_goal, planning_steps, total_collisions, shortest_dist,
    belief_errors, entropy_vals, ctx_std_vals,
) -> Dict[str, Any]:
    path_eff = (
        float(shortest_dist) / float(planning_steps)
        if (reached_goal and shortest_dist and planning_steps > 0)
        else 0.0
    )
    final = true_poses[-1]
    return {
        "trial":             trial_id,
        "noise_condition":   noise_name,
        "success":           int(reached_goal),
        "steps":             planning_steps,
        "collisions":        total_collisions,
        "start_row":         start_pose[0],
        "start_col":         start_pose[1],
        "start_heading":     start_pose[2],
        "goal_row":          goal_pos[0],
        "goal_col":          goal_pos[1],
        "final_row":         final[0],
        "final_col":         final[1],
        "shortest_dist":     shortest_dist if shortest_dist is not None else -1,
        "path_efficiency":   path_eff,
        "belief_error_rate": belief_errors / max(planning_steps, 1),
        "mean_entropy":      float(np.mean(entropy_vals)) if entropy_vals else 0.0,
        "mean_context_std":  float(np.mean(ctx_std_vals)) if ctx_std_vals else 0.0,
    }


# ============================================================
# Run one noise condition
# ============================================================

def run_condition(
    noise:          NoiseCondition,
    pairs:          List[Tuple[Tuple[int, int, str], Tuple[int, int]]],
    dist_maps:      Dict[Tuple[int, int], torch.Tensor],
    reachable_mask: torch.Tensor,
    model:          WorldModelV5,
    cfg:            SweepConfig,
    device:         torch.device,
) -> List[Dict[str, Any]]:
    num_rows = model.cfg.num_row_classes
    num_cols = model.cfg.num_col_classes

    efe_planner = EFEPlannerV3(
        model=model,
        dist_t=torch.zeros(num_rows, num_cols, device=device),
        reachable_mask=reachable_mask,
        cfg=cfg.efe_cfg,
    )

    results = []
    for i, (start_pose, goal_pos) in enumerate(pairs):
        dist_t = dist_maps[goal_pos].to(device)
        efe_planner.dist_t = dist_t
        efe_planner.neg_log_pref = efe_planner._build_negative_log_preferences(
            dist_t=dist_t, reachable_mask=reachable_mask
        )

        r = run_trial(
            trial_id=i, start_pose=start_pose, goal_pos=goal_pos,
            dist_t=dist_t, reachable_mask=reachable_mask,
            model=model, efe_planner=efe_planner,
            noise=noise, cfg=cfg, device=device,
        )
        results.append(r)
        status = "OK  " if r["success"] else "FAIL"
        print(
            f"  [{i+1:02d}/{len(pairs):02d}] {status} | "
            f"steps={r['steps']:02d} | "
            f"ent={r['mean_entropy']:.3f} | "
            f"ctx_std={r['mean_context_std']:.3f} | "
            f"bel_err={r['belief_error_rate']:.2f}"
        )
    return results


# ============================================================
# Summary & output
# ============================================================

def condition_summary(results: List[Dict[str, Any]], noise: NoiseCondition) -> Dict[str, Any]:
    n         = len(results)
    successes = [r for r in results if r["success"] == 1]

    def mean(vals):
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "noise_condition":        noise.name,
        "noise_label":            noise.label(),
        "gauss_sigma":            noise.gauss_sigma,
        "mask_fraction":          noise.mask_fraction,
        "num_trials":             n,
        "success_rate":           len(successes) / max(n, 1),
        "avg_steps_success":      mean([r["steps"] for r in successes]),
        "avg_steps_all":          mean([r["steps"] for r in results]),
        "collision_rate":         mean([float(r["collisions"] > 0) for r in results]),
        "mean_path_efficiency":   mean([r["path_efficiency"] for r in successes]),
        "mean_belief_error_rate": mean([r["belief_error_rate"] for r in results]),
        "mean_entropy":           mean([r["mean_entropy"] for r in results]),
        "mean_context_std":       mean([r["mean_context_std"] for r in results]),
    }


def print_comparison_table(summaries: List[Dict[str, Any]]) -> None:
    baseline = summaries[0]
    print()
    print("=" * 95)
    print("NOISE STRESS-TEST (V5e) — COMPARISON TABLE")
    print("=" * 95)
    hdr = (
        f"{'Condition':<12} {'Label':<22} {'Success':>7} {'Δ':>6} "
        f"{'Steps':>6} {'BelErr':>7} {'Entropy':>8} {'CtxStd':>8} {'PathEff':>8}"
    )
    print(hdr)
    print("-" * 95)
    for s in summaries:
        delta = s["success_rate"] - baseline["success_rate"]
        delta_str = f"{delta:+.3f}" if s["noise_condition"] != "baseline" else "  —   "
        print(
            f"{s['noise_condition']:<12} {s['noise_label']:<22} "
            f"{s['success_rate']:>7.3f} {delta_str:>6} "
            f"{s['avg_steps_all']:>6.1f} "
            f"{s['mean_belief_error_rate']:>7.3f} "
            f"{s['mean_entropy']:>8.4f} "
            f"{s['mean_context_std']:>8.4f} "
            f"{s['mean_path_efficiency']:>8.4f}"
        )
    print("=" * 95)
    print()
    print("Interpretation guide:")
    print("  Δ       : success rate change vs clean baseline")
    print("  BelErr  : fraction of steps where most-likely belief ≠ true pose")
    print("  Entropy : mean belief entropy per step (higher → more uncertain)")
    print("  CtxStd  : mean context VAE std per step")
    print("  PathEff : steps_optimal / steps_taken for successful trials")


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg    = SweepConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Device       : {device}")
    print(f"Trials/level : {cfg.num_trials}")
    print(f"Max steps    : {cfg.max_steps}")
    print(f"Inference    : online (forward_step_online)")
    print(f"Conditions   : {[c.name for c in NOISE_CONDITIONS]}")
    print()

    model    = build_model(cfg, device)
    num_rows = model.cfg.num_row_classes
    num_cols = model.cfg.num_col_classes

    print("Pre-computing start/goal pairs and distance maps...")
    template_env   = TinyIndoorEnv(seed=cfg.seed)
    reachable_mask = build_reachable_mask(template_env, num_rows, num_cols).to(device)
    pairs          = sample_pairs(template_env, cfg.num_trials, cfg.seed)

    dist_maps: Dict[Tuple[int, int], torch.Tensor] = {}
    for _, goal_pos in pairs:
        if goal_pos not in dist_maps:
            dist_maps[goal_pos] = build_dist_map(template_env, goal_pos, num_rows, num_cols)
    print(f"  {len(pairs)} pairs, {len(dist_maps)} unique distance maps pre-computed.\n")

    all_results:   List[Dict[str, Any]] = []
    all_summaries: List[Dict[str, Any]] = []

    t_total = time.time()
    for noise in NOISE_CONDITIONS:
        print(f"{'='*60}")
        print(f"CONDITION: {noise.name}  ({noise.label()})")
        print(f"{'='*60}")
        t0      = time.time()
        results = run_condition(noise, pairs, dist_maps, reachable_mask, model, cfg, device)
        elapsed = time.time() - t0
        summary = condition_summary(results, noise)
        summary["wall_clock_seconds"] = round(elapsed, 1)

        all_results.extend(results)
        all_summaries.append(summary)

        print(
            f"  → success={summary['success_rate']:.3f}  "
            f"entropy={summary['mean_entropy']:.4f}  "
            f"ctx_std={summary['mean_context_std']:.4f}  "
            f"({elapsed:.0f}s)\n"
        )

        csv_path = f"{cfg.output_dir}/results_{noise.name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    total_elapsed = time.time() - t_total

    print_comparison_table(all_summaries)

    summary_path = f"{cfg.output_dir}/noise_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"conditions": all_summaries, "total_wall_clock_seconds": round(total_elapsed, 1)},
            f, indent=2,
        )
    print(f"Saved summary : {summary_path}")

    combined_path = f"{cfg.output_dir}/all_results_combined.csv"
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Saved combined: {combined_path}")
    print(f"\nTotal time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()

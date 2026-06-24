"""
evaluate_compressed.py — Closed-loop navigation eval for WorldModelCompressed.

Usage:
    python evaluate_compressed.py --checkpoint ./checkpoints/CL1/best_model.pt
    python evaluate_compressed.py --checkpoint ./checkpoints/CL2/best_model.pt --trials 100 --noise_sigma 0.10
    python evaluate_compressed.py --checkpoint ./checkpoints/CL1/best_model.pt --mask_fraction 0.40

Outputs:
    results/<config_name>_eval_results.csv
    results/<config_name>_eval_summary.json

The script also reports inference_params and model_size_mb for every run so
ICRA tables can be populated directly from the JSON summaries.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from model_compressed import CompressedModelConfig, WorldModelCompressed
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
    checkpoint_path: str  = "./checkpoints/CL0/best_model.pt"
    num_trials:      int  = 100
    max_steps:       int  = 80
    seed:            int  = 42
    inference_mode:  str  = "online"

    # Noise / corruption for robustness tests
    noise_sigma:    float = 0.0
    mask_fraction:  float = 0.0

    output_dir:     str   = "./results"
    run_label:      str   = ""   # appended to filenames; auto-set if empty

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

def load_model(checkpoint_path: str, device: torch.device) -> WorldModelCompressed:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_cfg" not in ckpt:
        raise ValueError("Checkpoint missing model_cfg. Was it saved by train_compressed.py?")
    model_cfg = CompressedModelConfig(**ckpt["model_cfg"])
    model     = WorldModelCompressed(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded  : {checkpoint_path}")
    print(f"  gru={model_cfg.gru_hidden_dim}, feat={model_cfg.encoder_feat_dim}, "
          f"ctx={model_cfg.context_dim}, depthwise={model_cfg.use_depthwise}")
    print(f"  Total params    : {model.total_param_count:,}")
    print(f"  Inference params: {model.inference_param_count:,}")
    print(f"  Inference size  : {model.model_size_mb:.3f} MB (FP32)")
    return model


# ============================================================
# Environment helpers
# ============================================================

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
    rng       = np.random.default_rng(seed)
    free_cells = get_free_cells(env)
    headings   = ["N", "E", "S", "W"]
    pairs, tries = [], 0
    while len(pairs) < num_trials and tries < num_trials * 100:
        tries += 1
        s = free_cells[int(rng.integers(0, len(free_cells)))]
        g = free_cells[int(rng.integers(0, len(free_cells)))]
        if s == g: continue
        d = env.shortest_path_length(s, g)
        if d is None or d < 3: continue
        pairs.append(((s[0], s[1], headings[int(rng.integers(0, 4))]), g))
    if len(pairs) < num_trials:
        raise RuntimeError(f"Only sampled {len(pairs)} valid pairs.")
    return pairs


def apply_corruption(obs_np: np.ndarray, noise_sigma: float, mask_fraction: float) -> np.ndarray:
    """Apply noise + masking to a single HW frame (float32 [0,1])."""
    obs = obs_np.copy()
    if noise_sigma > 0.0:
        obs = np.clip(obs + np.random.randn(*obs.shape).astype(np.float32) * noise_sigma, 0.0, 1.0)
    if mask_fraction > 0.0:
        mask = np.random.binomial(1, 1.0 - mask_fraction, obs.shape).astype(np.float32)
        obs  = obs * mask
    return obs


def belief_entropy_masked(row_probs, col_probs, reachable_mask):
    joint, valid_mass = masked_joint_from_belief(row_probs, col_probs, reachable_mask)
    if valid_mass <= 1e-8:
        return 10.0
    eps = 1e-8
    return float((-(joint * joint.clamp_min(eps).log()).sum()).item())


# ============================================================
# Single trial
# ============================================================

def run_single_trial(
    trial_id:    int,
    cfg:         EvalConfig,
    model:       WorldModelCompressed,
    device:      torch.device,
    start_pose:  Tuple[int, int, str],
    goal_pos:    Tuple[int, int],
) -> Dict[str, Any]:
    env            = TinyIndoorEnv(seed=cfg.seed + trial_id)
    num_rows       = model.cfg.num_row_classes
    num_cols       = model.cfg.num_col_classes
    reachable_mask = build_reachable_mask(env, num_rows, num_cols).to(device)
    dist_t         = shortest_path_distances(env, goal_pos, num_rows, num_cols).to(device)

    pose_in = Pose(row=start_pose[0], col=start_pose[1], heading=start_pose[2])
    obs, info = env.reset(start_pose=pose_in, goal_pos=goal_pos, use_goal=True)
    true_pose = (int(info["row"]), int(info["col"]), str(info["heading"]))
    shortest_dist = env.shortest_path_length((start_pose[0], start_pose[1]), goal_pos)

    efe_planner = EFEPlannerV3(model=model, dist_t=dist_t, reachable_mask=reachable_mask, cfg=cfg.efe_cfg)

    obs_history:        List[np.ndarray]           = [np.asarray(obs, dtype=np.float32)]
    action_history:     List[int]                  = []
    recent_true_states: List[Tuple[int, int, str]] = [true_pose]

    h_state:     Optional[torch.Tensor]            = None
    prev_belief: Optional[Dict[str, torch.Tensor]] = None
    last_action: Optional[torch.Tensor]            = None

    reached_goal   = False
    total_collisions = 0
    belief_errors    = 0
    inf_times_ms:    List[float] = []
    entropy_values:  List[float] = []
    ctx_std_values:  List[float] = []

    # Warm-up: 4 steps to prime GRU
    for a in (2, 3, 3, 2):
        step_result = env.step(a)
        obs_raw, done, info = step_result.obs, step_result.done, step_result.info
        true_pose = (int(info["row"]), int(info["col"]), str(info["heading"]))
        obs_np    = np.asarray(obs_raw, dtype=np.float32)
        obs_history.append(obs_np)
        action_history.append(a)
        recent_true_states.append(true_pose)
        if info["collision"]: total_collisions += 1
        if done or info["reached_goal"]: reached_goal = True; break

        if cfg.inference_mode == "online":
            obs_c = apply_corruption(obs_np, cfg.noise_sigma, cfg.mask_fraction)
            obs_t = torch.from_numpy(obs_c).unsqueeze(0).unsqueeze(0).to(device)
            act_t = torch.tensor([a], dtype=torch.long, device=device)
            with torch.no_grad():
                prev_belief, h_state = model.forward_step_online(obs_t, last_action, h_state, prev_belief)
            last_action = act_t

    if reached_goal:
        return _make_result(trial_id, start_pose, goal_pos, True, 0, total_collisions,
                            shortest_dist, 0, [], [], [], cfg)

    # Main planning loop
    for _ in range(cfg.max_steps):
        t0 = time.perf_counter()

        obs_np  = obs_history[-1]
        obs_c   = apply_corruption(obs_np, cfg.noise_sigma, cfg.mask_fraction)
        obs_t   = torch.from_numpy(obs_c).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            belief, h_state = model.forward_step_online(obs_t, last_action, h_state, prev_belief)
        prev_belief = belief

        inf_times_ms.append((time.perf_counter() - t0) * 1000.0)

        ent = belief_entropy_masked(belief["row_probs"][0], belief["col_probs"][0], reachable_mask)
        entropy_values.append(ent)
        if "context_logvar" in belief:
            ctx_std_values.append(torch.exp(0.5 * belief["context_logvar"]).mean().item())

        pred_r, pred_c, pred_h = most_likely_state_from_belief(
            belief["row_probs"][0], belief["col_probs"][0], belief["heading_probs"][0], reachable_mask
        )
        if (pred_r, pred_c, IDX_TO_HEADING[pred_h]) != recent_true_states[-1]:
            belief_errors += 1

        with torch.no_grad():
            scored = efe_planner.score_action_sequences_pure(belief=belief, recent_true_states=recent_true_states)

        best_seq = scored.get("best_sequence") or [0]
        action   = int(best_seq[0])

        step_result = env.step(action)
        obs_raw, done, info = step_result.obs, step_result.done, step_result.info
        true_pose = (int(info["row"]), int(info["col"]), str(info["heading"]))
        obs_np    = np.asarray(obs_raw, dtype=np.float32)
        obs_history.append(obs_np)
        action_history.append(action)
        recent_true_states.append(true_pose)
        last_action = torch.tensor([action], dtype=torch.long, device=device)

        if len(recent_true_states) > 20:
            recent_true_states = recent_true_states[-20:]

        if info["collision"]: total_collisions += 1
        if done or info["reached_goal"]:
            reached_goal = True
            break

    return _make_result(
        trial_id, start_pose, goal_pos, reached_goal,
        len(inf_times_ms), total_collisions, shortest_dist, belief_errors,
        entropy_values, ctx_std_values, inf_times_ms, cfg,
    )


def _make_result(
    trial_id, start_pose, goal_pos, reached_goal, planning_steps,
    total_collisions, shortest_dist, belief_errors,
    entropy_values, ctx_std_values, inf_times_ms, cfg,
) -> Dict[str, Any]:
    path_eff = (float(shortest_dist) / float(planning_steps)
                if (reached_goal and shortest_dist and planning_steps > 0) else 0.0)
    return {
        "trial":              trial_id,
        "success":            int(reached_goal),
        "steps":              planning_steps,
        "collisions":         total_collisions,
        "start_row":          start_pose[0],
        "start_col":          start_pose[1],
        "start_heading":      start_pose[2],
        "goal_row":           goal_pos[0],
        "goal_col":           goal_pos[1],
        "shortest_dist":      shortest_dist if shortest_dist is not None else -1,
        "path_efficiency":    path_eff,
        "belief_error_rate":  belief_errors / max(planning_steps, 1),
        "mean_entropy":       float(np.mean(entropy_values)) if entropy_values else 0.0,
        "mean_context_std":   float(np.mean(ctx_std_values)) if ctx_std_values else 0.0,
        "mean_ms_per_step":   float(np.mean(inf_times_ms))   if inf_times_ms   else 0.0,
        "noise_sigma":        cfg.noise_sigma,
        "mask_fraction":      cfg.mask_fraction,
    }


# ============================================================
# Summary
# ============================================================

def summarize_results(results, model, cfg) -> Dict[str, Any]:
    n         = len(results)
    successes = [r for r in results if r["success"]]
    mean      = lambda vs: float(np.mean(vs)) if vs else float("nan")

    summary = {
        "config": {
            "checkpoint":        cfg.checkpoint_path,
            "gru_hidden_dim":    model.cfg.gru_hidden_dim,
            "encoder_feat_dim":  model.cfg.encoder_feat_dim,
            "context_dim":       model.cfg.context_dim,
            "use_depthwise":     model.cfg.use_depthwise,
            "total_params":      model.total_param_count,
            "inference_params":  model.inference_param_count,
            "inference_size_mb": round(model.model_size_mb, 4),
            "noise_sigma":       cfg.noise_sigma,
            "mask_fraction":     cfg.mask_fraction,
        },
        "num_trials":              n,
        "success_rate":            len(successes) / max(n, 1),
        "collision_rate":          mean([float(r["collisions"] > 0) for r in results]),
        "avg_steps_successful":    mean([r["steps"] for r in successes]),
        "avg_steps_all":           mean([r["steps"] for r in results]),
        "mean_path_efficiency":    mean([r["path_efficiency"] for r in successes]),
        "mean_belief_error_rate":  mean([r["belief_error_rate"] for r in results]),
        "mean_entropy":            mean([r["mean_entropy"] for r in results]),
        "mean_context_std":        mean([r["mean_context_std"] for r in results]),
        "mean_ms_per_step":        mean([r["mean_ms_per_step"] for r in results]),
    }

    print()
    print("=" * 75)
    print("COMPRESSED AIF EVALUATION SUMMARY")
    print("=" * 75)
    c = summary["config"]
    print(f"  gru={c['gru_hidden_dim']}, feat={c['encoder_feat_dim']}, ctx={c['context_dim']}, "
          f"depthwise={c['use_depthwise']}")
    print(f"  Inference params : {c['inference_params']:,}  ({c['inference_size_mb']} MB FP32)")
    print(f"  Noise/Mask       : sigma={c['noise_sigma']}, mask={c['mask_fraction']}")
    print(f"  Trials / Success : {summary['num_trials']} / {summary['success_rate']*100:.1f}%")
    print(f"  Collision rate   : {summary['collision_rate']*100:.1f}%")
    print(f"  Avg steps (suc)  : {summary['avg_steps_successful']:.2f}")
    print(f"  Path efficiency  : {summary['mean_path_efficiency']*100:.1f}%")
    print(f"  Belief error rate: {summary['mean_belief_error_rate']*100:.1f}%")
    print(f"  Mean entropy     : {summary['mean_entropy']:.4f}")
    print(f"  Mean context std : {summary['mean_context_std']:.4f}")
    print(f"  Mean ms/step     : {summary['mean_ms_per_step']:.2f} ms")
    return summary


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",     type=str, required=True)
    p.add_argument("--trials",         type=int, default=100)
    p.add_argument("--max_steps",      type=int, default=80)
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--noise_sigma",    type=float, default=0.0)
    p.add_argument("--mask_fraction",  type=float, default=0.0)
    p.add_argument("--output_dir",     type=str, default="./results")
    p.add_argument("--run_label",      type=str, default="")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)

    cfg = EvalConfig(
        checkpoint_path=args.checkpoint,
        num_trials=args.trials,
        max_steps=args.max_steps,
        seed=args.seed,
        noise_sigma=args.noise_sigma,
        mask_fraction=args.mask_fraction,
        output_dir=args.output_dir,
        run_label=args.run_label,
    )

    template_env = TinyIndoorEnv(seed=cfg.seed)
    pairs        = sample_start_goal_pairs(template_env, cfg.num_trials, cfg.seed)

    # Build label for output filenames
    ckpt_stem = Path(cfg.checkpoint_path).parent.name
    noise_tag  = f"_n{cfg.noise_sigma:.2f}" if cfg.noise_sigma > 0 else ""
    mask_tag   = f"_m{cfg.mask_fraction:.2f}" if cfg.mask_fraction > 0 else ""
    extra      = f"_{cfg.run_label}" if cfg.run_label else ""
    label      = f"{ckpt_stem}{noise_tag}{mask_tag}{extra}"

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path  = output_dir / f"{label}_results.csv"
    json_path = output_dir / f"{label}_summary.json"

    results: List[Dict[str, Any]] = []
    t_start = time.time()

    for i, (start_pose, goal_pos) in enumerate(pairs):
        result = run_single_trial(i, cfg, model, device, start_pose, goal_pos)
        results.append(result)
        print(
            f"[{i+1:03d}/{cfg.num_trials}] "
            f"{'OK' if result['success'] else 'FAIL'} | "
            f"steps={result['steps']:02d} | "
            f"coll={result['collisions']} | "
            f"ent={result['mean_entropy']:.3f} | "
            f"ms/step={result['mean_ms_per_step']:.1f}"
        )

    elapsed = time.time() - t_start
    summary = summarize_results(results, model, cfg)
    summary["total_wall_clock_s"] = round(elapsed, 2)

    # Save CSV
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV     : {csv_path}")

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary : {json_path}")
    print(f"Total   : {elapsed:.1f}s")


if __name__ == "__main__":
    main()

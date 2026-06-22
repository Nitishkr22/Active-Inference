# evaluate_aif_v4.py
#
# Full closed-loop AIF agent evaluation using WorldModelV4 + EFEPlannerV2.
# V4-specific additions vs evaluate_aif_v3:
#   - Uses EFEPlannerV2 (context-uncertainty epistemic bonus)
#   - Tracks mean context_std and vfe_kl per step in each trial
#   - Saves a JSON summary alongside the per-trial CSV

from __future__ import annotations

import json
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from model_v4 import WorldModelV4, ModelV4Config
from simulator import TinyIndoorEnv, Pose, StepResult
from aif_planner_v3 import (
    AIFPlannerV3,
    AIFPlannerConfig,
    IDX_TO_HEADING,
    most_likely_state_from_belief,
    build_reference_action_sequence,
    masked_joint_from_belief,
)
from aif_efe_planner_v2 import EFEPlannerV2, EFEPlannerConfig


# ============================================================
# Config
# ============================================================

@dataclass
class EvalConfig:
    checkpoint_path:   str = "./checkpoints_v4e/best_model_v4.pt"
    model_config_json: str = "./checkpoints_v4e/model_config.json"

    num_trials: int   = 100
    max_steps:  int   = 80
    seed:       int   = 42

    history_keep:              int   = 64
    recent_true_states_keep:   int   = 20
    warmup_actions:            Tuple[int, ...] = (2, 3, 3, 2)

    confidence_entropy_threshold: float = 0.22
    fallback_entropy_threshold:   float = 0.80

    # Options: "pure_efe" | "hybrid_efe" | "planner" | "graph_ref"
    controller: str = "pure_efe"
    output_csv:     str = "./aif_v4e_eval_results_pure_efe.csv"
    output_summary: str = "./aif_v4e_eval_summary.json"

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
            # V4: context-uncertainty epistemic bonus
            w_context_uncertainty=0.15,
            use_context_uncertainty=True,
            preference_precision=1.00,
            discount=0.90,
            reference_prefix_penalty=0.0,
            # Adaptive precision weighting (epistemic exploration)
            # When belief entropy is high, risk weight is halved and epistemic
            # weights are boosted up to 6x — implementing genuine uncertainty-driven
            # information-seeking behaviour.
            adaptive_precision=True,
            adaptive_entropy_threshold=0.10,
            adaptive_risk_min_scale=0.50,
            adaptive_epistemic_boost=6.0,
        )
    )

    planner_cfg: AIFPlannerConfig = field(
        default_factory=lambda: AIFPlannerConfig(
            horizon=5,
            allow_backward=True,
            max_candidates=128,
            planning_discount=0.85,
            goal_cost_weight=1.25,
            terminal_goal_weight=18.0,
            min_goal_cost_weight=7.0,
            progress_bonus_weight=7.5,
            regress_penalty_weight=9.0,
            no_progress_penalty=1.5,
            collision_penalty=10.0,
            entropy_penalty=0.05,
            cost_forward=0.00,
            cost_backward=1.80,
            cost_turn=0.20,
            inverse_fb_penalty=4.00,
            inverse_turn_penalty=2.00,
            heading_change_penalty=0.10,
            recent_state_revisit_penalty=4.00,
            immediate_recent_state_penalty=7.00,
            imagination_loop_penalty=5.00,
            imagination_same_state_penalty=6.00,
            reference_prefix_penalty=2.5,
            reference_prefix_decay=0.75,
        )
    )


# ============================================================
# Model loading
# ============================================================

def build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_config_json: Optional[str] = None,
) -> WorldModelV4:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_cfg" in ckpt:
        model_cfg = ModelV4Config(**ckpt["model_cfg"])
    elif model_config_json and Path(model_config_json).exists():
        with open(model_config_json) as f:
            model_cfg = ModelV4Config(**json.load(f))
        print(f"Loaded model_config from JSON: {model_config_json}")
    else:
        raise ValueError("No model_cfg in checkpoint and model_config_json not found.")

    model = WorldModelV4(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded checkpoint : {checkpoint_path}")
    print(f"use_predictive_prior: {model_cfg.use_predictive_prior}")
    return model


# ============================================================
# Environment helpers
# ============================================================

def reset_env(
    env: TinyIndoorEnv,
    start_pose: Tuple[int, int, str],
    goal_pos: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pose = Pose(row=start_pose[0], col=start_pose[1], heading=start_pose[2])
    return env.reset(start_pose=pose, goal_pos=goal_pos, use_goal=True)


def step_env(env: TinyIndoorEnv, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    result: StepResult = env.step(action)
    return result.obs, float(result.reward), bool(result.done), result.info


def get_true_pose(info: Dict[str, Any]) -> Tuple[int, int, str]:
    return int(info["row"]), int(info["col"]), str(info["heading"])


def build_reachable_mask(
    env: TinyIndoorEnv, num_rows: int, num_cols: int
) -> torch.Tensor:
    mask = torch.zeros(num_rows, num_cols, dtype=torch.float32)
    for r in range(num_rows):
        for c in range(num_cols):
            if 0 <= r < env.rows and 0 <= c < env.cols:
                mask[r, c] = 0.0 if env._is_blocking(r, c) else 1.0
    return mask


def shortest_path_distances(
    env: TinyIndoorEnv,
    goal_rc: Tuple[int, int],
    num_rows: int,
    num_cols: int,
    unreachable_cost: float = 50.0,
) -> torch.Tensor:
    out = torch.full((num_rows, num_cols), unreachable_cost, dtype=torch.float32)
    for r in range(num_rows):
        for c in range(num_cols):
            if 0 <= r < env.rows and 0 <= c < env.cols and not env._is_blocking(r, c):
                d = env.shortest_path_length((r, c), goal_rc)
                if d is not None:
                    out[r, c] = float(d)
    return out


def get_free_cells(env: TinyIndoorEnv) -> List[Tuple[int, int]]:
    return [
        (r, c)
        for r in range(env.rows)
        for c in range(env.cols)
        if not env._is_blocking(r, c)
    ]


def sample_start_goal_pairs(
    env: TinyIndoorEnv, num_trials: int, seed: int
) -> List[Tuple[Tuple[int, int, str], Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    free_cells = get_free_cells(env)
    headings = ["N", "E", "S", "W"]
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

def build_obs_tensor(
    obs_history: List[np.ndarray], device: torch.device
) -> torch.Tensor:
    obs_np = np.stack(obs_history, axis=0).astype(np.float32)
    return torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(2).to(device)


def build_act_tensor(
    action_history: List[int], device: torch.device
) -> Optional[torch.Tensor]:
    if not action_history:
        return None
    return torch.tensor(action_history, dtype=torch.long, device=device).unsqueeze(0)


def belief_entropy_masked(
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
# Single trial
# ============================================================

def run_single_trial(
    trial_id: int,
    cfg: EvalConfig,
    model: WorldModelV4,
    device: torch.device,
    start_pose: Tuple[int, int, str],
    goal_pos: Tuple[int, int],
) -> Dict[str, Any]:

    env           = TinyIndoorEnv(seed=cfg.seed + trial_id)
    num_rows      = model.cfg.num_row_classes
    num_cols      = model.cfg.num_col_classes

    reachable_mask = build_reachable_mask(env, num_rows, num_cols).to(device)
    dist_t         = shortest_path_distances(env, goal_pos, num_rows, num_cols).to(device)
    shortest_dist  = env.shortest_path_length((start_pose[0], start_pose[1]), goal_pos)

    planner = AIFPlannerV3(
        model=model, dist_t=dist_t, reachable_mask=reachable_mask, cfg=cfg.planner_cfg,
    )
    efe_planner = EFEPlannerV2(
        model=model, dist_t=dist_t, reachable_mask=reachable_mask, cfg=cfg.efe_cfg,
    )

    obs, info    = reset_env(env, start_pose, goal_pos)
    true_pose    = get_true_pose(info)

    obs_history:         List[np.ndarray]          = [np.asarray(obs, dtype=np.float32)]
    action_history:      List[int]                 = []
    recent_true_states:  List[Tuple[int, int, str]] = [true_pose]

    reached_goal    = False
    total_collisions = 0
    belief_errors   = 0

    entropy_values:      List[float] = []
    context_std_values:  List[float] = []   # V4: context uncertainty per step
    vfe_kl_values:       List[float] = []   # V4: posterior-vs-prior KL per step

    graph_ref_count = planner_count = efe_count = fallback_count = empty_fallback_count = 0
    planning_steps  = 0

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

    if reached_goal:
        return _make_result(
            trial_id, cfg, start_pose, goal_pos, recent_true_states,
            reached_goal=True, planning_steps=0, total_collisions=total_collisions,
            shortest_dist=shortest_dist, belief_errors=0,
            entropy_values=[], context_std_values=[], vfe_kl_values=[],
            graph_ref_count=0, planner_count=0, efe_count=0,
            fallback_count=0, empty_fallback_count=0,
        )

    # ---- Planning loop ----
    for _ in range(cfg.max_steps):
        obs_t = build_obs_tensor(obs_history, device)
        act_t = build_act_tensor(action_history, device)

        with torch.no_grad():
            belief = efe_planner.infer_current_belief(observations=obs_t, actions=act_t)

        # ---- Standard belief metrics ----
        belief_ent = belief_entropy_masked(
            belief["row_probs"][0], belief["col_probs"][0], reachable_mask
        )
        entropy_values.append(belief_ent)

        pred_r, pred_c, pred_h = most_likely_state_from_belief(
            belief["row_probs"][0], belief["col_probs"][0],
            belief["heading_probs"][0], reachable_mask,
        )
        if (pred_r, pred_c, IDX_TO_HEADING[pred_h]) != recent_true_states[-1]:
            belief_errors += 1

        # ---- V4: context uncertainty ----
        if "context_logvar" in belief:
            ctx_std = torch.exp(0.5 * belief["context_logvar"]).mean().item()
            context_std_values.append(ctx_std)

        # ---- V4: VFE KL from last filter step ----
        if "vfe_kl_last" in belief:
            vfe_kl_values.append(float(belief["vfe_kl_last"]))

        ref_seq_now = build_reference_action_sequence(
            r=pred_r, c=pred_c, h=pred_h,
            dist_t=dist_t, reachable_mask=reachable_mask,
            horizon=cfg.planner_cfg.horizon,
        )

        # ---- Controller ----
        mode = "unknown"

        if cfg.controller == "graph_ref":
            best_seq = ref_seq_now or [0]
            mode = "graph_ref"

        elif cfg.controller == "planner":
            with torch.no_grad():
                scored = planner.score_action_sequences(
                    belief=belief, recent_true_states=recent_true_states,
                )
            best_seq = scored.get("best_sequence") or (ref_seq_now or [0])
            mode = "planner"

        elif cfg.controller == "hybrid_efe":
            if belief_ent < cfg.confidence_entropy_threshold and ref_seq_now:
                best_seq = ref_seq_now
                mode = "graph_ref"
            elif belief_ent < cfg.fallback_entropy_threshold:
                with torch.no_grad():
                    scored = efe_planner.score_action_sequences(
                        belief=belief, recent_true_states=recent_true_states,
                    )
                best_seq = scored.get("best_sequence") or (ref_seq_now or [0])
                mode = "efe" if best_seq else "fallback"
            else:
                best_seq = ref_seq_now or [0]
                mode = "fallback"

        elif cfg.controller == "pure_efe":
            with torch.no_grad():
                scored = efe_planner.score_action_sequences_pure(
                    belief=belief, recent_true_states=recent_true_states,
                )
            best_seq = scored.get("best_sequence") or []
            mode = "efe" if best_seq else "empty_fallback"
            if not best_seq:
                best_seq = [0]

        else:
            raise ValueError(f"Unknown controller: {cfg.controller!r}")

        if not best_seq:
            best_seq = [0]
            mode = "empty_fallback"

        action = int(best_seq[0])

        if mode == "graph_ref":     graph_ref_count      += 1
        elif mode == "planner":     planner_count        += 1
        elif mode == "efe":         efe_count            += 1
        elif mode == "fallback":    fallback_count       += 1
        else:                       empty_fallback_count += 1

        obs, _, done, info = step_env(env, action)
        true_pose = get_true_pose(info)

        obs_history.append(np.asarray(obs, dtype=np.float32))
        action_history.append(action)
        recent_true_states.append(true_pose)

        # Trim histories
        if len(obs_history) > cfg.history_keep:
            obs_history    = obs_history[-cfg.history_keep:]
        if len(action_history) > cfg.history_keep - 1:
            action_history = action_history[-(cfg.history_keep - 1):]
        if len(recent_true_states) > cfg.recent_true_states_keep:
            recent_true_states = recent_true_states[-cfg.recent_true_states_keep:]

        planning_steps += 1
        if info["collision"]:
            total_collisions += 1
        if done or info["reached_goal"]:
            reached_goal = True
            break

    return _make_result(
        trial_id, cfg, start_pose, goal_pos, recent_true_states,
        reached_goal=reached_goal, planning_steps=planning_steps,
        total_collisions=total_collisions, shortest_dist=shortest_dist,
        belief_errors=belief_errors,
        entropy_values=entropy_values,
        context_std_values=context_std_values,
        vfe_kl_values=vfe_kl_values,
        graph_ref_count=graph_ref_count, planner_count=planner_count,
        efe_count=efe_count, fallback_count=fallback_count,
        empty_fallback_count=empty_fallback_count,
    )


def _make_result(
    trial_id: int,
    cfg: EvalConfig,
    start_pose: Tuple[int, int, str],
    goal_pos: Tuple[int, int],
    recent_true_states: List[Tuple[int, int, str]],
    reached_goal: bool,
    planning_steps: int,
    total_collisions: int,
    shortest_dist: Optional[int],
    belief_errors: int,
    entropy_values: List[float],
    context_std_values: List[float],
    vfe_kl_values: List[float],
    graph_ref_count: int,
    planner_count: int,
    efe_count: int,
    fallback_count: int,
    empty_fallback_count: int,
) -> Dict[str, Any]:
    path_eff = (
        float(shortest_dist) / float(planning_steps)
        if (reached_goal and shortest_dist and planning_steps > 0)
        else 0.0
    )
    final = recent_true_states[-1]
    return {
        "trial":                trial_id,
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
        # V4-specific
        "mean_context_std":     float(np.mean(context_std_values)) if context_std_values else 0.0,
        "mean_vfe_kl":          float(np.mean(vfe_kl_values)) if vfe_kl_values else 0.0,
        # Mode counts
        "graph_ref_count":      graph_ref_count,
        "planner_count":        planner_count,
        "efe_count":            efe_count,
        "fallback_count":       fallback_count,
        "empty_fallback_count": empty_fallback_count,
    }


# ============================================================
# Summary / saving
# ============================================================

def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n         = len(results)
    successes = [r for r in results if r["success"] == 1]
    failures  = [r for r in results if r["success"] == 0]

    def mean(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "num_trials":              n,
        "success_rate":            len(successes) / max(n, 1),
        "timeout_rate":            len(failures)  / max(n, 1),
        "avg_steps_successful":    mean([r["steps"] for r in successes]),
        "avg_steps_all":           mean([r["steps"] for r in results]),
        "avg_collisions":          mean([r["collisions"] for r in results]),
        "collision_rate":          mean([float(r["collisions"] > 0) for r in results]),
        "mean_path_efficiency":    mean([r["path_efficiency"] for r in successes]),
        "mean_belief_error_rate":  mean([r["belief_error_rate"] for r in results]),
        "mean_entropy":            mean([r["mean_entropy"] for r in results]),
        # V4
        "mean_context_std":        mean([r["mean_context_std"] for r in results]),
        "mean_vfe_kl":             mean([r["mean_vfe_kl"] for r in results]),
        # Mode usage
        "total_graph_ref":         sum(r["graph_ref_count"]      for r in results),
        "total_planner":           sum(r["planner_count"]        for r in results),
        "total_efe":               sum(r["efe_count"]            for r in results),
        "total_fallback":          sum(r["fallback_count"]       for r in results),
        "total_empty_fallback":    sum(r["empty_fallback_count"] for r in results),
    }

    print()
    print("=" * 80)
    print("AIF V4 BATCH EVALUATION SUMMARY")
    print("=" * 80)
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
    print(f"  Mean context std (V4)   : {summary['mean_context_std']:.4f}")
    print(f"  Mean VFE KL (V4)        : {summary['mean_vfe_kl']:.6f}")
    print()
    total_dec = max(
        summary["total_graph_ref"] + summary["total_planner"] +
        summary["total_efe"] + summary["total_fallback"] + summary["total_empty_fallback"], 1
    )
    print(f"  Mode usage (graph_ref / planner / efe / fallback / empty_fb):")
    print(
        f"    {summary['total_graph_ref']} / {summary['total_planner']} / "
        f"{summary['total_efe']} / {summary['total_fallback']} / "
        f"{summary['total_empty_fallback']}  "
        f"(fracs: {summary['total_graph_ref']/total_dec:.2f} / "
        f"{summary['total_planner']/total_dec:.2f} / "
        f"{summary['total_efe']/total_dec:.2f} / "
        f"{summary['total_fallback']/total_dec:.2f} / "
        f"{summary['total_empty_fallback']/total_dec:.2f})"
    )

    if failures:
        print("\n  First 10 failures:")
        for r in failures[:10]:
            print(
                f"    trial={r['trial']:03d} | "
                f"start=({r['start_row']},{r['start_col']},{r['start_heading']}) | "
                f"goal=({r['goal_row']},{r['goal_col']}) | "
                f"steps={r['steps']} | coll={r['collisions']} | "
                f"belief_err={r['belief_error_rate']:.2f} | "
                f"ctx_std={r['mean_context_std']:.3f}"
            )

    return summary


def save_csv(results: List[Dict[str, Any]], path: str) -> None:
    if not results:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved per-trial CSV : {path}")


def save_summary_json(summary: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON  : {path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg    = EvalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device  : {device}")
    print(f"Controller    : {cfg.controller}")
    print(f"Num trials    : {cfg.num_trials}")
    print(f"Max steps     : {cfg.max_steps}")

    model = build_model_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        device=device,
        model_config_json=cfg.model_config_json,
    )

    template_env = TinyIndoorEnv(seed=cfg.seed)
    pairs        = sample_start_goal_pairs(template_env, cfg.num_trials, cfg.seed)

    results: List[Dict[str, Any]] = []
    t_start = time.time()

    for i, (start_pose, goal_pos) in enumerate(pairs):
        result = run_single_trial(
            trial_id=i, cfg=cfg, model=model, device=device,
            start_pose=start_pose, goal_pos=goal_pos,
        )
        results.append(result)

        print(
            f"[{i+1:03d}/{cfg.num_trials:03d}] "
            f"{'OK' if result['success'] else 'FAIL'} | "
            f"steps={result['steps']:02d} | "
            f"coll={result['collisions']} | "
            f"g:{result['graph_ref_count']} "
            f"e:{result['efe_count']} "
            f"f:{result['fallback_count']} | "
            f"ent={result['mean_entropy']:.3f} | "
            f"ctx_std={result['mean_context_std']:.3f} | "
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

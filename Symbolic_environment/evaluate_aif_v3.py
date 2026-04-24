# evaluate_aif_v3.py

from __future__ import annotations

import json
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from model_v3 import WorldModelV3, ModelV3Config
from simulator import TinyIndoorEnv, Pose, StepResult
from aif_planner_v3 import (
    AIFPlannerV3,
    AIFPlannerConfig,
    IDX_TO_HEADING,
    most_likely_state_from_belief,
    build_reference_action_sequence,
    masked_joint_from_belief,
)
from aif_efe_planner_v1 import EFEPlannerV1, EFEPlannerConfig


# ============================================================
# Config
# ============================================================

@dataclass
class EvalConfig:
    checkpoint_path: str = "./checkpoints_v3/best_model.pt"
    model_config_json: str = "./checkpoints_v3/model_config.json"

    num_trials: int = 100
    max_steps: int = 80
    seed: int = 42

    history_keep: int = 64
    recent_true_states_keep: int = 20
    warmup_actions: Tuple[int, ...] = (2, 3, 3, 2)

    confidence_entropy_threshold: float = 0.22
    fallback_entropy_threshold: float = 0.8

    # Options: "hybrid", "planner", "graph_ref", "hybrid_efe"
    controller: str = "hybrid_efe"
    output_csv: str = "./aif_v3_eval_results_hybrid_efe.csv"

    efe_cfg: EFEPlannerConfig = field(
        default_factory=lambda: EFEPlannerConfig(
            horizon=5,
            max_candidates=128,
            allow_backward=True,
            w_risk=4.0,
            w_terminal_risk=10.0,
            w_min_risk=4.0,
            w_ambiguity=0.10,
            w_collision=18.0,
            w_info_gain=0.25,
            preference_precision=1.00,
            discount=0.90,
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

def load_model_config_from_json(path: str) -> ModelV3Config:
    with open(path, "r") as f:
        return ModelV3Config(**json.load(f))


def build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_config_json: Optional[str] = None,
) -> WorldModelV3:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_config" in ckpt:
        model_cfg = ModelV3Config(**ckpt["model_config"])
    else:
        if model_config_json is None or not Path(model_config_json).exists():
            raise ValueError("Checkpoint has no model_config and model_config_json was not found.")
        print(f"Loaded model_config from JSON: {model_config_json}")
        model_cfg = load_model_config_from_json(model_config_json)

    model = WorldModelV3(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


# ============================================================
# Environment / map helpers
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


def build_reachable_mask_from_env(
    env: TinyIndoorEnv,
    num_row_classes: int,
    num_col_classes: int,
) -> torch.Tensor:
    mask = torch.zeros(num_row_classes, num_col_classes, dtype=torch.float32)

    for r in range(num_row_classes):
        for c in range(num_col_classes):
            if 0 <= r < env.rows and 0 <= c < env.cols:
                mask[r, c] = 0.0 if env._is_blocking(r, c) else 1.0
            else:
                mask[r, c] = 0.0

    return mask


def shortest_path_distances(
    env: TinyIndoorEnv,
    goal_rc: Tuple[int, int],
    num_row_classes: int,
    num_col_classes: int,
    unreachable_cost: float = 50.0,
) -> torch.Tensor:
    out = torch.full((num_row_classes, num_col_classes), unreachable_cost, dtype=torch.float32)

    for r in range(num_row_classes):
        for c in range(num_col_classes):
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
    env: TinyIndoorEnv,
    num_trials: int,
    seed: int,
) -> List[Tuple[Tuple[int, int, str], Tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    free_cells = get_free_cells(env)
    headings = ["N", "E", "S", "W"]

    pairs = []
    tries = 0

    while len(pairs) < num_trials and tries < num_trials * 100:
        tries += 1

        start_rc = free_cells[int(rng.integers(0, len(free_cells)))]
        goal_rc = free_cells[int(rng.integers(0, len(free_cells)))]

        if start_rc == goal_rc:
            continue

        d = env.shortest_path_length(start_rc, goal_rc)
        if d is None or d < 3:
            continue

        heading = headings[int(rng.integers(0, len(headings)))]
        pairs.append(((start_rc[0], start_rc[1], heading), goal_rc))

    if len(pairs) < num_trials:
        raise RuntimeError(f"Could only sample {len(pairs)} valid pairs.")

    return pairs


# ============================================================
# Tensor / belief helpers
# ============================================================

def build_observation_tensor(obs_history: List[np.ndarray], device: torch.device) -> torch.Tensor:
    obs_np = np.stack(obs_history, axis=0).astype(np.float32)
    return torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(2).to(device)


def build_action_tensor(action_history: List[int], device: torch.device) -> Optional[torch.Tensor]:
    if len(action_history) == 0:
        return None
    act_np = np.array(action_history, dtype=np.int64)
    return torch.from_numpy(act_np).unsqueeze(0).to(device)


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


def greedy_action_from_reference(ref_seq: List[int]) -> int:
    return int(ref_seq[0]) if len(ref_seq) > 0 else 0


def safe_best_sequence_from_planner(
    planner: AIFPlannerV3,
    belief: Dict[str, torch.Tensor],
    recent_true_states: List[Tuple[int, int, str]],
    ref_seq_now: List[int],
) -> Tuple[List[int], Dict[str, Any]]:
    with torch.no_grad():
        scored = planner.score_action_sequences(
            belief=belief,
            recent_true_states=recent_true_states,
        )

    best_seq = scored.get("best_sequence", [])
    if len(best_seq) == 0:
        best_seq = [greedy_action_from_reference(ref_seq_now)]

    return best_seq, scored


def safe_best_sequence_from_efe(
    efe_planner: EFEPlannerV1,
    belief: Dict[str, torch.Tensor],
    recent_true_states: List[Tuple[int, int, str]],
    ref_seq_now: List[int],
) -> Tuple[List[int], Dict[str, Any]]:
    with torch.no_grad():
        scored = efe_planner.score_action_sequences(
            belief=belief,
            recent_true_states=recent_true_states,
        )

    best_seq = scored.get("best_sequence", [])
    if len(best_seq) == 0:
        best_seq = [greedy_action_from_reference(ref_seq_now)]

    return best_seq, scored


# ============================================================
# Single trial
# ============================================================

def run_single_trial(
    trial_id: int,
    cfg: EvalConfig,
    model: WorldModelV3,
    device: torch.device,
    start_pose: Tuple[int, int, str],
    goal_pos: Tuple[int, int],
) -> Dict[str, Any]:

    env = TinyIndoorEnv(seed=cfg.seed + trial_id)

    reachable_mask = build_reachable_mask_from_env(
        env,
        model.cfg.num_row_classes,
        model.cfg.num_col_classes,
    ).to(device)

    dist_t = shortest_path_distances(
        env,
        goal_pos,
        model.cfg.num_row_classes,
        model.cfg.num_col_classes,
        unreachable_cost=50.0,
    ).to(device)

    shortest_dist = env.shortest_path_length((start_pose[0], start_pose[1]), goal_pos)

    planner = AIFPlannerV3(
        model=model,
        dist_t=dist_t,
        reachable_mask=reachable_mask,
        cfg=cfg.planner_cfg,
    )

    efe_planner = EFEPlannerV1(
        model=model,
        dist_t=dist_t,
        reachable_mask=reachable_mask,
        cfg=cfg.efe_cfg,
    )

    obs, info = reset_env(env, start_pose, goal_pos)
    true_pose = get_true_pose(info)

    obs_history: List[np.ndarray] = [np.asarray(obs, dtype=np.float32)]
    action_history: List[int] = []
    recent_true_states: List[Tuple[int, int, str]] = [true_pose]

    total_collisions = 0
    reached_goal = False
    used_modes: List[str] = []

    belief_errors = 0
    entropy_values: List[float] = []

    graph_ref_count = 0
    planner_count = 0
    efe_count = 0
    fallback_count = 0
    empty_fallback_count = 0

    # Warm-up
    for a in cfg.warmup_actions:
        obs, reward, done, info = step_env(env, a)
        true_pose = get_true_pose(info)

        obs_history.append(np.asarray(obs, dtype=np.float32))
        action_history.append(a)
        recent_true_states.append(true_pose)

        if bool(info["collision"]):
            total_collisions += 1

        if done or bool(info["reached_goal"]):
            reached_goal = True
            break

    if reached_goal:
        return {
            "trial": trial_id,
            "controller": cfg.controller,
            "success": 1,
            "timeout": 0,
            "steps": 0,
            "collisions": total_collisions,
            "start_row": start_pose[0],
            "start_col": start_pose[1],
            "start_heading": start_pose[2],
            "goal_row": goal_pos[0],
            "goal_col": goal_pos[1],
            "final_row": recent_true_states[-1][0],
            "final_col": recent_true_states[-1][1],
            "final_heading": recent_true_states[-1][2],
            "shortest_dist": shortest_dist if shortest_dist is not None else -1,
            "path_efficiency": 1.0,
            "belief_error_rate": 0.0,
            "mean_entropy": 0.0,
            "graph_ref_count": 0,
            "planner_count": 0,
            "efe_count": 0,
            "fallback_count": 0,
            "empty_fallback_count": 0,
            "modes": "warmup_success",
        }

    planning_steps = 0

    for _ in range(cfg.max_steps):
        obs_t = build_observation_tensor(obs_history, device)
        act_t = build_action_tensor(action_history, device)

        # Cheap belief inference only
        with torch.no_grad():
            belief = planner.infer_current_belief(
                observations=obs_t,
                actions=act_t,
            )

        belief_ent = belief_entropy_masked(
            belief["row_probs"][0],
            belief["col_probs"][0],
            reachable_mask,
        )
        entropy_values.append(belief_ent)

        pred_r, pred_c, pred_h = most_likely_state_from_belief(
            belief["row_probs"][0],
            belief["col_probs"][0],
            belief["heading_probs"][0],
            reachable_mask,
        )
        pred_pose = (pred_r, pred_c, IDX_TO_HEADING[pred_h])

        if pred_pose != recent_true_states[-1]:
            belief_errors += 1

        ref_seq_now = build_reference_action_sequence(
            r=pred_r,
            c=pred_c,
            h=pred_h,
            dist_t=dist_t,
            reachable_mask=reachable_mask,
            horizon=cfg.planner_cfg.horizon,
        )

        # Controller selection
        if cfg.controller == "graph_ref":
            best_seq = ref_seq_now
            mode = "graph_ref"

        elif cfg.controller == "planner":
            best_seq, _ = safe_best_sequence_from_planner(
                planner=planner,
                belief=belief,
                recent_true_states=recent_true_states,
                ref_seq_now=ref_seq_now,
            )
            mode = "planner"

        elif cfg.controller == "hybrid":
            # confident_pose = belief_ent < cfg.confidence_entropy_threshold
            # use_fallback = belief_ent > cfg.fallback_entropy_threshold

            if confident_pose and len(ref_seq_now) > 0:
                best_seq = ref_seq_now
                mode = "graph_ref"
            elif use_fallback:
                best_seq = [greedy_action_from_reference(ref_seq_now)]
                mode = "fallback"
            else:
                best_seq, _ = safe_best_sequence_from_planner(
                    planner=planner,
                    belief=belief,
                    recent_true_states=recent_true_states,
                    ref_seq_now=ref_seq_now,
                )
                mode = "planner"

        elif cfg.controller == "hybrid_efe":

            # NEW: direct entropy-based gating (no booleans)
            if belief_ent < 0.22 and len(ref_seq_now) > 0: #cfg.confidence_entropy_threshold = 0.22
                # Very confident → use graph shortest path
                best_seq = ref_seq_now
                mode = "graph_ref"

            elif belief_ent < 0.8: #cfg.fallback_entropy_threshold = 0.8
                # Medium uncertainty → use EFE planner
                scored = efe_planner.score_action_sequences(
                    belief=belief,
                    recent_true_states=recent_true_states,
                )
                best_seq = scored["best_sequence"]

                if len(best_seq) == 0:
                    best_seq = [greedy_action_from_reference(ref_seq_now)]
                    mode = "fallback"
                else:
                    mode = "efe"

            else:
                # Very uncertain → safe fallback
                best_seq = [greedy_action_from_reference(ref_seq_now)]
                mode = "fallback"

        else:
            raise ValueError(f"Unknown controller: {cfg.controller}")

        if len(best_seq) == 0:
            best_seq = [0]
            mode = "empty_fallback"

        action = int(best_seq[0])
        used_modes.append(mode)

        if mode == "graph_ref":
            graph_ref_count += 1
        elif mode == "planner":
            planner_count += 1
        elif mode == "efe":
            efe_count += 1
        elif mode == "fallback":
            fallback_count += 1
        elif mode == "empty_fallback":
            empty_fallback_count += 1

        obs, reward, done, info = step_env(env, action)
        true_pose = get_true_pose(info)

        obs_history.append(np.asarray(obs, dtype=np.float32))
        action_history.append(action)
        recent_true_states.append(true_pose)

        if len(obs_history) > cfg.history_keep:
            obs_history = obs_history[-cfg.history_keep:]
        if len(action_history) > cfg.history_keep - 1:
            action_history = action_history[-(cfg.history_keep - 1):]
        if len(recent_true_states) > cfg.recent_true_states_keep:
            recent_true_states = recent_true_states[-cfg.recent_true_states_keep:]

        planning_steps += 1

        if bool(info["collision"]):
            total_collisions += 1

        if done or bool(info["reached_goal"]):
            reached_goal = True
            break

    timeout = int(not reached_goal)

    if reached_goal and shortest_dist is not None and planning_steps > 0:
        path_eff = float(shortest_dist) / float(planning_steps)
    else:
        path_eff = 0.0

    return {
        "trial": trial_id,
        "controller": cfg.controller,
        "success": int(reached_goal),
        "timeout": timeout,
        "steps": planning_steps,
        "collisions": total_collisions,
        "start_row": start_pose[0],
        "start_col": start_pose[1],
        "start_heading": start_pose[2],
        "goal_row": goal_pos[0],
        "goal_col": goal_pos[1],
        "final_row": recent_true_states[-1][0],
        "final_col": recent_true_states[-1][1],
        "final_heading": recent_true_states[-1][2],
        "shortest_dist": shortest_dist if shortest_dist is not None else -1,
        "path_efficiency": path_eff,
        "belief_error_rate": belief_errors / max(planning_steps, 1),
        "mean_entropy": float(np.mean(entropy_values)) if entropy_values else 0.0,
        "graph_ref_count": graph_ref_count,
        "planner_count": planner_count,
        "efe_count": efe_count,
        "fallback_count": fallback_count,
        "empty_fallback_count": empty_fallback_count,
        "modes": "|".join(used_modes),
    }


# ============================================================
# Summary / saving
# ============================================================

def summarize_results(results: List[Dict[str, Any]]) -> None:
    n = len(results)
    successes = [r for r in results if r["success"] == 1]
    failures = [r for r in results if r["success"] == 0]

    success_rate = len(successes) / max(n, 1)
    timeout_rate = len(failures) / max(n, 1)

    avg_steps_success = np.mean([r["steps"] for r in successes]) if successes else float("nan")
    avg_steps_all = np.mean([r["steps"] for r in results]) if results else float("nan")
    avg_collisions = np.mean([r["collisions"] for r in results]) if results else float("nan")
    collision_rate = np.mean([1.0 if r["collisions"] > 0 else 0.0 for r in results]) if results else float("nan")
    avg_eff = np.mean([r["path_efficiency"] for r in successes]) if successes else float("nan")
    avg_belief_error = np.mean([r["belief_error_rate"] for r in results]) if results else float("nan")
    avg_entropy = np.mean([r["mean_entropy"] for r in results]) if results else float("nan")

    total_graph_ref = sum(r["graph_ref_count"] for r in results)
    total_planner = sum(r["planner_count"] for r in results)
    total_efe = sum(r["efe_count"] for r in results)
    total_fallback = sum(r["fallback_count"] for r in results)
    total_empty = sum(r["empty_fallback_count"] for r in results)

    total_decisions = total_graph_ref + total_planner + total_efe + total_fallback + total_empty

    def frac(x: int) -> float:
        return x / max(total_decisions, 1)

    print()
    print("=" * 100)
    print("AIF V3 BATCH EVALUATION SUMMARY")
    print("=" * 100)
    print(f"Trials:                         {n}")
    print(f"Success rate:                   {success_rate:.4f}")
    print(f"Timeout/failure rate:           {timeout_rate:.4f}")
    print(f"Average steps, successful only: {avg_steps_success:.2f}")
    print(f"Average steps, all trials:      {avg_steps_all:.2f}")
    print(f"Average collisions per trial:   {avg_collisions:.4f}")
    print(f"Collision rate:                 {collision_rate:.4f}")
    print(f"Mean path efficiency:           {avg_eff:.4f}")
    print(f"Mean belief error rate:         {avg_belief_error:.4f}")
    print(f"Mean belief entropy:            {avg_entropy:.4f}")

    print()
    print("Mode usage:")
    print(f"Total graph_ref decisions:      {total_graph_ref} ({frac(total_graph_ref):.4f})")
    print(f"Total planner decisions:        {total_planner} ({frac(total_planner):.4f})")
    print(f"Total EFE decisions:            {total_efe} ({frac(total_efe):.4f})")
    print(f"Total fallback decisions:       {total_fallback} ({frac(total_fallback):.4f})")
    print(f"Total empty fallback decisions: {total_empty} ({frac(total_empty):.4f})")
    print(f"Avg graph_ref per trial:        {np.mean([r['graph_ref_count'] for r in results]):.2f}")
    print(f"Avg planner per trial:          {np.mean([r['planner_count'] for r in results]):.2f}")
    print(f"Avg EFE per trial:              {np.mean([r['efe_count'] for r in results]):.2f}")
    print(f"Avg fallback per trial:         {np.mean([r['fallback_count'] for r in results]):.2f}")

    if failures:
        print()
        print("Failure cases:")
        for r in failures[:10]:
            print(
                f"  trial={r['trial']} | "
                f"start=({r['start_row']},{r['start_col']},{r['start_heading']}) | "
                f"goal=({r['goal_row']},{r['goal_col']}) | "
                f"final=({r['final_row']},{r['final_col']},{r['final_heading']}) | "
                f"steps={r['steps']} | collisions={r['collisions']} | "
                f"belief_err={r['belief_error_rate']:.3f} | "
                f"mean_ent={r['mean_entropy']:.3f}"
            )


def save_csv(results: List[Dict[str, Any]], path: str) -> None:
    if not results:
        return

    keys = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved per-trial results to: {path}")


# ============================================================
# Main
# ============================================================

def main():
    cfg = EvalConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Controller: {cfg.controller}")
    print(f"Num trials: {cfg.num_trials}")
    print(f"Max steps:  {cfg.max_steps}")

    model = build_model_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        device=device,
        model_config_json=cfg.model_config_json,
    )

    template_env = TinyIndoorEnv(seed=cfg.seed)
    pairs = sample_start_goal_pairs(template_env, cfg.num_trials, cfg.seed)

    results: List[Dict[str, Any]] = []
    t_start = time.time()

    for i, (start_pose, goal_pos) in enumerate(pairs):
        result = run_single_trial(
            trial_id=i,
            cfg=cfg,
            model=model,
            device=device,
            start_pose=start_pose,
            goal_pos=goal_pos,
        )
        results.append(result)

        print(
            f"[{i + 1:03d}/{cfg.num_trials:03d}] "
            f"success={result['success']} | "
            f"steps={result['steps']:02d} | "
            f"collisions={result['collisions']} | "
            f"modes=g:{result['graph_ref_count']} "
            f"p:{result['planner_count']} "
            f"e:{result['efe_count']} "
            f"f:{result['fallback_count']} | "
            f"belief_err={result['belief_error_rate']:.2f} | "
            f"start=({result['start_row']},{result['start_col']},{result['start_heading']}) | "
            f"goal=({result['goal_row']},{result['goal_col']}) | "
            f"final=({result['final_row']},{result['final_col']},{result['final_heading']})"
        )

    elapsed = time.time() - t_start

    summarize_results(results)
    save_csv(results, cfg.output_csv)

    print(f"Total wall-clock time: {elapsed:.2f} sec")


if __name__ == "__main__":
    main()
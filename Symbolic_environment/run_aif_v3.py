# run_aif_v3.py

from __future__ import annotations

import json
from dataclasses import dataclass
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



ACTION_NAMES = {
    0: "forward",
    1: "backward",
    2: "turn_left",
    3: "turn_right",
}



@dataclass
class RunnerConfig:
    checkpoint_path: str = "./checkpoints_v32/best_model_v32.pt"
    model_config_json: str = "./checkpoints_v32/model_config.json"

    start_pose: Tuple[int, int, str] = (8,4, "N")
    goal_pos: Tuple[int, int] = (6,5)

    max_steps: int = 80
    warmup_actions: Tuple[int, ...] = (2, 3, 3, 2)
    history_keep: int = 64
    recent_true_states_keep: int = 20

    # mode switching
    confidence_entropy_threshold: float = 0.35
    fallback_entropy_threshold: float = 1.10

    topk_to_print: int = 5

    # planner config
    planner_cfg: AIFPlannerConfig = AIFPlannerConfig(
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


# ============================================================
# Model loading
# ============================================================

def load_model_config_from_json(path: str) -> ModelV3Config:
    with open(path, "r") as f:
        d = json.load(f)
    return ModelV3Config(**d)


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
            raise ValueError(
                "Checkpoint does not contain 'model_config' and model_config_json was not found."
            )
        print(f"Loaded model_config from JSON: {model_config_json}")
        model_cfg = load_model_config_from_json(model_config_json)

    model = WorldModelV3(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
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
    obs, info = env.reset(
        start_pose=pose,
        goal_pos=goal_pos,
        use_goal=True,
    )
    return obs, info


def step_env(env: TinyIndoorEnv, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    result: StepResult = env.step(action)
    return result.obs, float(result.reward), bool(result.done), result.info


def get_true_pose(info: Dict[str, Any]) -> Tuple[int, int, str]:
    return int(info["row"]), int(info["col"]), str(info["heading"])


def render_ascii(env: TinyIndoorEnv) -> str:
    return env.render_topdown_ascii(show_goal=True)


# ============================================================
# Grid helpers
# ============================================================

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


def belief_entropy_masked(
    row_probs: torch.Tensor,
    col_probs: torch.Tensor,
    reachable_mask: torch.Tensor,
) -> float:
    joint, valid_mass = masked_joint_from_belief(row_probs, col_probs, reachable_mask)
    if valid_mass <= 1e-8:
        return 10.0
    eps = 1e-8
    ent = -(joint * joint.clamp_min(eps).log()).sum()
    return float(ent.item())


# ============================================================
# Observation history helpers
# ============================================================

def build_observation_tensor(obs_history: List[np.ndarray], device: torch.device) -> torch.Tensor:
    obs_np = np.stack(obs_history, axis=0).astype(np.float32)       # [T,H,W]
    obs_t = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(2).to(device)  # [1,T,1,H,W]
    return obs_t


def build_action_tensor(action_history: List[int], device: torch.device) -> Optional[torch.Tensor]:
    if len(action_history) == 0:
        return None
    act_np = np.array(action_history, dtype=np.int64)
    act_t = torch.from_numpy(act_np).unsqueeze(0).to(device)        # [1,T-1]
    return act_t


def greedy_action_from_reference(
    ref_seq: List[int],
) -> int:
    if len(ref_seq) == 0:
        return 0
    return int(ref_seq[0])


# ============================================================
# Main
# ============================================================

def main():
    cfg = RunnerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        device=device,
        model_config_json=cfg.model_config_json,
    )

    env = TinyIndoorEnv(seed=42)

    obs, info = reset_env(env, cfg.start_pose, cfg.goal_pos)
    true_pose = get_true_pose(info)

    print(f"Start pose: {cfg.start_pose}")
    print(f"Goal pos:  {cfg.goal_pos}")
    print()

    reachable_mask = build_reachable_mask_from_env(
        env=env,
        num_row_classes=model.cfg.num_row_classes,
        num_col_classes=model.cfg.num_col_classes,
    )
    dist_t = shortest_path_distances(
        env=env,
        goal_rc=cfg.goal_pos,
        num_row_classes=model.cfg.num_row_classes,
        num_col_classes=model.cfg.num_col_classes,
        unreachable_cost=50.0,
    )

    planner = AIFPlannerV3(
        model=model,
        dist_t=dist_t.to(device),
        reachable_mask=reachable_mask.to(device),
        cfg=cfg.planner_cfg,
    )

    obs_history: List[np.ndarray] = []
    action_history: List[int] = []
    recent_true_states: List[Tuple[int, int, str]] = []

    obs_history.append(np.asarray(obs, dtype=np.float32))
    recent_true_states.append(true_pose)

    print("=" * 110)
    print("WARM-UP PHASE")
    print("=" * 110)

    for i, a in enumerate(cfg.warmup_actions):
        obs, reward, done, info = step_env(env, a)
        true_pose = get_true_pose(info)

        obs_history.append(np.asarray(obs, dtype=np.float32))
        action_history.append(a)
        recent_true_states.append(true_pose)

        print(f"Warm-up step {i}: action={a} ({ACTION_NAMES[a]})")
        print(render_ascii(env))
        print(
            f"Collision: {bool(info['collision'])} | "
            f"Reached goal: {bool(info['reached_goal'])}"
        )
        print()

        if done or info["reached_goal"]:
            print("Goal reached during warm-up.")
            return

    total_collisions = 0
    reached_goal = False

    for step_idx in range(cfg.max_steps):
        obs_t = build_observation_tensor(obs_history, device)
        act_t = build_action_tensor(action_history, device)

        plan_out = planner.plan(
            observations=obs_t,
            actions=act_t,
            recent_true_states=recent_true_states,
        )

        belief = plan_out["belief"]

        belief_ent = belief_entropy_masked(
            belief["row_probs"][0],
            belief["col_probs"][0],
            reachable_mask.to(device),
        )

        pred_r, pred_c, pred_h = most_likely_state_from_belief(
            belief["row_probs"][0],
            belief["col_probs"][0],
            belief["heading_probs"][0],
            reachable_mask.to(device),
        )
        pred_pose = (pred_r, pred_c, IDX_TO_HEADING[pred_h])

        ref_seq_now = build_reference_action_sequence(
            r=pred_r,
            c=pred_c,
            h=pred_h,
            dist_t=dist_t.to(device),
            reachable_mask=reachable_mask.to(device),
            horizon=cfg.planner_cfg.horizon,
        )

        confident_pose = belief_ent < cfg.confidence_entropy_threshold
        use_fallback = belief_ent > cfg.fallback_entropy_threshold

        if confident_pose and len(ref_seq_now) > 0:
            best_seq = ref_seq_now
            scored = []
            mode = "graph_ref"
        elif use_fallback:
            best_seq = [greedy_action_from_reference(ref_seq_now)]
            scored = []
            mode = "fallback"
        else:
            best_seq = plan_out["best_sequence"]
            scored = plan_out["all_details"]
            if len(best_seq) == 0:
                best_seq = [greedy_action_from_reference(ref_seq_now)]
                scored = []
                mode = "fallback"
            else:
                mode = "planner"

        action = int(best_seq[0])

        print("=" * 110)
        print(f"Step {step_idx}")
        # print(f"Mode                   : {mode}")
        print(f"Current TRUE pose      : {recent_true_states[-1]}")
        print(f"Current FILTERED pose  : {pred_pose}")
        # print(f"Belief entropy         : {belief_ent:.4f}")
        print(f"Goal                   : {cfg.goal_pos}")
        print(f"Reference sequence     : {ref_seq_now}")
        print(f"Chosen sequence        : {best_seq}")
        print(f"Chosen first action    : {action} ({ACTION_NAMES[action]})")

        if len(scored) > 0:
            print(f"Best score             : {scored[0]['score']:.4f}")
            print("Top candidates:")
            for d in scored[:cfg.topk_to_print]:
                print(
                    f"  seq={d['sequence']} | score={d['score']:.4f} | "
                    f"goal={d['goal']:.4f} | term={d['term']:.4f} | "
                    f"min_goal={d['min_goal']:.4f} | prog={d['prog']:.4f} | "
                    f"reg={d['reg']:.4f} | coll={d['coll']:.4f} | "
                    f"ent={d['ent']:.4f} | act={d['act']:.4f} | "
                    f"rev={d['rev']:.4f} | hdg={d['hdg']:.4f} | "
                    f"loop={d['loop']:.4f} | ref={d['ref']:.4f} | "
                    f"final_cost={d['final_cost']:.4f} | "
                    f"min_cost_seen={d['min_cost_seen']:.4f} | "
                    f"ref_seq={d['ref_seq']}"
                )
        else:
            print("Best score             : nan")

        print()
        print("ASCII before action:")
        print(render_ascii(env))
        print()

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

        collision = bool(info["collision"])
        reached_goal = bool(info["reached_goal"])

        if collision:
            total_collisions += 1

        print("ASCII after action:")
        print(render_ascii(env))
        print(f"Collision: {collision} | Reached goal: {reached_goal}")
        print()

        if done or reached_goal:
            print("=" * 110)
            print(f"Goal reached in {step_idx + 1} planning steps.")
            print()
            break

    print("=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)
    executed_steps = len(action_history) - len(cfg.warmup_actions)
    print(f"Total executed planning steps: {max(executed_steps, 0)}")
    print(f"Total collisions:             {total_collisions}")
    print(f"Final pose:                   {recent_true_states[-1]}")
    print(f"Goal:                         {cfg.goal_pos}")
    print(f"Reached goal:                 {reached_goal}")


if __name__ == "__main__":
    main()
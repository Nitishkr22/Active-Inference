# evaluate_latent_efe_50_trials.py

from __future__ import annotations

import json
import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from model_v3 import WorldModelV3, ModelV3Config
from simulator import TinyIndoorEnv, Pose, StepResult
from latent_efe_planner_v1 import (
    LatentEFEPlannerV1,
    LatentEFEConfig,
    OnlineVFEConfig,
    ACTION_NAMES,
)


HEADING_TO_IDX = {"N": 0, "E": 1, "S": 2, "W": 3}
IDX_TO_HEADING = {v: k for k, v in HEADING_TO_IDX.items()}

HEADING_DELTAS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


@dataclass
class EvalConfig:
    checkpoint_path: str = "./checkpoints_v3_predictive_latent/best_predictive_latent.pt"
    model_config_json: str = "./checkpoints_v32/model_config.json"

    num_trials: int = 50
    seed: int = 123

    max_steps_per_trial: int = 80
    history_keep: int = 64

    # Set fixed_goal=None for random goals
    fixed_goal: Optional[Tuple[int, int]] = None
    min_start_goal_dist: int = 4

    output_csv: str = "./latent_efe_50_trial_results.csv"

    latent_cfg: LatentEFEConfig = field(
        default_factory=lambda: LatentEFEConfig(
            horizon=5,
            allow_backward=False,
            max_candidates=128,

            w_latent_risk=12.0,
            w_terminal_latent_risk=25.0,

            w_graph_path=4.0,
            w_terminal_graph_path=8.0,
            w_graph_progress=40.0,

            w_collision=30.0,
            w_entropy=0.05,
            w_info_gain=1.0,
            w_wall_mass=20.0,
            w_no_progress=12.0,

            w_action=0.20,
            w_backward=1.50,
            w_turn=0.20,
            w_inverse=8.00,
            w_context_smoothness=0.25,

            discount=0.90,
        )
    )

    vfe_cfg: OnlineVFEConfig = field(
        default_factory=lambda: OnlineVFEConfig(
            enabled=False,
            num_steps=3,
            lr=0.15,
            entropy_weight=0.02,
            unreachable_weight=5.0,
        )
    )


# ============================================================
# Model helpers
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

    if "model_config" in ckpt and ckpt["model_config"] is not None:
        model_cfg = ModelV3Config(**ckpt["model_config"])
    else:
        if model_config_json is None or not Path(model_config_json).exists():
            raise ValueError("Checkpoint has no model_config and model_config_json was not found.")
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
):
    pose = Pose(row=start_pose[0], col=start_pose[1], heading=start_pose[2])
    return env.reset(start_pose=pose, goal_pos=goal_pos, use_goal=True)


def step_env(env: TinyIndoorEnv, action: int):
    result: StepResult = env.step(action)
    return result.obs, float(result.reward), bool(result.done), result.info


def get_true_pose(info: Dict[str, Any]) -> Tuple[int, int, str]:
    return int(info["row"]), int(info["col"]), str(info["heading"])


def valid_cell(env: TinyIndoorEnv, r: int, c: int) -> bool:
    return 0 <= r < env.rows and 0 <= c < env.cols and not env._is_blocking(r, c)


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
    out = torch.full(
        (num_row_classes, num_col_classes),
        unreachable_cost,
        dtype=torch.float32,
    )

    for r in range(num_row_classes):
        for c in range(num_col_classes):
            if 0 <= r < env.rows and 0 <= c < env.cols and not env._is_blocking(r, c):
                d = env.shortest_path_length((r, c), goal_rc)
                if d is not None:
                    out[r, c] = float(d)

    return out


def shortest_path_distance_map(env: TinyIndoorEnv, goal: Tuple[int, int]) -> np.ndarray:
    dist = np.full((env.rows, env.cols), np.inf, dtype=np.float32)

    gr, gc = goal
    if not valid_cell(env, gr, gc):
        return dist

    q = [(gr, gc)]
    dist[gr, gc] = 0.0
    head = 0

    while head < len(q):
        r, c = q[head]
        head += 1

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if valid_cell(env, nr, nc) and not np.isfinite(dist[nr, nc]):
                dist[nr, nc] = dist[r, c] + 1.0
                q.append((nr, nc))

    return dist


def forward_cell(r: int, c: int, h: int) -> Tuple[int, int]:
    dr, dc = HEADING_DELTAS[h]
    return r + dr, c + dc


def backward_cell(r: int, c: int, h: int) -> Tuple[int, int]:
    dr, dc = HEADING_DELTAS[h]
    return r - dr, c - dc


def reference_action(
    env: TinyIndoorEnv,
    r: int,
    c: int,
    h: int,
    dist: np.ndarray,
) -> int:
    candidates = []

    nr, nc = forward_cell(r, c, h)
    if valid_cell(env, nr, nc):
        candidates.append((float(dist[nr, nc]), 0))

    nr, nc = backward_cell(r, c, h)
    if valid_cell(env, nr, nc):
        candidates.append((float(dist[nr, nc]) + 1.2, 1))

    hl = (h - 1) % 4
    nr, nc = forward_cell(r, c, hl)
    if valid_cell(env, nr, nc):
        candidates.append((float(dist[nr, nc]) + 0.3, 2))
    else:
        candidates.append((float(dist[r, c]) + 0.8, 2))

    hr = (h + 1) % 4
    nr, nc = forward_cell(r, c, hr)
    if valid_cell(env, nr, nc):
        candidates.append((float(dist[nr, nc]) + 0.3, 3))
    else:
        candidates.append((float(dist[r, c]) + 0.8, 3))

    candidates = [(s, a) for s, a in candidates if np.isfinite(s)]

    if not candidates:
        return 0

    candidates.sort(key=lambda x: x[0])
    return int(candidates[0][1])


# ============================================================
# Tensor helpers
# ============================================================

def ensure_obs_shape(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 2:
        obs = obs[None, :, :]
    return obs


def build_observation_tensor(
    obs_history: List[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    obs_np = np.stack([ensure_obs_shape(o) for o in obs_history], axis=0).astype(np.float32)
    return torch.from_numpy(obs_np).unsqueeze(0).to(device)


def build_action_tensor(
    action_history: List[int],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if len(action_history) == 0:
        return None
    act_np = np.asarray(action_history, dtype=np.int64)
    return torch.from_numpy(act_np).unsqueeze(0).to(device)


@torch.no_grad()
def infer_belief(
    model: WorldModelV3,
    obs_history: List[np.ndarray],
    action_history: List[int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    obs_t = build_observation_tensor(obs_history, device)

    if len(action_history) == 0:
        B, T = obs_t.shape[0], obs_t.shape[1]
        if T < 2:
            obs_t = torch.cat([obs_t, obs_t], dim=1)
            act_t = torch.zeros((B, 1), dtype=torch.long, device=device)
        else:
            act_t = torch.zeros((B, T - 1), dtype=torch.long, device=device)
    else:
        act_t = build_action_tensor(action_history, device)

    out = model.forward_filter(obs_t, act_t)

    return {
        "row_probs": out["row_probs_seq"][:, -1, :],
        "col_probs": out["col_probs_seq"][:, -1, :],
        "heading_probs": out["heading_probs_seq"][:, -1, :],
        "context": out["context_seq"][:, -1, :],
    }


def most_likely_pose_from_belief(belief: Dict[str, torch.Tensor]) -> Tuple[int, int, str]:
    r = int(torch.argmax(belief["row_probs"][0]).item())
    c = int(torch.argmax(belief["col_probs"][0]).item())
    h = int(torch.argmax(belief["heading_probs"][0]).item())
    return r, c, IDX_TO_HEADING[h]


# ============================================================
# Goal context construction
# ============================================================

@torch.no_grad()
def build_goal_context_from_reference_rollout(
    model: WorldModelV3,
    start_pose: Tuple[int, int, str],
    goal_pos: Tuple[int, int],
    max_steps: int,
    device: torch.device,
    seed: int,
) -> Optional[torch.Tensor]:
    env = TinyIndoorEnv(seed=seed)
    obs, info = reset_env(env, start_pose, goal_pos)
    dist = shortest_path_distance_map(env, goal_pos)

    obs_hist: List[np.ndarray] = [np.asarray(obs, dtype=np.float32)]
    act_hist: List[int] = []

    reached = bool(info["reached_goal"])

    for _ in range(max_steps):
        r, c, h_str = get_true_pose(info)
        h = HEADING_TO_IDX[h_str]

        if (r, c) == goal_pos:
            reached = True
            break

        a = reference_action(env, r, c, h, dist)

        obs, reward, done, info = step_env(env, a)
        obs_hist.append(np.asarray(obs, dtype=np.float32))
        act_hist.append(a)

        if done or bool(info["reached_goal"]):
            reached = True
            break

    if not reached:
        return None

    belief = infer_belief(
        model=model,
        obs_history=obs_hist,
        action_history=act_hist,
        device=device,
    )

    return belief["context"].detach().clone()


# ============================================================
# Trial sampling
# ============================================================

def sample_start_goal(
    env: TinyIndoorEnv,
    rng: random.Random,
    fixed_goal: Optional[Tuple[int, int]],
    min_dist: int,
) -> Tuple[Tuple[int, int, str], Tuple[int, int]]:
    free = list(env.free_cells)

    while True:
        sr, sc = rng.choice(free)
        heading = rng.choice(["N", "E", "S", "W"])

        if fixed_goal is None:
            gr, gc = rng.choice(free)
        else:
            gr, gc = fixed_goal

        if (sr, sc) == (gr, gc):
            continue

        d = env.shortest_path_length((sr, sc), (gr, gc))
        if d is None:
            continue

        if d < min_dist:
            continue

        return (sr, sc, heading), (gr, gc)


# ============================================================
# Single trial
# ============================================================

def run_one_trial(
    trial_idx: int,
    model: WorldModelV3,
    cfg: EvalConfig,
    device: torch.device,
    rng: random.Random,
) -> Dict[str, Any]:
    env = TinyIndoorEnv(seed=cfg.seed + trial_idx)

    start_pose, goal_pos = sample_start_goal(
        env=env,
        rng=rng,
        fixed_goal=cfg.fixed_goal,
        min_dist=cfg.min_start_goal_dist,
    )

    obs, info = reset_env(env, start_pose, goal_pos)
    true_pose = get_true_pose(info)

    z_goal = build_goal_context_from_reference_rollout(
        model=model,
        start_pose=start_pose,
        goal_pos=goal_pos,
        max_steps=cfg.max_steps_per_trial,
        device=device,
        seed=cfg.seed + 10000 + trial_idx,
    )

    if z_goal is None:
        return {
            "trial": trial_idx,
            "success": False,
            "reason": "z_goal_reference_failed",
            "start_pose": start_pose,
            "goal_pos": goal_pos,
            "final_pose": true_pose,
            "steps": 0,
            "collisions": 0,
            "final_filtered_pose": None,
        }

    dist_t = shortest_path_distances(
        env=env,
        goal_rc=goal_pos,
        num_row_classes=model.cfg.num_row_classes,
        num_col_classes=model.cfg.num_col_classes,
        unreachable_cost=50.0,
    ).to(device)

    reachable_mask = build_reachable_mask_from_env(
        env=env,
        num_row_classes=model.cfg.num_row_classes,
        num_col_classes=model.cfg.num_col_classes,
    ).to(device)

    planner = LatentEFEPlannerV1(
        model=model,
        dist_t=dist_t,
        reachable_mask=reachable_mask,
        cfg=cfg.latent_cfg,
        vfe_cfg=cfg.vfe_cfg,
    )

    obs_history: List[np.ndarray] = [np.asarray(obs, dtype=np.float32)]
    action_history: List[int] = []

    total_collisions = 0
    reached_goal = False
    final_filtered_pose = None
    executed_steps=0

    for step_idx in range(cfg.max_steps_per_trial):
        obs_t = build_observation_tensor(obs_history, device)
        act_t = build_action_tensor(action_history, device)

        out = planner.plan(
            observations=obs_t,
            actions=act_t,
            z_goal=z_goal,
        )

        belief = out["belief"]
        final_filtered_pose = most_likely_pose_from_belief(belief)

        action = int(out["best_first_action"])

        obs, reward, done, info = step_env(env, action)
        executed_steps += 1
        true_pose = get_true_pose(info)

        obs_history.append(np.asarray(obs, dtype=np.float32))
        action_history.append(action)

        if len(obs_history) > cfg.history_keep:
            obs_history = obs_history[-cfg.history_keep:]

        if len(action_history) > cfg.history_keep - 1:
            action_history = action_history[-(cfg.history_keep - 1):]

        if bool(info["collision"]):
            total_collisions += 1

        if done or bool(info["reached_goal"]):
            reached_goal = True
            break

    return {
        "trial": trial_idx,
        "success": reached_goal,
        "reason": "ok" if reached_goal else "max_steps",
        "start_pose": start_pose,
        "goal_pos": goal_pos,
        "final_pose": true_pose,
        "steps": executed_steps,
        "collisions": total_collisions,
        "final_filtered_pose": final_filtered_pose,

    }


# ============================================================
# Main
# ============================================================

def main():
    cfg = EvalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(cfg.seed)

    print("=" * 110)
    print("EVALUATE LATENT-EFE V1: 50 TRIALS")
    print("=" * 110)
    print(f"Using device: {device}")
    print(f"Checkpoint: {cfg.checkpoint_path}")
    print(f"Num trials: {cfg.num_trials}")
    print(f"Max steps per trial: {cfg.max_steps_per_trial}")
    print(f"Fixed goal: {cfg.fixed_goal}")
    print(f"Online VFE enabled: {cfg.vfe_cfg.enabled}")
    print()

    model = build_model_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        device=device,
        model_config_json=cfg.model_config_json,
    )

    results: List[Dict[str, Any]] = []

    for trial_idx in range(cfg.num_trials):
        result = run_one_trial(
            trial_idx=trial_idx,
            model=model,
            cfg=cfg,
            device=device,
            rng=rng,
        )

        results.append(result)

        print(
            f"[{trial_idx + 1:03d}/{cfg.num_trials:03d}] "
            f"success={result['success']} | "
            f"steps={result['steps']:03d} | "
            f"collisions={result['collisions']:02d} | "
            f"start={result['start_pose']} | "
            f"goal={result['goal_pos']} | "
            f"final={result['final_pose']} | "
            f"reason={result['reason']}"
        )

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]

    success_rate = len(successes) / max(len(results), 1)

    avg_steps_success = (
        sum(r["steps"] for r in successes) / len(successes)
        if successes else float("nan")
    )

    avg_collisions = sum(r["collisions"] for r in results) / max(len(results), 1)

    print()
    print("=" * 110)
    print("FINAL 50-TRIAL SUMMARY")
    print("=" * 110)
    print(f"Trials:                  {len(results)}")
    print(f"Successes:               {len(successes)}")
    print(f"Failures:                {len(failures)}")
    print(f"Success rate:            {100.0 * success_rate:.2f}%")
    print(f"Avg steps successful:    {avg_steps_success:.2f}")
    print(f"Avg collisions/trial:    {avg_collisions:.2f}")

    if failures:
        print()
        print("Failed trials:")
        for r in failures:
            print(
                f"  trial={r['trial']} | start={r['start_pose']} | "
                f"goal={r['goal_pos']} | final={r['final_pose']} | "
                f"steps={r['steps']} | reason={r['reason']}"
            )

    with open(cfg.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trial",
                "success",
                "reason",
                "start_pose",
                "goal_pos",
                "final_pose",
                "steps",
                "collisions",
                "final_filtered_pose",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print()
    print(f"Saved CSV to: {cfg.output_csv}")


if __name__ == "__main__":
    main()
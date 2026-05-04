# run_latent_efe_v1.py

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from model_v3 import WorldModelV3, ModelV3Config
from simulator import TinyIndoorEnv, Pose, StepResult
from latent_efe_planner_v1 import LatentEFEPlannerV1, LatentEFEConfig, ACTION_NAMES, OnlineVFEConfig


HEADING_TO_IDX = {"N": 0, "E": 1, "S": 2, "W": 3}
IDX_TO_HEADING = {v: k for k, v in HEADING_TO_IDX.items()}

HEADING_DELTAS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


@dataclass
class RunConfig:
    checkpoint_path: str = "./checkpoints_v3_predictive_latent/best_predictive_latent.pt"
    model_config_json: str = "./checkpoints_v3/model_config.json"

    start_pose: Tuple[int, int, str] = (8, 3, "N")
    goal_pos: Tuple[int, int] = (5,5)

    max_steps: int = 80
    history_keep: int = 64

    # For pure latent-EFE testing, keep warmup empty.
    # You can set this back to (2, 3, 3, 2) if needed.
    warmup_actions: Tuple[int, ...] = ()

    topk_to_print: int = 5

    vfe_cfg: OnlineVFEConfig = field(
        default_factory=lambda: OnlineVFEConfig(
            enabled=True,
            num_steps=3,
            lr=0.15,
            entropy_weight=0.02,
            unreachable_weight=5.0,
        )
    )

    latent_cfg: LatentEFEConfig = field(
        default_factory=lambda: LatentEFEConfig(
            horizon=5,
            allow_backward=False,
            max_candidates=128,

            w_latent_risk=8.0,
            w_terminal_latent_risk=16.0,

            w_graph_path=18.0,
            w_terminal_graph_path=35.0,

            w_collision=30.0,
            w_entropy=0.05,
            w_info_gain=1.0,

            w_action=0.20,
            w_backward=2.50,
            w_turn=0.20,
            w_inverse=8.00,
            w_context_smoothness=0.15,

            discount=0.90,
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

    if "model_config" in ckpt and ckpt["model_config"] is not None:
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
# Env helpers
# ============================================================

def reset_env(
    env: TinyIndoorEnv,
    start_pose: Tuple[int, int, str],
    goal_pos: Tuple[int, int],
) -> Tuple[np.ndarray, Dict[str, Any]]:

    pose = Pose(row=start_pose[0], col=start_pose[1], heading=start_pose[2])

    return env.reset(
        start_pose=pose,
        goal_pos=goal_pos,
        use_goal=True,
    )


def step_env(env: TinyIndoorEnv, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    result: StepResult = env.step(action)
    return result.obs, float(result.reward), bool(result.done), result.info


def get_true_pose(info: Dict[str, Any]) -> Tuple[int, int, str]:
    return int(info["row"]), int(info["col"]), str(info["heading"])


def render_ascii(env: TinyIndoorEnv) -> str:
    return env.render_topdown_ascii(show_goal=True)


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


def forward_cell(r: int, c: int, h: int) -> Tuple[int, int]:
    dr, dc = HEADING_DELTAS[h]
    return r + dr, c + dc


def backward_cell(r: int, c: int, h: int) -> Tuple[int, int]:
    dr, dc = HEADING_DELTAS[h]
    return r - dr, c - dc


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

    obs_np = np.stack(
        [ensure_obs_shape(o) for o in obs_history],
        axis=0,
    ).astype(np.float32)

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
    act_t = build_action_tensor(action_history, device)

    out = model.forward_filter(obs_t, act_t)

    return {
        "row_probs": out["row_probs_seq"][:, -1, :],
        "col_probs": out["col_probs_seq"][:, -1, :],
        "heading_probs": out["heading_probs_seq"][:, -1, :],
        "context": out["context_seq"][:, -1, :],
        "raw": out,
    }


def most_likely_pose_from_belief(
    belief: Dict[str, torch.Tensor],
) -> Tuple[int, int, str]:

    r = int(torch.argmax(belief["row_probs"][0]).item())
    c = int(torch.argmax(belief["col_probs"][0]).item())
    h = int(torch.argmax(belief["heading_probs"][0]).item())

    return r, c, IDX_TO_HEADING[h]


def belief_entropy(belief: Dict[str, torch.Tensor]) -> float:
    eps = 1e-8

    def ent(p: torch.Tensor) -> torch.Tensor:
        p = p.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    e = (
        ent(belief["row_probs"])
        + ent(belief["col_probs"])
        + ent(belief["heading_probs"])
    )

    return float(e.item())


# ============================================================
# Reference rollout only for constructing z_goal
# ============================================================

def shortest_path_distance_map(env: TinyIndoorEnv, goal: Tuple[int, int]) -> np.ndarray:
    dist = np.full((env.rows, env.cols), np.inf, dtype=np.float32)

    gr, gc = goal
    if not valid_cell(env, gr, gc):
        return dist

    q: List[Tuple[int, int]] = [(gr, gc)]
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


def reference_action(
    env: TinyIndoorEnv,
    r: int,
    c: int,
    h: int,
    dist: np.ndarray,
) -> int:

    candidates: List[Tuple[float, int]] = []

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


@torch.no_grad()
def build_goal_context_from_reference_rollout(
    model: WorldModelV3,
    cfg: RunConfig,
    device: torch.device,
) -> torch.Tensor:

    env = TinyIndoorEnv(seed=999)

    obs, info = reset_env(env, cfg.start_pose, cfg.goal_pos)
    dist = shortest_path_distance_map(env, cfg.goal_pos)

    obs_hist: List[np.ndarray] = [np.asarray(obs, dtype=np.float32)]
    act_hist: List[int] = []

    reached = bool(info["reached_goal"])

    for _ in range(100):
        r, c, h_str = get_true_pose(info)
        h = HEADING_TO_IDX[h_str]

        if (r, c) == cfg.goal_pos:
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
        raise RuntimeError(
            "Could not construct z_goal because reference rollout did not reach the goal."
        )

    belief = infer_belief(
        model=model,
        obs_history=obs_hist,
        action_history=act_hist,
        device=device,
    )

    z_goal = belief["context"].detach().clone()

    print("Built z_goal from reference rollout.")
    print(f"Reference rollout length: {len(act_hist)}")
    print(f"Final reference pose: {get_true_pose(info)}")

    return z_goal


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = RunConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 110)
    print("RUN LATENT EFE V1")
    print("=" * 110)
    print(f"Using device: {device}")
    print(f"Checkpoint: {cfg.checkpoint_path}")
    print(f"Start pose: {cfg.start_pose}")
    print(f"Goal pos:   {cfg.goal_pos}")
    env = TinyIndoorEnv(seed=42)
    model = build_model_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        device=device,
        model_config_json=cfg.model_config_json,
    )

    
    obs, info = reset_env(env, cfg.start_pose, cfg.goal_pos)
    true_pose = get_true_pose(info)

    dist_t = shortest_path_distances(
        env=env,
        goal_rc=cfg.goal_pos,
        num_row_classes=model.cfg.num_row_classes,
        num_col_classes=model.cfg.num_col_classes,
        unreachable_cost=50.0,
    ).to(device)

    reachable_mask = build_reachable_mask_from_env(
        env=env,
        num_row_classes=model.cfg.num_row_classes,
        num_col_classes=model.cfg.num_col_classes,
    ).to(device)

    z_goal = build_goal_context_from_reference_rollout(
        model=model,
        cfg=cfg,
        device=device,
    )

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

    if len(cfg.warmup_actions) > 0:
        print()
        print("=" * 110)
        print("WARM-UP PHASE")
        print("=" * 110)

    for i, a in enumerate(cfg.warmup_actions):
        obs, reward, done, info = step_env(env, a)
        true_pose = get_true_pose(info)

        obs_history.append(np.asarray(obs, dtype=np.float32))
        action_history.append(a)

        print(f"Warm-up step {i}: action={a} ({ACTION_NAMES[a]})")
        print(render_ascii(env))
        print(f"Collision: {bool(info['collision'])} | Reached goal: {bool(info['reached_goal'])}")
        print()

        if bool(info["collision"]):
            total_collisions += 1

        if done or bool(info["reached_goal"]):
            reached_goal = True
            print("Goal reached during warm-up.")
            break

    if reached_goal:
        return

    print()
    print("=" * 110)
    print("LATENT-EFE PLANNING")
    print("=" * 110)

    for step_idx in range(cfg.max_steps):
        obs_t = build_observation_tensor(obs_history, device)
        act_t = build_action_tensor(action_history, device)

        with torch.no_grad():
            out = planner.plan(
                observations=obs_t,
                actions=act_t,
                z_goal=z_goal,
            )

        belief = out["belief"]
        pred_pose = most_likely_pose_from_belief(belief)
        ent = belief_entropy(belief)

        best_seq = out["best_sequence"]
        action = int(out["best_first_action"])

        print("=" * 110)
        print(f"Step {step_idx}")
        print(f"Current TRUE pose      : {true_pose}")
        print(f"Current FILTERED pose  : {pred_pose}")
        print(f"Belief entropy         : {ent:.4f}")
        print(f"Goal                   : {cfg.goal_pos}")
        print(f"Chosen sequence        : {best_seq}")
        print(f"Chosen first action    : {action} ({ACTION_NAMES[action]})")
        print(f"Best latent-EFE score  : {out['best_score']:.4f}")

        print("Top candidates:")
        for d in out["all_details"][: cfg.topk_to_print]:
            print(
                f"  seq={d['sequence']} | score={d['score']:.4f} | "
                f"latent={d['latent_risk']:.4f} | term_z={d['terminal_risk']:.4f} | "
                f"graph={d['graph']:.4f} | term_graph={d['terminal_graph']:.4f} | "
                f"coll={d['collision']:.4f} | ent={d['entropy']:.4f} | "
                f"info={d['info_gain']:.4f} | act={d['action']:.4f} | "
                f"inv={d['inverse']:.4f} | smooth={d['smooth']:.4f} | "
                f"final_zdist={d['final_latent_dist']:.4f}"
                f"gprog={d['graph_progress']:.4f} | "
                f"wall={d['wall_mass']:.4f} | nprog={d['no_progress']:.1f} | "
            )

        print()
        print("ASCII before action:")
        print(render_ascii(env))
        print()

        obs, reward, done, info = step_env(env, action)
        true_pose = get_true_pose(info)

        obs_history.append(np.asarray(obs, dtype=np.float32))
        action_history.append(action)

        if len(obs_history) > cfg.history_keep:
            obs_history = obs_history[-cfg.history_keep:]

        if len(action_history) > cfg.history_keep - 1:
            action_history = action_history[-(cfg.history_keep - 1):]

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
            print(f"Goal reached in {step_idx + 1} latent-EFE planning steps.")
            print()
            break

    print("=" * 110)
    print("FINAL SUMMARY")
    print("=" * 110)

    executed_steps = len(action_history) - len(cfg.warmup_actions)

    print(f"Total executed planning steps: {executed_steps}")
    print(f"Total collisions:             {total_collisions}")
    print(f"Final pose:                   {true_pose}")
    print(f"Goal:                         {cfg.goal_pos}")
    print(f"Reached goal:                 {reached_goal}")


if __name__ == "__main__":
    main()
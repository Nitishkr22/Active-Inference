# generate_dataset.py

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np

from simulator import TinyIndoorEnv, Pose, ACTION_NAMES, ACTION_TO_IDX, HEADINGS


# ============================================================
# Config
# ============================================================

@dataclass
class DatasetConfig:
    num_episodes: int = 1000
    seq_len: int = 16
    obs_height: int = 64
    obs_width: int = 64
    save_path: str = "tiny_nav_dataset_v6.npz"
    seed: int = 42

    # Behavior chunk probabilities
    prob_forward_run: float = 0.42
    prob_backward_run: float = 0.10
    prob_left_then_forward: float = 0.24
    prob_right_then_forward: float = 0.24

    # Chunk lengths (slightly shorter than v5)
    forward_run_min: int = 2
    forward_run_max: int = 4

    backward_run_min: int = 1
    backward_run_max: int = 3

    turn_forward_min: int = 2
    turn_forward_max: int = 5

    # Coverage bias
    use_visit_bias: bool = True
    visit_bias_strength: float = 0.15

    # Collision behavior
    post_collision_turn_bias: float = 3.0
    post_collision_backward_bias: float = 1.4
    keep_collision_examples: bool = True
    collision_retry_prob: float = 0.03

    log_every: int = 100


# ============================================================
# Utility helpers
# ============================================================

def normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = probs.sum()
    if total <= 0:
        return np.ones_like(probs) / len(probs)
    return probs / total


def heading_to_delta(heading: str) -> Tuple[int, int]:
    if heading == "N":
        return -1, 0
    if heading == "E":
        return 0, 1
    if heading == "S":
        return 1, 0
    if heading == "W":
        return 0, -1
    raise ValueError(f"Invalid heading: {heading}")


def next_cell(row: int, col: int, heading: str) -> Tuple[int, int]:
    dr, dc = heading_to_delta(heading)
    return row + dr, col + dc


def prev_cell(row: int, col: int, heading: str) -> Tuple[int, int]:
    dr, dc = heading_to_delta(heading)
    return row - dr, col - dc


def is_forward_blocked(env: TinyIndoorEnv, row: int, col: int, heading: str) -> bool:
    nr, nc = next_cell(row, col, heading)
    if not (0 <= nr < env.rows and 0 <= nc < env.cols):
        return True
    return env._is_blocking(nr, nc)


def is_backward_blocked(env: TinyIndoorEnv, row: int, col: int, heading: str) -> bool:
    nr, nc = prev_cell(row, col, heading)
    if not (0 <= nr < env.rows and 0 <= nc < env.cols):
        return True
    return env._is_blocking(nr, nc)


def sample_start_pose(env: TinyIndoorEnv, rng: np.random.Generator) -> Pose:
    idx = int(rng.integers(0, len(env.free_cells)))
    row, col = env.free_cells[idx]
    heading = HEADINGS[int(rng.integers(0, 4))]
    return Pose(row=row, col=col, heading=heading)


# ============================================================
# Collision-aware chunked exploration policy
# ============================================================

class CollisionAwareChunkedExplorationPolicy:
    """
    Chunked exploration policy with stronger collision recovery.

    Supported plan types:
      - forward_run
      - backward_run
      - left_then_forward
      - right_then_forward

    Improvements over v5:
      - collision clears current chunk immediately
      - next chunk after collision is biased toward turning
      - shorter forward runs reduce repeated wall-hits
    """

    def __init__(self, cfg: DatasetConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

        self.current_plan: List[str] = []
        self.prev_action: Optional[str] = None
        self.prev_turn: Optional[str] = None
        self.just_collided: bool = False

    def reset(self) -> None:
        self.current_plan = []
        self.prev_action = None
        self.prev_turn = None
        self.just_collided = False

    def choose_action(
        self,
        env: TinyIndoorEnv,
        visit_counts: Dict[Tuple[int, int], int],
    ) -> str:
        if not self.current_plan:
            self.current_plan = self._sample_new_plan(env, visit_counts)

        action = self.current_plan.pop(0)
        self.prev_action = action

        if action == "turn_left":
            self.prev_turn = "turn_left"
        elif action == "turn_right":
            self.prev_turn = "turn_right"

        return action

    def _sample_new_plan(
        self,
        env: TinyIndoorEnv,
        visit_counts: Dict[Tuple[int, int], int],
    ) -> List[str]:
        assert env.pose is not None

        row, col, heading = env.pose.row, env.pose.col, env.pose.heading

        f_blocked = is_forward_blocked(env, row, col, heading)
        b_blocked = is_backward_blocked(env, row, col, heading)

        plan_types = [
            "forward_run",
            "backward_run",
            "left_then_forward",
            "right_then_forward",
        ]

        probs = np.array([
            self.cfg.prob_forward_run,
            self.cfg.prob_backward_run,
            self.cfg.prob_left_then_forward,
            self.cfg.prob_right_then_forward,
        ], dtype=np.float64)

        # If forward blocked, reduce pure forward plan sharply.
        if f_blocked:
            probs[0] *= 0.15

        # If backward blocked, reduce pure backward plan sharply.
        if b_blocked:
            probs[1] *= 0.20

        # Avoid immediate turn oscillations a bit
        if self.prev_turn == "turn_left":
            probs[3] *= 0.65
        elif self.prev_turn == "turn_right":
            probs[2] *= 0.65

        # Strong post-collision bias: prefer turning or a short backward escape
        if self.just_collided:
            probs[2] *= self.cfg.post_collision_turn_bias
            probs[3] *= self.cfg.post_collision_turn_bias
            probs[1] *= self.cfg.post_collision_backward_bias
            probs[0] *= 0.10  # strongly discourage another immediate forward chunk

        # Coverage bias
        if self.cfg.use_visit_bias:
            scores = self._plan_visit_scores(env, visit_counts)
            probs *= scores

        probs = normalize_probs(probs)
        chosen = str(self.rng.choice(plan_types, p=probs))

        # Collision flag only influences immediate replanning once
        self.just_collided = False

        if chosen == "forward_run":
            k = int(self.rng.integers(self.cfg.forward_run_min, self.cfg.forward_run_max + 1))
            return ["forward"] * k

        if chosen == "backward_run":
            k = int(self.rng.integers(self.cfg.backward_run_min, self.cfg.backward_run_max + 1))
            return ["backward"] * k

        if chosen == "left_then_forward":
            k = int(self.rng.integers(self.cfg.turn_forward_min, self.cfg.turn_forward_max + 1))
            return ["turn_left"] + ["forward"] * k

        if chosen == "right_then_forward":
            k = int(self.rng.integers(self.cfg.turn_forward_min, self.cfg.turn_forward_max + 1))
            return ["turn_right"] + ["forward"] * k

        raise ValueError(f"Unknown plan type: {chosen}")

    def _plan_visit_scores(
        self,
        env: TinyIndoorEnv,
        visit_counts: Dict[Tuple[int, int], int],
    ) -> np.ndarray:
        assert env.pose is not None
        row, col, heading = env.pose.row, env.pose.col, env.pose.heading

        targets = []

        # forward_run target
        if is_forward_blocked(env, row, col, heading):
            targets.append((row, col))
        else:
            targets.append(next_cell(row, col, heading))

        # backward_run target
        if is_backward_blocked(env, row, col, heading):
            targets.append((row, col))
        else:
            targets.append(prev_cell(row, col, heading))

        # left_then_forward target
        left_heading = HEADINGS[(HEADINGS.index(heading) - 1) % 4]
        if is_forward_blocked(env, row, col, left_heading):
            targets.append((row, col))
        else:
            targets.append(next_cell(row, col, left_heading))

        # right_then_forward target
        right_heading = HEADINGS[(HEADINGS.index(heading) + 1) % 4]
        if is_forward_blocked(env, row, col, right_heading):
            targets.append((row, col))
        else:
            targets.append(next_cell(row, col, right_heading))

        scores = []
        for cell in targets:
            v = visit_counts.get(cell, 0)
            score = 1.0 / (1.0 + self.cfg.visit_bias_strength * float(v))
            score = 0.8 + 0.4 * score
            scores.append(score)

        return np.array(scores, dtype=np.float64)

    def on_collision(self) -> None:
        """
        Collision immediately clears current plan and marks next replanning
        as collision-recovery focused.
        """
        self.current_plan = []
        self.just_collided = True

    def maybe_keep_collision_example(self) -> bool:
        return bool(self.rng.random() < self.cfg.collision_retry_prob)


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(cfg: DatasetConfig) -> None:
    rng = np.random.default_rng(cfg.seed)

    env = TinyIndoorEnv(
        obs_height=cfg.obs_height,
        obs_width=cfg.obs_width,
        seed=cfg.seed,
    )

    policy = CollisionAwareChunkedExplorationPolicy(cfg, rng)

    N = cfg.num_episodes
    T = cfg.seq_len
    H = cfg.obs_height
    W = cfg.obs_width

    observations = np.zeros((N, T, 1, H, W), dtype=np.float32)
    actions = np.zeros((N, T - 1), dtype=np.int64)
    rows = np.zeros((N, T), dtype=np.int64)
    cols = np.zeros((N, T), dtype=np.int64)
    headings = np.zeros((N, T), dtype=np.int64)
    collisions = np.zeros((N, T - 1), dtype=np.uint8)

    start_rows = np.zeros((N,), dtype=np.int64)
    start_cols = np.zeros((N,), dtype=np.int64)
    start_headings = np.zeros((N,), dtype=np.int64)

    action_hist = {a: 0 for a in ACTION_NAMES}
    total_collision_count = 0
    unique_start_cells = set()
    unique_visited_cells = set()

    for ep in range(N):
        start_pose = sample_start_pose(env, rng)
        obs, info = env.reset(start_pose=start_pose, use_goal=False)

        policy.reset()

        visit_counts: Dict[Tuple[int, int], int] = {}

        start_cell = (int(info["row"]), int(info["col"]))
        visit_counts[start_cell] = 1

        unique_start_cells.add(start_cell)
        unique_visited_cells.add(start_cell)

        observations[ep, 0, 0] = obs
        rows[ep, 0] = int(info["row"])
        cols[ep, 0] = int(info["col"])
        headings[ep, 0] = int(info["heading_idx"])

        start_rows[ep] = int(info["row"])
        start_cols[ep] = int(info["col"])
        start_headings[ep] = int(info["heading_idx"])

        for t in range(T - 1):
            action_name = policy.choose_action(env, visit_counts)
            action_idx = ACTION_TO_IDX[action_name]

            result = env.step(action_name)

            observations[ep, t + 1, 0] = result.obs
            actions[ep, t] = action_idx
            rows[ep, t + 1] = int(result.info["row"])
            cols[ep, t + 1] = int(result.info["col"])
            headings[ep, t + 1] = int(result.info["heading_idx"])
            collisions[ep, t] = 1 if result.info["collision"] else 0

            action_hist[action_name] += 1
            total_collision_count += int(result.info["collision"])

            cell = (int(result.info["row"]), int(result.info["col"]))
            visit_counts[cell] = visit_counts.get(cell, 0) + 1
            unique_visited_cells.add(cell)

            if result.info["collision"]:
                # usually clear plan immediately and switch to collision recovery
                if not policy.maybe_keep_collision_example():
                    policy.on_collision()

        if (ep + 1) % cfg.log_every == 0 or (ep + 1) == N:
            collision_rate = total_collision_count / max((ep + 1) * (T - 1), 1)
            visited_ratio = len(unique_visited_cells) / max(len(env.free_cells), 1)
            print(
                f"[{ep + 1:6d}/{N}] episodes generated | "
                f"collision_rate_so_far={collision_rate:.4f} | "
                f"visited_cells={len(unique_visited_cells)}/{len(env.free_cells)} "
                f"({visited_ratio:.4f})"
            )

    save_dict = {
        "observations": observations,
        "actions": actions,
        "rows": rows,
        "cols": cols,
        "headings": headings,
        "collisions": collisions,
        "start_rows": start_rows,
        "start_cols": start_cols,
        "start_headings": start_headings,
        "action_names": np.array(ACTION_NAMES),
        "heading_names": np.array(HEADINGS),
    }

    np.savez_compressed(cfg.save_path, **save_dict)

    total_steps = N * (T - 1)

    print("\nDataset generation complete.")
    print(f"Saved to: {cfg.save_path}")
    print(f"observations shape: {observations.shape}")
    print(f"actions shape:      {actions.shape}")
    print(f"rows shape:         {rows.shape}")
    print(f"cols shape:         {cols.shape}")
    print(f"headings shape:     {headings.shape}")
    print(f"collisions shape:   {collisions.shape}")
    print(f"Total transitions:  {total_steps}")
    print(f"Collision rate:     {total_collision_count / max(total_steps, 1):.4f}")
    print(f"Unique start cells: {len(unique_start_cells)} / {len(env.free_cells)}")
    print(f"Unique visited cells: {len(unique_visited_cells)} / {len(env.free_cells)}")

    print("\nAction histogram:")
    for a in ACTION_NAMES:
        count = action_hist[a]
        frac = count / max(total_steps, 1)
        print(f"  {a:>10s}: {count:8d} ({frac:.4f})")


# ============================================================
# Inspection utility
# ============================================================

def inspect_dataset(npz_path: str) -> None:
    data = np.load(npz_path)

    print(f"\nInspecting: {npz_path}")
    for k in data.files:
        arr = data[k]
        print(f"{k:15s} shape={arr.shape} dtype={arr.dtype}")

    observations = data["observations"]
    actions = data["actions"]
    collisions = data["collisions"]
    rows = data["rows"]
    cols = data["cols"]

    visited_cells = set()
    N, T = rows.shape
    for ep in range(N):
        for t in range(T):
            visited_cells.add((int(rows[ep, t]), int(cols[ep, t])))

    print("\nQuick stats:")
    print(
        f"obs min/max/mean: "
        f"{observations.min():.4f} / {observations.max():.4f} / {observations.mean():.4f}"
    )
    print(f"collision rate:   {collisions.mean():.4f}")
    print(f"unique visited cells in saved dataset: {len(visited_cells)}")

    unique_actions, counts = np.unique(actions, return_counts=True)
    print("action distribution:")
    for a_idx, cnt in zip(unique_actions, counts):
        print(f"  {int(a_idx)} ({ACTION_NAMES[int(a_idx)]}): {int(cnt)}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate goal-free sequential dataset for TinyIndoorEnv using collision-aware chunked exploration."
    )

    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--obs_height", type=int, default=64)
    parser.add_argument("--obs_width", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="tiny_nav_dataset_v6.npz")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prob_forward_run", type=float, default=0.42)
    parser.add_argument("--prob_backward_run", type=float, default=0.10)
    parser.add_argument("--prob_left_then_forward", type=float, default=0.24)
    parser.add_argument("--prob_right_then_forward", type=float, default=0.24)

    parser.add_argument("--forward_run_min", type=int, default=2)
    parser.add_argument("--forward_run_max", type=int, default=4)
    parser.add_argument("--backward_run_min", type=int, default=1)
    parser.add_argument("--backward_run_max", type=int, default=3)
    parser.add_argument("--turn_forward_min", type=int, default=2)
    parser.add_argument("--turn_forward_max", type=int, default=5)

    parser.add_argument("--post_collision_turn_bias", type=float, default=3.0)
    parser.add_argument("--post_collision_backward_bias", type=float, default=1.4)
    parser.add_argument("--collision_retry_prob", type=float, default=0.03)

    parser.add_argument("--use_visit_bias", action="store_true")
    parser.add_argument("--no_visit_bias", action="store_true")
    parser.add_argument("--visit_bias_strength", type=float, default=0.15)

    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--inspect_only", action="store_true")

    args = parser.parse_args()

    use_visit_bias = True
    if args.use_visit_bias:
        use_visit_bias = True
    if args.no_visit_bias:
        use_visit_bias = False

    cfg = DatasetConfig(
        num_episodes=args.num_episodes,
        seq_len=args.seq_len,
        obs_height=args.obs_height,
        obs_width=args.obs_width,
        save_path=args.save_path,
        seed=args.seed,
        prob_forward_run=args.prob_forward_run,
        prob_backward_run=args.prob_backward_run,
        prob_left_then_forward=args.prob_left_then_forward,
        prob_right_then_forward=args.prob_right_then_forward,
        forward_run_min=args.forward_run_min,
        forward_run_max=args.forward_run_max,
        backward_run_min=args.backward_run_min,
        backward_run_max=args.backward_run_max,
        turn_forward_min=args.turn_forward_min,
        turn_forward_max=args.turn_forward_max,
        post_collision_turn_bias=args.post_collision_turn_bias,
        post_collision_backward_bias=args.post_collision_backward_bias,
        collision_retry_prob=args.collision_retry_prob,
        use_visit_bias=use_visit_bias,
        visit_bias_strength=args.visit_bias_strength,
        log_every=args.log_every,
    )
    return cfg, args.inspect_only


if __name__ == "__main__":
    cfg, inspect_only = parse_args()

    if inspect_only:
        if not os.path.exists(cfg.save_path):
            raise FileNotFoundError(f"Dataset file not found: {cfg.save_path}")
        inspect_dataset(cfg.save_path)
    else:
        generate_dataset(cfg)
        inspect_dataset(cfg.save_path)

        ## python3 generate_dataset.py --num_episodes 5000 --seq_len 32 --save_path ../../dataset/train_dataset__v6.npz ##
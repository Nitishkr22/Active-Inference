"""
world_model_training/collect_data.py — Robust multi-strategy data collector for RSSM training.

Inspired by the chunked, collision-aware, visit-biased policy in Symbolic_environment/generate_dataset.py.

Episode types (mixing ratio):
  30% — Chunked exploration   : forward runs, turn-then-forward, backward escape
  20% — Wall-following        : uses WallFollowExplorer (same as demo, covers room perimeter)
  30% — Goal-directed         : navmesh waypoints, simulate real nav behaviour
  20% — Recovery-focused      : starts near walls, forces collision+recovery data

For each episode, the agent resets to a RANDOM navigable position + random heading
so the dataset covers ALL rooms and viewpoints in the scene.

Scenarios covered
-----------------
  • Open-space straight navigation
  • Wall approach, contact, and avoidance (collision recovery)
  • Doorway approach and traversal (both sides)
  • Corner navigation (two walls meet)
  • Multi-room traversal via doorways
  • Various distances from obstacles (near / mid / far)
  • All 12 discrete headings (0°, 30°, 60°, …, 330°)

Data saved per step
-------------------
  rgb      [3, 64, 64]   float32  0-1  (downsampled from 256×256)
  depth    [1, 64, 64]   float32  0-1  (clipped at 8m, normalised)
  lidar    [64, 2]       float32  (angles_rad, dists_m)
  action   scalar        int64    0=FORWARD 1=TURN_L 2=TURN_R 3=STOP
  pose     [3]           float32  (x, y, theta_rad) V6 frame
  occ_gt   [64, 64]      float32  binary (1=obstacle)

Usage
-----
    conda activate habitat
    cd /home/nitish/Documents/AIF_code/Active-Inference
    python Symbolic_AIF_v6/world_model_training/collect_data.py

    # For a quick test run (500 steps):
    python Symbolic_AIF_v6/world_model_training/collect_data.py --total_steps 500

    # Inspect saved data:
    python Symbolic_AIF_v6/world_model_training/collect_data.py --inspect_only

Data is saved in:
    Symbolic_AIF_v6/world_model_training/data/episode_XXXXXX.pt
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE    = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _V6_ROOT)

from habitat_adapter.env_wrapper import HabitatEnv
from planner import WallFollowExplorer
from world_model_training.config import CollectConfig

IMG_SIZE   = 64    # CNN input resolution
CLIP_DEPTH = 8.0   # metres — normalisation ceiling for depth


# ── Image utilities ───────────────────────────────────────────────────────────

def _resize(t: torch.Tensor, size: int) -> torch.Tensor:
    """Resize [C, H, W] to [C, size, size]."""
    return F.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear",
                         align_corners=False).squeeze(0)


def _prepare_obs(rgb: torch.Tensor, depth: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Downsample and normalise a raw observation pair."""
    rgb_s   = _resize(rgb.float(), IMG_SIZE).clamp(0, 1)             # [3, 64, 64]
    d       = depth.squeeze(0) if depth.dim() == 3 else depth        # [H, W]
    d       = d.unsqueeze(0)                                          # [1, H, W]
    depth_s = _resize(d.float(), IMG_SIZE).clamp(0, CLIP_DEPTH) / CLIP_DEPTH  # [1, 64, 64]
    return rgb_s, depth_s


# ── Policy helpers ────────────────────────────────────────────────────────────

def _heading_err(target_angle: float, current_angle: float) -> float:
    return math.atan2(
        math.sin(target_angle - current_angle),
        math.cos(target_angle - current_angle),
    )


def _steer_toward(goal_xy: torch.Tensor, pose_v6: torch.Tensor,
                  turn_step_rad: float) -> int:
    """Return FORWARD / TURN_L / TURN_R to move toward goal_xy from pose_v6."""
    dx     = float(goal_xy[0] - pose_v6[0])
    dy     = float(goal_xy[1] - pose_v6[1])
    g_dir  = math.atan2(dy, dx)
    herr   = _heading_err(g_dir, float(pose_v6[2]))
    if abs(herr) <= math.radians(30):
        return 0   # FORWARD
    return 1 if herr > 0 else 2   # TURN_L / TURN_R


# ── Episode policies ──────────────────────────────────────────────────────────

class ChunkedPolicy:
    """
    Collision-aware chunked exploration.  Mirrors the toy-env policy.

    Plan types:
      forward_run         — straight line exploration
      backward_run        — retreat from wall
      left_then_forward   — turn left then go straight
      right_then_forward  — turn right then go straight
      spin                — full turn (discover new headings)
    """

    def __init__(self, rng: random.Random, turn_step_rad: float):
        self._rng  = rng
        self._ts   = turn_step_rad
        self._plan: List[int] = []
        self._just_collided   = False
        self._prev_turn: Optional[int] = None   # 1=L, 2=R
        self._visit_counts: Dict[Tuple[int, int], int] = {}

    def reset(self):
        self._plan          = []
        self._just_collided = False
        self._prev_turn     = None
        self._visit_counts  = {}

    def _key(self, pose_v6: torch.Tensor) -> Tuple[int, int]:
        return (int(float(pose_v6[0]) / 0.5), int(float(pose_v6[1]) / 0.5))

    def step(self, pose_v6: torch.Tensor, collided: bool) -> int:
        if collided:
            self._plan          = []
            self._just_collided = True

        # Track position for visit bias
        k = self._key(pose_v6)
        self._visit_counts[k] = self._visit_counts.get(k, 0) + 1

        if not self._plan:
            self._plan = self._new_plan()

        action = self._plan.pop(0)
        if action == 1: self._prev_turn = 1
        if action == 2: self._prev_turn = 2
        return action

    def _new_plan(self) -> List[int]:
        rng = self._rng
        # Weights for each plan type
        w = {"fwd": 0.35, "bwd": 0.10, "ltf": 0.25, "rtf": 0.25, "spin": 0.05}

        if self._just_collided:
            w["fwd"]  *= 0.10
            w["bwd"]  *= 1.50
            w["ltf"]  *= 2.50
            w["rtf"]  *= 2.50
            self._just_collided = False

        if self._prev_turn == 1: w["rtf"] *= 0.50  # avoid oscillation
        if self._prev_turn == 2: w["ltf"] *= 0.50

        total = sum(w.values())
        keys  = list(w.keys())
        probs = [w[k] / total for k in keys]
        chosen = rng.choices(keys, probs)[0]

        if chosen == "fwd":
            n = rng.randint(2, 5)
            return [0] * n
        if chosen == "bwd":
            n = rng.randint(1, 3)
            return [2, 2, 0] * n   # turn 180° then forward (no back action in Habitat)
        if chosen == "ltf":
            n = rng.randint(2, 6)
            return [1] + [0] * n
        if chosen == "rtf":
            n = rng.randint(2, 6)
            return [2] + [0] * n
        if chosen == "spin":
            # Full rotation: 360° / turn_step
            n_turns = max(1, round(2 * math.pi / self._ts))
            turn    = 1 if rng.random() < 0.5 else 2
            return [turn] * n_turns + [0, 0]
        return [0] * 3


class WallFollowPolicy:
    """Wraps WallFollowExplorer as a policy."""

    def __init__(self, turn_step_rad: float):
        self._ts = turn_step_rad
        self._wf: Optional[WallFollowExplorer] = None

    def reset(self):
        self._wf = WallFollowExplorer(
            turn_step   = self._ts,
            max_seek    = 8,
            max_follow  = 30,
            return_dist = 0.8,
        )

    @property
    def done(self) -> bool:
        return self._wf is None or self._wf.done

    def step(self, pose_v6: torch.Tensor, collided: bool) -> int:
        if self._wf is None or self._wf.done:
            return 0  # FORWARD fallback
        act = self._wf.get_action(pose_v6, collided)
        return act if act is not None else 0


class GoalDirectedPolicy:
    """
    Follow navmesh waypoints toward a sampled goal, then reset.
    Models the NAVIGATE phase behaviour from the demo.
    """

    def __init__(self, env: HabitatEnv, rng: random.Random, turn_step_rad: float):
        self._env  = env
        self._rng  = rng
        self._ts   = turn_step_rad
        self._wps: Optional[List[torch.Tensor]] = None
        self._wi   = 0
        self._blocked: set = set()

    def reset(self):
        self._wps     = None
        self._wi      = 0
        self._blocked = set()
        self._sample_goal()

    def _sample_goal(self):
        for _ in range(15):
            g   = self._env.sample_goal(min_dist=2.0, max_dist=7.0)
            wps = self._env.get_path_waypoints(g, waypoint_spacing=0.8)
            if wps is not None:
                self._wps = wps
                self._wi  = min(1, len(wps) - 1)
                return
        self._wps = None

    def step(self, pose_v6: torch.Tensor, collided: bool) -> int:
        if self._wps is None or self._wi >= len(self._wps):
            self._sample_goal()
        if self._wps is None:
            return self._rng.choice([1, 2])  # spin if no goal

        wp = self._wps[self._wi]
        dist = float((pose_v6[:2] - wp[:2]).norm())

        # Advance waypoint
        if dist < 0.8 and self._wi < len(self._wps) - 1:
            self._wi += 1
            self._blocked.clear()
            wp = self._wps[self._wi]

        # Sample new goal if current reached
        if self._wi == len(self._wps) - 1 and dist < 0.6:
            self._sample_goal()
            return 0

        action = _steer_toward(wp, pose_v6, self._ts)

        # Collision handling: try alternate heading
        if collided and action == 0:
            action = 1 if self._rng.random() < 0.5 else 2

        return action


class RecoveryPolicy:
    """
    Starts near a wall, intentionally hits it, then recovers.
    Produces rich collision+recovery data which the model needs
    to learn collision avoidance.
    """

    def __init__(self, rng: random.Random, turn_step_rad: float):
        self._rng  = rng
        self._ts   = turn_step_rad
        self._plan: List[int] = []
        self._phase           = "approach"   # approach → recover

    def reset(self):
        self._plan  = []
        self._phase = "approach"

    def step(self, pose_v6: torch.Tensor, collided: bool) -> int:
        if collided and self._phase == "approach":
            # Hit wall — now recover with random turns then forward
            n_turns = self._rng.randint(2, 5)
            turn    = 1 if self._rng.random() < 0.5 else 2
            self._plan  = [turn] * n_turns + [0, 0, 0, 0, 0]
            self._phase = "recover"

        if not self._plan:
            if self._phase == "recover":
                # After recovery, go straight for a bit then approach again
                self._plan  = [0] * self._rng.randint(3, 8)
                self._phase = "approach"
            else:
                self._plan  = [0] * self._rng.randint(3, 8)  # approach: forward

        return self._plan.pop(0)


# ── Reset helper ──────────────────────────────────────────────────────────────

def _reset_to_random(env: HabitatEnv, rng: random.Random
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reset the environment to a RANDOM navigable position and random heading.
    Returns (rgb, depth, cam_to_v6, pose_v6).

    This ensures dataset covers ALL rooms and viewpoints, not just the default start.
    """
    pf      = env.sim.pathfinder
    nav_pt  = pf.get_random_navigable_point()

    # Build a random heading (one of the 12 discrete steps = 0, 30, 60, …, 330°)
    n_steps = round(2 * math.pi / env.cfg.pose.turn_step_rad)
    step_i  = rng.randint(0, n_steps - 1)
    angle_v6 = step_i * env.cfg.pose.turn_step_rad   # yaw in V6

    # Convert V6 yaw to Habitat quaternion (Y-up world, robot looks -Z initially)
    import habitat_sim
    import magnum as mn
    agent_state = env.agent.get_state()
    agent_state.position = nav_pt
    # V6 forward=x → Habitat -Z; rotation around Y by (-angle_v6)
    agent_state.rotation = mn.Quaternion.rotation(
        mn.Rad(-angle_v6), mn.Vector3(0, 1, 0)
    )
    env.agent.set_state(agent_state)

    obs = env.sim.get_sensor_observations()
    return env._process_obs(obs)   # rgb, depth, cam_to_v6


# ── Main collector ────────────────────────────────────────────────────────────

POLICY_NAMES   = ["chunked", "wall_follow", "goal_directed", "recovery"]
POLICY_WEIGHTS = [0.30,       0.20,          0.30,            0.20]


def collect(cfg: CollectConfig, total_steps: Optional[int] = None,
            seed: Optional[int] = None) -> None:
    save_dir   = cfg.save_dir
    n_target   = total_steps or cfg.total_steps
    ep_steps   = cfg.episode_steps
    _seed      = seed if seed is not None else cfg.seed
    os.makedirs(save_dir, exist_ok=True)

    rng = random.Random(_seed)
    np.random.seed(_seed % (2 ** 31))

    env = HabitatEnv()
    ts  = math.radians(env.TURN_DEG)    # 30° → 0.5236 rad

    # Instantiate all policies
    policies = {
        "chunked":      ChunkedPolicy(rng, ts),
        "wall_follow":  WallFollowPolicy(ts),
        "goal_directed": GoalDirectedPolicy(env, rng, ts),
        "recovery":     RecoveryPolicy(rng, ts),
    }

    total_collected = 0
    episode_idx     = 0
    policy_counts   = {n: 0 for n in POLICY_NAMES}
    collision_count = 0
    t_start         = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"  Habitat World Model Data Collector")
    print(f"  Target steps : {n_target:,}")
    print(f"  Episode len  : {ep_steps} steps")
    print(f"  Save dir     : {os.path.abspath(save_dir)}")
    print(f"  Policies     : {dict(zip(POLICY_NAMES, POLICY_WEIGHTS))}")
    print(f"{'='*60}\n")

    while total_collected < n_target:
        # ── choose episode policy ──────────────────────────────────────────────
        policy_name = rng.choices(POLICY_NAMES, POLICY_WEIGHTS)[0]
        policy      = policies[policy_name]

        # ── reset to random position ──────────────────────────────────────────
        try:
            rgb_raw, depth_raw, cam_to_v6 = _reset_to_random(env, rng)
            pose_v6 = torch.from_numpy(env._get_pose_v6())
        except Exception:
            # Fallback: standard reset
            rgb_raw, depth_raw, cam_to_v6, pose_v6, _ = env.reset()

        policy.reset()

        ep: Dict[str, List] = {
            "rgb":    [],
            "depth":  [],
            "lidar":  [],
            "action": [],
            "pose":   [pose_v6.clone()],
            "occ_gt": [],
        }
        ep_collisions = 0
        prev_action   = 3   # STOP

        for _s in range(ep_steps):
            rgb_s, depth_s = _prepare_obs(rgb_raw, depth_raw)

            # LiDAR from full-res depth
            angles, dists = HabitatEnv.depth_to_lidar(depth_raw, n_beams=64)

            # Occupancy GT
            occ = env.get_occupancy_gt(
                resolution=cfg.occ_resolution,
                size=cfg.occ_grid_size,
            )
            occ_t = torch.from_numpy(occ)

            ep["rgb"].append(rgb_s)
            ep["depth"].append(depth_s)
            ep["lidar"].append(torch.stack([angles, dists], dim=-1))
            ep["occ_gt"].append(occ_t)

            # Policy selects action
            collided = False   # will be set from last step result
            action   = policy.step(pose_v6, collided)

            ep["action"].append(action)

            # Step
            rgb_raw, depth_raw, cam_to_v6, odom, pose_v6, done, info = env.step(action)
            collided = info.get("collided", False)
            if collided:
                ep_collisions += 1

            # Feed collision back to policy NEXT step (it reads `collided` arg)
            # For wall_follow and goal_directed this matters:
            # re-call step with updated collided=True so it replans
            if collided:
                policy.step(pose_v6, collided=True)   # replanning call (action discarded)

            ep["pose"].append(pose_v6.clone())
            prev_action = action

            if done:
                break

        # ── save episode ───────────────────────────────────────────────────────
        T = len(ep["action"])
        if T < 5:
            continue

        ep_data = {
            "rgb":         torch.stack(ep["rgb"]),           # [T, 3, 64, 64]
            "depth":       torch.stack(ep["depth"]),         # [T, 1, 64, 64]
            "lidar":       torch.stack(ep["lidar"]),         # [T, 64, 2]
            "action":      torch.tensor(ep["action"]),       # [T]
            "pose":        torch.stack(ep["pose"]),          # [T+1, 3]
            "occ_gt":      torch.stack(ep["occ_gt"]),        # [T, G, G]
            "policy":      policy_name,
            "ep_collisions": ep_collisions,
        }

        save_path = os.path.join(save_dir, f"episode_{episode_idx:06d}.pt")
        torch.save(ep_data, save_path)

        total_collected  += T
        episode_idx      += 1
        policy_counts[policy_name] += 1
        collision_count  += ep_collisions

        # ── progress report ────────────────────────────────────────────────────
        elapsed  = time.perf_counter() - t_start
        rate     = total_collected / elapsed
        eta      = (n_target - total_collected) / max(rate, 1e-9)
        coll_pct = 100.0 * collision_count / max(total_collected, 1)
        print(f"  ep {episode_idx:4d} [{policy_name:>13s}]"
              f" | steps {total_collected:6d}/{n_target}"
              f" | {rate:.0f} sps"
              f" | collision {coll_pct:.1f}%"
              f" | ETA {eta/60:.1f}min")

    env.close()
    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"  Collection complete")
    print(f"  Episodes       : {episode_idx}")
    print(f"  Steps total    : {total_collected}")
    print(f"  Time           : {elapsed/60:.1f} min")
    print(f"  Rate           : {total_collected/elapsed:.1f} sps")
    print(f"  Collision rate : {100.*collision_count/max(total_collected,1):.2f}%")
    print(f"  Policy mix:")
    for pn, cnt in policy_counts.items():
        print(f"    {pn:>15s}: {cnt:4d} episodes")
    print(f"  Data saved to  : {os.path.abspath(save_dir)}")
    print(f"{'='*60}\n")


# ── Dataset class (for train.py) ─────────────────────────────────────────────

class EpisodeDataset(torch.utils.data.Dataset):
    """
    Loads saved episode .pt files and yields non-overlapping sequences of `seq_len` steps.

    Each item is a dict:
      rgb            [seq_len, 3, 64, 64]   float32
      depth          [seq_len, 1, 64, 64]   float32
      lidar          [seq_len, 64, 2]       float32
      action_onehot  [seq_len, 4]           float32
      occ_gt         [seq_len, G, G]        float32
    """

    ACTION_DIM = 4

    def __init__(self, data_dir: str, seq_len: int = 20, split: str = "train",
                 val_split: float = 0.05, seed: int = 0):
        super().__init__()
        all_files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pt") and f.startswith("episode_")
        ])
        if not all_files:
            self.index = []
            self.files = []
            return

        rng = random.Random(seed)
        rng.shuffle(all_files)
        n_val = max(1, int(len(all_files) * val_split))
        self.files = all_files[:n_val] if split == "val" else all_files[n_val:]

        self.seq_len = seq_len
        self.index: List[Tuple[int, int]] = []

        for fi, fp in enumerate(self.files):
            try:
                meta = torch.load(fp, map_location="cpu")
                T    = int(meta["action"].shape[0])
                for start in range(0, T - seq_len + 1, seq_len):
                    self.index.append((fi, start))
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fi, start = self.index[idx]
        data = torch.load(self.files[fi], map_location="cpu")
        sl   = slice(start, start + self.seq_len)
        a    = data["action"][sl].long()
        a_oh = F.one_hot(a, num_classes=self.ACTION_DIM).float()
        return {
            "rgb":           data["rgb"][sl],
            "depth":         data["depth"][sl],
            "lidar":         data["lidar"][sl],
            "action_onehot": a_oh,
            "occ_gt":        data["occ_gt"][sl],
        }


# ── Inspection utility ────────────────────────────────────────────────────────

def inspect(data_dir: str) -> None:
    files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".pt") and f.startswith("episode_")
    ])
    if not files:
        print(f"No episodes found in {data_dir}")
        return

    total_steps   = 0
    policy_counts: Dict[str, int] = {}
    total_coll    = 0
    action_counts = [0, 0, 0, 0]

    for fp in files:
        d = torch.load(fp, map_location="cpu")
        T = int(d["action"].shape[0])
        total_steps += T
        pn = str(d.get("policy", "unknown"))
        policy_counts[pn] = policy_counts.get(pn, 0) + 1
        total_coll += int(d.get("ep_collisions", 0))
        for a in d["action"].tolist():
            action_counts[int(a)] += 1

    print(f"\nDataset inspection: {data_dir}")
    print(f"  Episodes     : {len(files)}")
    print(f"  Total steps  : {total_steps:,}")
    print(f"  Collision %  : {100.*total_coll/max(total_steps,1):.2f}%")
    print(f"  Action dist  : FWD={action_counts[0]} L={action_counts[1]}"
          f" R={action_counts[2]} STOP={action_counts[3]}")
    print(f"  Policy mix:")
    for pn, cnt in sorted(policy_counts.items()):
        print(f"    {pn:>15s}: {cnt} episodes")
    # Sample first episode shapes
    d0 = torch.load(files[0], map_location="cpu")
    print(f"\n  First episode shapes:")
    for k, v in d0.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:12s}: {tuple(v.shape)}  {v.dtype}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps",  type=int, default=50_000,
                        help="Total environment steps to collect (default 50000)")
    parser.add_argument("--episode_steps", type=int, default=200,
                        help="Max steps per episode (default 200)")
    parser.add_argument("--save_dir",     type=str,
                        default="Symbolic_AIF_v6/world_model_training/data",
                        help="Directory to save episode files")
    parser.add_argument("--occ_resolution", type=float, default=0.1)
    parser.add_argument("--occ_grid_size",  type=int,   default=64)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--inspect_only",   action="store_true",
                        help="Inspect existing dataset instead of collecting")
    args = parser.parse_args()

    if args.inspect_only:
        inspect(args.save_dir)
    else:
        cfg = CollectConfig(
            total_steps      = args.total_steps,
            episode_steps    = args.episode_steps,
            save_dir         = args.save_dir,
            occ_resolution   = args.occ_resolution,
            occ_grid_size    = args.occ_grid_size,
            seed             = args.seed,
        )
        collect(cfg, total_steps=args.total_steps, seed=args.seed)
        inspect(args.save_dir)

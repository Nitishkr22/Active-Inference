"""
habitat_demo.py — V6 Active Inference navigation in Habitat-Sim.

What this script does, step by step:
  1. Load the apartment_1 scene in Habitat-Sim with an RGBD agent
  2. Reset the agent to a random navigable start position
  3. Sample a goal position that is 3-8m (geodesic) from the start
  4. Create a WorldModelV6 (Faster R-CNN + EKF belief)
  5. Create an EFEPlannerV4 targeting the goal
  6. Run the navigation loop (max 80 steps):
       a. RGBD frame → Faster R-CNN detections → slot + pose update (VFE)
       b. EFE planner selects the best action from 243 imagined sequences
       c. Execute action in Habitat
       d. Print per-step log: pose estimate, goal dist, detections, action
  7. Report success / failure

Run with:
  conda activate habitat
  cd /home/nitish/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/habitat_adapter
  python habitat_demo.py

Expected runtime: ~5–15 seconds per step (Faster R-CNN + EFE on first run;
  should drop to ~0.5–1s after warm-up on GPU).
"""

from __future__ import annotations

import datetime
import math
import os
import random
import sys
import time
from typing import IO, List, Optional

import numpy as np
import torch

# ---- Add V6 model to path ----
_HERE   = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _V6_ROOT)

from model_v6 import WorldModelV6, ModelV6Config, BeliefState
from planner  import EFEPlannerV4
from habitat_adapter.env_wrapper import HabitatEnv

ACTION_NAMES = {0: "FORWARD", 1: "TURN_L", 2: "TURN_R", 3: "STOP"}
MAX_STEPS    = 80
GOAL_RADIUS  = 0.5   # metres — V6 config default

_LOG_DIR = os.path.join(_V6_ROOT, "results")


class _Tee:
    """Write to multiple streams simultaneously (stdout + log file)."""
    def __init__(self, *streams: IO):
        self._streams = streams
    def write(self, s: str) -> None:
        for st in self._streams:
            st.write(s)
            st.flush()
    def flush(self) -> None:
        for st in self._streams:
            st.flush()
    def fileno(self):
        return sys.__stdout__.fileno()


def main():
    # ------------------------------------------------------------------ #
    # 0. Set up log file (tee stdout → terminal + timestamped file)
    # ------------------------------------------------------------------ #
    os.makedirs(_LOG_DIR, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(_LOG_DIR, f"demo_{ts}.txt")
    log_file = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print(f"Logging to: {log_path}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ------------------------------------------------------------------ #
    # 1. Build V6 model
    # ------------------------------------------------------------------ #
    print("Loading WorldModelV6 (Faster R-CNN + EKF)...", end=" ", flush=True)
    cfg   = ModelV6Config()
    model = WorldModelV6(cfg).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"OK  ({n_params:,} learnable params)")

    # ------------------------------------------------------------------ #
    # 2. Start Habitat-Sim
    # ------------------------------------------------------------------ #
    print("Starting Habitat-Sim...", end=" ", flush=True)
    env = HabitatEnv()
    print("OK")

    # ------------------------------------------------------------------ #
    # 3. Reset episode + sample goal
    # ------------------------------------------------------------------ #
    rgb, depth, cam_to_v6, pose_v6, info = env.reset()
    env.sim.seed(random.randint(1, 1_000_000))   # randomise goal each run
    goal_v6 = env.sample_goal(min_dist=3.0, max_dist=8.0)

    print(f"\nStart pose (V6) : ({pose_v6[0]:.2f}, {pose_v6[1]:.2f}, "
          f"{math.degrees(pose_v6[2].item()):.1f}°)")
    print(f"Goal  (V6)      : ({goal_v6[0]:.2f}, {goal_v6[1]:.2f})")
    geo_dist = env.goal_dist_habitat(goal_v6)
    print(f"Geodesic dist   : {geo_dist:.2f}m")

    # ------------------------------------------------------------------ #
    # 4. Initial belief  (pose is known at episode start)
    # ------------------------------------------------------------------ #
    belief = model.initial_belief(device, known_pose=pose_v6.to(device))

    # ------------------------------------------------------------------ #
    # 5. EFE planner
    # ------------------------------------------------------------------ #
    planner = EFEPlannerV4(model=model, goal_pos=goal_v6.to(device), cfg=cfg.efe)

    # ------------------------------------------------------------------ #
    # 6. Navigation loop
    # ------------------------------------------------------------------ #
    print()
    print(f"{'Step':>4} | {'Action':>7} | "
          f"{'Pose (x, y, θ)':>28} | "
          f"{'GoalDist':>8} | {'Dets':>4} | {'Slots':>5} | {'ms':>6}")
    print("-" * 80)

    prev_action         = 3                               # STOP: no prior movement at episode start
    prev_odom           = torch.zeros(3, device=device)  # no odometry before first step
    blocked_headings: set = set()                         # headings (rad) where FORWARD hit a wall
    steps_no_progress   = 0                               # steps without meaningful goal-distance reduction
    prev_goal_dist      = float("inf")
    recent_poses: List[torch.Tensor] = [pose_v6.to(device)]
    reached       = False
    step_times_ms: List[float] = []

    for step in range(MAX_STEPS):
        t_step = time.perf_counter()

        # ---- Belief update: new RGBD observation + actual odom from prev action ----
        # odom is passed directly into forward_step so predict() uses it instead of
        # re-running kinematics — avoids applying the same turn/forward twice.
        with torch.no_grad():
            belief, aux = model.forward_step(
                rgb          = rgb.to(device),
                depth        = depth.to(device),
                action       = prev_action,
                cam_to_world = cam_to_v6.to(device),
                prev_belief  = belief,
                odom         = prev_odom,
                gt_pose      = None,
            )

        n_dets    = int(aux["n_matched"].item() + aux["n_new_slots"].item())
        n_slots   = int(belief.slot_conf_logit.gt(cfg.slots.conf_logit_empty_threshold).sum().item())
        goal_dist = planner.goal_distance(belief)

        # ---- Check goal reached ----
        if planner.reached_goal(belief):
            print(f"\n  >> Goal reached at step {step + 1}!")
            reached = True
            break

        # ---- EFE planning ----
        with torch.no_grad():
            plan = planner.select_action(belief, recent_poses,
                                         blocked_headings=blocked_headings)
        action = plan["best_action"]

        # ---- Log ----
        x, y, th = belief.pose_mu.tolist()
        ms = (time.perf_counter() - t_step) * 1000
        step_times_ms.append(ms)

        print(
            f"{step+1:>4} | {ACTION_NAMES[action]:>7} | "
            f"({x:6.2f}, {y:6.2f}, {math.degrees(th):5.1f}°) | "
            f"{goal_dist:8.3f}m | {n_dets:>4} | {n_slots:>5} | {ms:6.0f}"
        )

        # ---- Execute action in Habitat ----
        rgb, depth, cam_to_v6, odom, pose_v6, done, info = env.step(action)
        collided = info.get("collided", False)
        if collided:
            print(f"       [collision at step {step+1}]")

        # Update blocked_headings: add current heading when FORWARD is blocked;
        # clear when FORWARD succeeds (agent moved into a new area).
        if action == 0:
            turn_step = cfg.pose.turn_step_rad
            curr_h = math.atan2(math.sin(belief.pose_mu[2].item()),
                                 math.cos(belief.pose_mu[2].item()))
            rounded_h = round(curr_h / turn_step) * turn_step
            rounded_h = math.atan2(math.sin(rounded_h), math.cos(rounded_h))
            if collided:
                blocked_headings.add(rounded_h)

        # Progress-based escape: count steps where goal distance did NOT decrease
        # by at least 0.05 m. This catches loops where FORWARD occasionally
        # succeeds (resetting a forward-only counter) but in the wrong direction.
        if goal_dist < prev_goal_dist - 0.05:
            steps_no_progress = 0
            if action == 0 and not collided:
                blocked_headings.clear()   # moved toward goal — reset wall memory
        else:
            steps_no_progress += 1

        prev_goal_dist = goal_dist

        # Escape: if no meaningful progress in 8 steps, clear wall memory.
        if steps_no_progress >= 8:
            if blocked_headings:
                print(f"       [escape: clearing {len(blocked_headings)} blocked headings]")
            blocked_headings.clear()
            steps_no_progress = 0

        # Store odom and action for next iteration's forward_step call.
        # Do NOT call pose_est.predict here — that would double-apply the turn.
        prev_odom   = odom.to(device)
        prev_action = action

        recent_poses.append(belief.pose_mu.detach().clone())
        if len(recent_poses) > 10:
            recent_poses = recent_poses[-10:]

        if done:
            print(f"\n  >> Stop action executed at step {step+1}.")
            break

    # ------------------------------------------------------------------ #
    # 7. Results
    # ------------------------------------------------------------------ #
    final_dist = planner.goal_distance(belief)
    print()
    print("=" * 80)
    print(f"  Result          : {'SUCCESS' if reached else 'TIMEOUT / NOT REACHED'}")
    print(f"  Steps taken     : {step + 1}")
    print(f"  Final goal dist : {final_dist:.3f}m")
    if step_times_ms:
        import statistics
        print(f"  Mean ms/step    : {statistics.mean(step_times_ms):.0f} ms")
    print("=" * 80)

    env.close()

    # ------------------------------------------------------------------ #
    # Close log
    # ------------------------------------------------------------------ #
    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"\nLog saved → {log_path}")


if __name__ == "__main__":
    main()

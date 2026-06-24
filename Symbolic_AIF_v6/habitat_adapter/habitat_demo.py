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
    # 3. Reset episode + sample a REACHABLE goal
    # ------------------------------------------------------------------ #
    # Seed numpy before sample_goal so goal direction varies each run.
    # The Habitat sim seed controls start position (via get_random_navigable_point).
    _seed = random.randint(1, 1_000_000)
    env.sim.seed(_seed)
    np.random.seed(_seed % (2**31))
    rgb, depth, cam_to_v6, pose_v6, info = env.reset()

    # Keep resampling until the pathfinder can actually find a path.
    waypoints_v6: Optional[list] = None
    for _attempt in range(20):
        goal_v6 = env.sample_goal(min_dist=3.0, max_dist=8.0)
        waypoints_v6 = env.get_path_waypoints(goal_v6, waypoint_spacing=1.0)
        if waypoints_v6 is not None:
            break
        print(f"  [goal attempt {_attempt+1}: no path, resampling]")
    if waypoints_v6 is None:
        print("  [WARNING: no reachable goal found — navigating directly]")
        waypoints_v6 = [goal_v6]

    print(f"\nStart pose (V6) : ({pose_v6[0]:.2f}, {pose_v6[1]:.2f}, "
          f"{math.degrees(pose_v6[2].item()):.1f}°)")
    print(f"Goal  (V6)      : ({goal_v6[0]:.2f}, {goal_v6[1]:.2f})")
    print(f"Waypoints       : {len(waypoints_v6)}  "
          f"(spacing ~1m along navigable path)")
    for wi, wp in enumerate(waypoints_v6):
        print(f"   wp[{wi}] = ({wp[0]:.2f}, {wp[1]:.2f})")

    # ------------------------------------------------------------------ #
    # 4. Initial belief  (pose is known at episode start)
    # ------------------------------------------------------------------ #
    belief = model.initial_belief(device, known_pose=pose_v6.to(device))

    # ------------------------------------------------------------------ #
    # 5. EFE planner  (initially targets the first waypoint past start)
    # ------------------------------------------------------------------ #
    wp_idx      = min(1, len(waypoints_v6) - 1)   # skip waypoint 0 (≈ start)
    wp_advance  = 0.8                              # advance when within this dist
    current_wp  = waypoints_v6[wp_idx]
    planner = EFEPlannerV4(model=model, goal_pos=current_wp.to(device), cfg=cfg.efe)
    print(f"\nNavigating via {len(waypoints_v6)} waypoints "
          f"(EFE targets wp[{wp_idx}] first)")

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
    consecutive_non_fwd = 0                               # consecutive non-FORWARD actions
    wall_avoid_queue: List[int] = []                      # forced action queue for wall detour
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

        # ---- Waypoint advancement ----
        # Advance to the next waypoint when the agent is close enough to the
        # current one. The planner's goal is updated to the new waypoint.
        # The LAST waypoint is the actual goal — reaching it counts as success.
        curr_xy = belief.pose_mu[:2]
        wp_dist = float((curr_xy - current_wp.to(device)).norm())
        if wp_dist < wp_advance and wp_idx < len(waypoints_v6) - 1:
            wp_idx    += 1
            current_wp = waypoints_v6[wp_idx]
            planner.set_goal(current_wp.to(device))
            blocked_headings.clear()   # wall memory from previous segment is stale
            print(f"  >> Waypoint reached → now targeting wp[{wp_idx}]/"
                  f"{len(waypoints_v6)-1}: ({current_wp[0]:.2f}, {current_wp[1]:.2f})")

        goal_dist = planner.goal_distance(belief)   # distance to current waypoint / final goal

        # ---- Check final goal reached ----
        if wp_idx == len(waypoints_v6) - 1 and planner.reached_goal(belief):
            print(f"\n  >> Goal reached at step {step + 1}!")
            reached = True
            break

        # ---- EFE planning ----
        with torch.no_grad():
            plan = planner.select_action(belief, recent_poses,
                                         blocked_headings=blocked_headings)
        action = plan["best_action"]

        # ---- Forced wall-avoidance queue (highest priority) ----
        # Populated by the escape logic below; each element is an action int.
        if wall_avoid_queue:
            action = wall_avoid_queue.pop(0)
            print(f"       [wall-avoid: {ACTION_NAMES[action]}, "
                  f"{len(wall_avoid_queue)} left]")

        # ---- Spin-oscillation override (only when queue is empty) ----
        elif consecutive_non_fwd >= 6 and goal_dist < 3.0 and planner.goal_pos is not None:
            dx_g     = float(planner.goal_pos[0] - belief.pose_mu[0])
            dy_g     = float(planner.goal_pos[1] - belief.pose_mu[1])
            goal_dir = math.atan2(dy_g, dx_g)
            curr_h   = float(belief.pose_mu[2])
            herr     = math.atan2(math.sin(goal_dir - curr_h),
                                   math.cos(goal_dir - curr_h))
            if abs(herr) < math.radians(45):
                ov_action = 0
            elif herr > 0:
                ov_action = 1
            else:
                ov_action = 2
            # Only apply override if FORWARD at this heading is not already blocked.
            _blocked_fwd = False
            if ov_action == 0 and blocked_headings:
                _ts = cfg.pose.turn_step_rad
                _rh = math.atan2(math.sin(round(curr_h / _ts) * _ts),
                                  math.cos(round(curr_h / _ts) * _ts))
                _blocked_fwd = _rh in blocked_headings
            if not _blocked_fwd:
                action = ov_action
                consecutive_non_fwd = 0
                print(f"       [spin-override: goal_dir={math.degrees(goal_dir):.0f}°, "
                      f"h_err={math.degrees(herr):.0f}°, -> {ACTION_NAMES[action]}]")

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

        # Escape: no meaningful progress in 8 steps.
        # Build a forced action queue: turn to the perpendicular direction
        # (computed turns) + 3 FORWARD steps to physically clear the wall.
        if steps_no_progress >= 8:
            if blocked_headings and planner.goal_pos is not None and not wall_avoid_queue:
                dx_g      = float(planner.goal_pos[0] - belief.pose_mu[0])
                dy_g      = float(planner.goal_pos[1] - belief.pose_mu[1])
                _goal_dir = math.atan2(dy_g, dx_g)
                _curr_h   = float(belief.pose_mu[2])
                _ts       = cfg.pose.turn_step_rad
                # Pick whichever perpendicular (±90°) requires fewer turns
                _perp1    = _goal_dir + math.pi / 2
                _perp2    = _goal_dir - math.pi / 2
                _herr1    = math.atan2(math.sin(_perp1 - _curr_h),
                                        math.cos(_perp1 - _curr_h))
                _herr2    = math.atan2(math.sin(_perp2 - _curr_h),
                                        math.cos(_perp2 - _curr_h))
                if abs(_herr1) <= abs(_herr2):
                    _herr_p, _perp_dir = _herr1, _perp1
                else:
                    _herr_p, _perp_dir = _herr2, _perp2
                _n_turns  = max(1, round(abs(_herr_p) / _ts))
                _turn_dir = 1 if _herr_p > 0 else 2
                wall_avoid_queue = [_turn_dir] * _n_turns + [0, 0, 0]
                print(f"       [wall-escape: {_n_turns} turns to "
                      f"{math.degrees(_perp_dir):.0f}° + 3 FWD queued]")
            if blocked_headings:
                print(f"       [escape: clearing {len(blocked_headings)} blocked headings]")
            blocked_headings.clear()
            steps_no_progress = 0

        # Update spin counter: reset on any FORWARD, increment on turns.
        if action == 0:
            consecutive_non_fwd = 0
        else:
            consecutive_non_fwd += 1

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

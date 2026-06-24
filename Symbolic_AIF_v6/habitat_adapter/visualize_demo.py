"""
visualize_demo.py — V6 AIF navigation with side-by-side video output.

Saves results/viz_YYYYMMDD_HHMMSS.mp4 showing:
  Left  : first-person RGB camera view annotated with step / action / goal dist
  Right : 2D overhead trajectory in Habitat world frame
          (navmesh background, path taken, heading arrow, start ★, goal ✦)

Run:
  conda activate habitat
  cd .../Symbolic_AIF_v6/habitat_adapter
  python visualize_demo.py
"""

from __future__ import annotations

import datetime
import math
import os
import random
import sys
import time
from typing import List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")   # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

_HERE    = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _V6_ROOT)

from model_v6 import WorldModelV6, ModelV6Config, BeliefState
from planner  import EFEPlannerV4
from habitat_adapter.env_wrapper import HabitatEnv

ACTION_NAMES = {0: "FORWARD", 1: "TURN_L", 2: "TURN_R", 3: "STOP"}
MAX_STEPS    = 80
GOAL_RADIUS  = 0.5
_LOG_DIR     = os.path.join(_V6_ROOT, "results")

# ── video settings ────────────────────────────────────────────────────────────
FRAME_W  = 1280    # total frame width  (two 640-wide panels)
FRAME_H  = 540     # total frame height
VID_FPS  = 4       # slow enough to follow each step


# ── navmesh helpers ───────────────────────────────────────────────────────────

def _get_topdown(sim) -> Tuple[Optional[np.ndarray], Optional[object], float]:
    """
    Fetch a boolean navigability bitmap from the pathfinder.

    Returns (topdown_array, bounds, meters_per_pixel).
    topdown[row, col] == True  →  navigable.
    Rows index the Habitat Z axis (depth), cols index the Habitat X axis.
    """
    pf  = sim.pathfinder
    mpp = 0.05
    try:
        bounds   = pf.get_bounds()           # (min_corner, max_corner) as mn.Vector3
        height   = float(bounds[0][1]) + 0.5
        topdown  = pf.get_topdown_view(mpp, height)
        return topdown, bounds, mpp
    except Exception:
        return None, None, mpp


def _v6_to_hab(x_v6: float, y_v6: float) -> Tuple[float, float]:
    """V6 (x, y) → Habitat (X, Z) for overlaying onto topdown axes."""
    return -y_v6, -x_v6   # X_hab = -y_v6,  Z_hab = -x_v6


# ── per-step frame renderer ───────────────────────────────────────────────────

def _render_frame(
    rgb_np:       np.ndarray,                    # [H, W, 3] float32 0-1
    belief_pose:  List[float],                   # [x, y, theta] in V6
    goal_v6:      torch.Tensor,                  # [2]
    step:         int,
    action_name:  str,
    goal_dist:    float,
    path_v6:      List[Tuple[float, float]],     # list of (x, y) V6
    start_v6:     Tuple[float, float],
    topdown:      Optional[np.ndarray],
    bounds,                                      # habitat_sim bounds object or None
    waypoints_v6: Optional[List[torch.Tensor]] = None,   # planned waypoints
    wp_idx:       int = 0,                               # current waypoint index
) -> np.ndarray:
    """Return a (FRAME_H, FRAME_W, 3) uint8 RGB frame."""

    fig, (ax_rgb, ax_map) = plt.subplots(
        1, 2,
        figsize=(FRAME_W / 100, FRAME_H / 100),
        dpi=100,
    )

    # ── left panel: RGB camera view ──────────────────────────────────────────
    ax_rgb.imshow(rgb_np)
    ax_rgb.set_title(
        f"Step {step:>3}  |  {action_name:<7}  |  Goal: {goal_dist:.2f} m\n"
        f"Pose ({belief_pose[0]:5.2f}, {belief_pose[1]:5.2f})  "
        f"θ = {math.degrees(belief_pose[2]):6.1f}°",
        fontsize=9,
        loc="center",
        pad=4,
    )
    ax_rgb.axis("off")

    # ── right panel: 2-D overhead map ────────────────────────────────────────
    ax_map.set_facecolor("#e8e8e8")
    ax_map.set_aspect("equal")

    # Navmesh background (Habitat X on horizontal, Habitat Z on vertical)
    if topdown is not None and bounds is not None:
        X_min, Z_min = float(bounds[0][0]), float(bounds[0][2])
        X_max, Z_max = float(bounds[1][0]), float(bounds[1][2])
        ax_map.imshow(
            topdown,
            cmap="Blues",
            alpha=0.35,
            origin="lower",
            extent=[X_min, X_max, Z_min, Z_max],
            zorder=0,
        )
        ax_map.set_xlim(X_min - 0.3, X_max + 0.3)
        ax_map.set_ylim(Z_min - 0.3, Z_max + 0.3)

    # Path taken
    if len(path_v6) > 1:
        hab_path = [_v6_to_hab(x, y) for x, y in path_v6]
        xs_h = [p[0] for p in hab_path]
        zs_h = [p[1] for p in hab_path]
        ax_map.plot(xs_h, zs_h, color="royalblue", linewidth=1.8,
                    alpha=0.7, zorder=2)
        ax_map.scatter(xs_h[1:], zs_h[1:], s=10, c="royalblue",
                       alpha=0.5, zorder=3)

    # Start marker
    sx_h, sz_h = _v6_to_hab(*start_v6)
    ax_map.scatter([sx_h], [sz_h], s=120, c="gold", marker="*",
                   edgecolors="black", linewidths=0.5, zorder=5, label="Start")

    # Goal marker + radius circle
    gx_v6, gy_v6 = float(goal_v6[0]), float(goal_v6[1])
    gx_h, gz_h   = _v6_to_hab(gx_v6, gy_v6)
    ax_map.scatter([gx_h], [gz_h], s=160, c="red", marker="*",
                   edgecolors="darkred", linewidths=0.5, zorder=5, label="Goal")
    circle = plt.Circle((gx_h, gz_h), GOAL_RADIUS,
                         color="red", fill=False, linestyle="--",
                         linewidth=1.2, alpha=0.6, zorder=4)
    ax_map.add_patch(circle)

    # Planned waypoints
    if waypoints_v6 and len(waypoints_v6) > 1:
        wps_hab = [_v6_to_hab(float(w[0]), float(w[1])) for w in waypoints_v6]
        wx_h = [p[0] for p in wps_hab]
        wz_h = [p[1] for p in wps_hab]
        ax_map.plot(wx_h, wz_h, color="orange", linewidth=1.2,
                    linestyle=":", alpha=0.7, zorder=2, label="Path plan")
        ax_map.scatter(wx_h, wz_h, s=25, c="orange", alpha=0.6, zorder=3)
        # Highlight current target waypoint
        if wp_idx < len(wps_hab):
            ax_map.scatter([wps_hab[wp_idx][0]], [wps_hab[wp_idx][1]],
                           s=80, c="darkorange", marker="D",
                           edgecolors="black", linewidths=0.5, zorder=6,
                           label=f"Target wp[{wp_idx}]")

    # Current pose: dot + heading arrow
    cx_v6, cy_v6, cth = belief_pose
    cx_h, cz_h = _v6_to_hab(cx_v6, cy_v6)
    ax_map.scatter([cx_h], [cz_h], s=100, c="limegreen",
                   edgecolors="darkgreen", linewidths=0.8, zorder=6)
    # heading in Hab frame: X_hab = -y_v6, Z_hab = -x_v6
    # dx_hab/dθ = -dy_v6/dθ = -sin(θ) ; dz_hab/dθ = -dx_v6/dθ = -cos(θ)
    arrow_len = 0.5
    dX = -math.sin(cth) * arrow_len
    dZ = -math.cos(cth) * arrow_len
    ax_map.annotate(
        "",
        xy=(cx_h + dX, cz_h + dZ),
        xytext=(cx_h, cz_h),
        arrowprops=dict(arrowstyle="->", color="darkgreen", lw=2),
        zorder=7,
    )

    ax_map.set_xlabel("Habitat X  →  (V6 -y)", fontsize=7)
    ax_map.set_ylabel("Habitat Z  →  (V6 -x)", fontsize=7)
    ax_map.set_title("Top-down trajectory", fontsize=9)
    ax_map.legend(fontsize=7, loc="upper right", markerscale=0.8)
    ax_map.grid(True, alpha=0.25, linewidth=0.5)
    ax_map.tick_params(labelsize=6)

    fig.tight_layout(pad=0.8)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)

    # Resize to exact target dimensions
    if (w, h) != (FRAME_W, FRAME_H):
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    return frame   # RGB uint8


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(_LOG_DIR, exist_ok=True)
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    vid_path  = os.path.join(_LOG_DIR, f"viz_{ts}.mp4")
    log_path  = os.path.join(_LOG_DIR, f"demo_{ts}.txt")

    # ── log file ──────────────────────────────────────────────────────────────
    log_file = open(log_path, "w", buffering=1)

    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"Logging to: {log_path}")
    log(f"Video   to: {vid_path}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device : {device}")

    # ── model ─────────────────────────────────────────────────────────────────
    log("Loading WorldModelV6...", )
    cfg   = ModelV6Config()
    model = WorldModelV6(cfg).to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"OK  ({n_params:,} learnable params)")

    # ── Habitat ───────────────────────────────────────────────────────────────
    log("Starting Habitat-Sim...", )
    env = HabitatEnv()
    log("OK")

    # ── navmesh topdown map ───────────────────────────────────────────────────
    topdown, bounds, _mpp = _get_topdown(env.sim)
    if topdown is not None:
        log(f"Navmesh topdown: {topdown.shape} at {_mpp} m/px")
    else:
        log("Navmesh topdown not available — plain background will be used")

    # ── video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vid_path, fourcc, VID_FPS, (FRAME_W, FRAME_H))

    # ── episode reset ─────────────────────────────────────────────────────────
    _seed = random.randint(1, 1_000_000)
    env.sim.seed(_seed)
    np.random.seed(_seed % (2**31))
    rgb, depth, cam_to_v6, pose_v6, info = env.reset()

    # Sample a goal that is geodesically reachable (pathfinder can find a path).
    waypoints_v6 = None
    for _attempt in range(20):
        goal_v6 = env.sample_goal(min_dist=3.0, max_dist=8.0)
        waypoints_v6 = env.get_path_waypoints(goal_v6, waypoint_spacing=1.0)
        if waypoints_v6 is not None:
            break
        log(f"  [goal attempt {_attempt+1}: no path, resampling]")
    if waypoints_v6 is None:
        log("  [WARNING: no reachable goal found — navigating directly]")
        waypoints_v6 = [goal_v6]

    start_xy = (float(pose_v6[0]), float(pose_v6[1]))

    log(f"\nStart pose (V6) : ({pose_v6[0]:.2f}, {pose_v6[1]:.2f}, "
        f"{math.degrees(pose_v6[2].item()):.1f}°)")
    log(f"Goal  (V6)      : ({goal_v6[0]:.2f}, {goal_v6[1]:.2f})")
    log(f"Waypoints       : {len(waypoints_v6)}")

    # ── belief + planner (initially targeting first waypoint past start) ──────
    belief      = model.initial_belief(device, known_pose=pose_v6.to(device))
    wp_idx      = min(1, len(waypoints_v6) - 1)
    wp_advance  = 0.8
    current_wp  = waypoints_v6[wp_idx]
    planner = EFEPlannerV4(model=model, goal_pos=current_wp.to(device), cfg=cfg.efe)

    # ── loop state ────────────────────────────────────────────────────────────
    log()
    log(f"{'Step':>4} | {'Action':>7} | "
        f"{'Pose (x, y, θ)':>28} | "
        f"{'GoalDist':>8} | {'Dets':>4} | {'Slots':>5} | {'ms':>6}")
    log("-" * 80)

    prev_action         = 3
    prev_odom           = torch.zeros(3, device=device)
    blocked_headings: set = set()
    steps_no_progress   = 0
    prev_goal_dist      = float("inf")
    consecutive_non_fwd = 0
    wall_avoid_queue: List[int] = []
    recent_poses: List[torch.Tensor] = [pose_v6.to(device)]
    path_xy: List[Tuple[float, float]] = [start_xy]
    reached        = False
    step_times_ms: List[float] = []

    for step in range(MAX_STEPS):
        t_step = time.perf_counter()

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
        n_slots   = int(belief.slot_conf_logit.gt(
                         cfg.slots.conf_logit_empty_threshold).sum().item())

        # ── Waypoint advancement ───────────────────────────────────────────────
        curr_xy = belief.pose_mu[:2]
        wp_dist = float((curr_xy - current_wp.to(device)).norm())
        if wp_dist < wp_advance and wp_idx < len(waypoints_v6) - 1:
            wp_idx    += 1
            current_wp = waypoints_v6[wp_idx]
            planner.set_goal(current_wp.to(device))
            blocked_headings.clear()
            log(f"  >> wp[{wp_idx}]/{len(waypoints_v6)-1}: "
                f"({current_wp[0]:.2f}, {current_wp[1]:.2f})")

        goal_dist = planner.goal_distance(belief)

        if wp_idx == len(waypoints_v6) - 1 and planner.reached_goal(belief):
            log(f"\n  >> Goal reached at step {step + 1}!")
            reached = True
            # Render a final "reached" frame
            rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
            frame  = _render_frame(
                rgb_np, belief.pose_mu.tolist(), goal_v6,
                step + 1, "REACHED", goal_dist,
                path_xy, start_xy, topdown, bounds,
                waypoints_v6=waypoints_v6, wp_idx=wp_idx,
            )
            # Write a few copies so the success frame lingers
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            for _ in range(VID_FPS * 2):
                writer.write(bgr)
            break

        # ── EFE planning ──────────────────────────────────────────────────────
        with torch.no_grad():
            plan = planner.select_action(belief, recent_poses,
                                         blocked_headings=blocked_headings)
        action = plan["best_action"]

        # ── Forced wall-avoidance queue (highest priority) ───────────────────
        if wall_avoid_queue:
            action = wall_avoid_queue.pop(0)
            log(f"       [wall-avoid: {ACTION_NAMES[action]}, "
                f"{len(wall_avoid_queue)} left]")

        # ── Spin-oscillation override (only when queue empty) ─────────────────
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
            _blocked_fwd = False
            if ov_action == 0 and blocked_headings:
                _ts = cfg.pose.turn_step_rad
                _rh = math.atan2(math.sin(round(curr_h / _ts) * _ts),
                                  math.cos(round(curr_h / _ts) * _ts))
                _blocked_fwd = _rh in blocked_headings
            if not _blocked_fwd:
                action = ov_action
                consecutive_non_fwd = 0
                log(f"       [spin-override: goal_dir={math.degrees(goal_dir):.0f}°, "
                    f"h_err={math.degrees(herr):.0f}°, -> {ACTION_NAMES[action]}]")

        # ── log line ──────────────────────────────────────────────────────────
        x, y, th = belief.pose_mu.tolist()
        ms = (time.perf_counter() - t_step) * 1000
        step_times_ms.append(ms)
        log(
            f"{step+1:>4} | {ACTION_NAMES[action]:>7} | "
            f"({x:6.2f}, {y:6.2f}, {math.degrees(th):5.1f}°) | "
            f"{goal_dist:8.3f}m | {n_dets:>4} | {n_slots:>5} | {ms:6.0f}"
        )

        # ── render & write video frame ─────────────────────────────────────────
        rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
        frame  = _render_frame(
            rgb_np, [x, y, th], goal_v6,
            step + 1, ACTION_NAMES[action], goal_dist,
            path_xy, start_xy, topdown, bounds,
            waypoints_v6=waypoints_v6, wp_idx=wp_idx,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # ── execute action ────────────────────────────────────────────────────
        rgb, depth, cam_to_v6, odom, pose_v6, done, info = env.step(action)
        collided = info.get("collided", False)
        if collided:
            log(f"       [collision at step {step+1}]")

        if action == 0:
            turn_step = cfg.pose.turn_step_rad
            curr_h = math.atan2(math.sin(belief.pose_mu[2].item()),
                                 math.cos(belief.pose_mu[2].item()))
            rounded_h = round(curr_h / turn_step) * turn_step
            rounded_h = math.atan2(math.sin(rounded_h), math.cos(rounded_h))
            if collided:
                blocked_headings.add(rounded_h)

        if goal_dist < prev_goal_dist - 0.05:
            steps_no_progress = 0
            if action == 0 and not collided:
                blocked_headings.clear()
        else:
            steps_no_progress += 1

        prev_goal_dist = goal_dist

        if steps_no_progress >= 8:
            if blocked_headings and planner.goal_pos is not None and not wall_avoid_queue:
                dx_g      = float(planner.goal_pos[0] - belief.pose_mu[0])
                dy_g      = float(planner.goal_pos[1] - belief.pose_mu[1])
                _goal_dir = math.atan2(dy_g, dx_g)
                _curr_h   = float(belief.pose_mu[2])
                _ts       = cfg.pose.turn_step_rad
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
                log(f"       [wall-escape: {_n_turns} turns to "
                    f"{math.degrees(_perp_dir):.0f}° + 3 FWD queued]")
            if blocked_headings:
                log(f"       [escape: clearing {len(blocked_headings)} blocked headings]")
            blocked_headings.clear()
            steps_no_progress = 0

        if action == 0:
            consecutive_non_fwd = 0
        else:
            consecutive_non_fwd += 1

        prev_odom   = odom.to(device)
        prev_action = action

        recent_poses.append(belief.pose_mu.detach().clone())
        if len(recent_poses) > 10:
            recent_poses = recent_poses[-10:]

        # Track true trajectory for map overlay
        path_xy.append((float(pose_v6[0]), float(pose_v6[1])))

        if done:
            log(f"\n  >> Stop action executed at step {step+1}.")
            break

    # ── results ───────────────────────────────────────────────────────────────
    final_dist = planner.goal_distance(belief)
    log()
    log("=" * 80)
    log(f"  Result          : {'SUCCESS' if reached else 'TIMEOUT / NOT REACHED'}")
    log(f"  Steps taken     : {step + 1}")
    log(f"  Final goal dist : {final_dist:.3f} m")
    if step_times_ms:
        import statistics
        log(f"  Mean ms/step    : {statistics.mean(step_times_ms):.0f} ms")
    log("=" * 80)

    writer.release()
    env.close()
    log_file.close()

    print(f"\nVideo saved → {vid_path}")
    print(f"Log   saved → {log_path}")


if __name__ == "__main__":
    main()

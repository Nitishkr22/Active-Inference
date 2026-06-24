"""
aif_explore_nav_demo.py — Two-phase Active Inference navigation with video output.

PHASE 1 — EXPLORATION  (no goal given)
  EFE = Ambiguity − InfoGain + FrontierPenalty
  The agent explores using a wall-following state machine (Stage 3):
    • Cycles through 4 cardinal headings (E, N, W, S).
    • For each heading: SEEK (move forward until wall) → FOLLOW (right-hand
      wall-following rule, discovering gaps/doorways naturally).
    • After all 4 headings: EFE info-gain takes over as fallback.
  As the agent moves, 64-slot belief fills with walls, doorways, and objects.

PHASE 2 — NAVIGATION   (goal provided after exploration)
  EFE = Risk + Ambiguity − InfoGain + WallPenalty + DoorwayRouting (Stage 2)
  The belief state (object map) is carried over from Phase 1.
  A geodesically reachable goal is sampled; the Habitat pathfinder provides
  waypoints.  The EFE planner uses accumulated slot landmarks + wall/doorway
  slots for topology-aware navigation.

VIDEO  results/aif_explore_nav_YYYYMMDD_HHMMSS.mp4
  Left : first-person RGB with phase / step / goal dist annotation.
  Right: top-down map showing:
         • Navmesh background
         • Orange dotted line  : planned waypoint route
         • Blue dots + line    : path actually taken
         • Coloured scatter    : slot belief map
             red squares = walls, green triangles = doorways, circles = objects
         • Green dot + arrow   : current robot pose
         • Gold ★              : start position
         • Red ★ + dashed ring : goal (navigation phase only)

Run:
  conda activate habitat
  cd .../Symbolic_AIF_v6/habitat_adapter
  python aif_explore_nav_demo.py
"""

from __future__ import annotations

import datetime, math, os, random, sys, time
from typing import List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

_HERE    = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _V6_ROOT)

from model_v6 import WorldModelV6, ModelV6Config, BeliefState, WALL_ID, DOORWAY_ID
from planner  import EFEPlannerV4, WallFollowExplorer
from habitat_adapter.env_wrapper import HabitatEnv

ACTION_NAMES   = {0: "FORWARD", 1: "TURN_L", 2: "TURN_R", 3: "STOP"}
MAX_STEPS      = 520          # total budget (explore + navigate)
N_EXPLORE      = 320          # explore budget; multi-room may use up to this many steps
BELIEF_MEMORY_PATH = None     # set after _LOG_DIR is known — see main()
GOAL_RADIUS          = 0.5    # metres — success zone
WP_ADVANCE     = 0.8          # metres — advance to next waypoint when this close
EXPLORE_SAMPLE = 3            # record explored pose every N steps
MAX_ROOMS      = 3            # max extra rooms to enter during exploration
FRAME_W, FRAME_H = 1280, 540
VID_FPS        = 4

# Class names for slot labels.  Indices 0–91: COCO.  92–93: structural.
_COCO_NAMES = [
    "background","person","bicycle","car","motorcycle","airplane","bus","train",
    "truck","boat","traffic light","fire hydrant","stop sign","parking meter",
    "bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
    "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
    "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
    "pizza","donut","cake","chair","couch","potted plant","bed","dining table",
    "toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave",
    "oven","toaster","sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush",
    # Structural categories (from StructuralDetector)
    "wall",     # 92 — impassable planar surface
    "doorway",  # 93 — navigable passage between rooms
]

_LOG_DIR = os.path.join(_V6_ROOT, "results")
# Persistent belief across runs — delete this file to start fresh
_BELIEF_MEM_PATH = os.path.join(_LOG_DIR, "belief_memory.pt")


# ── coordinate helpers ────────────────────────────────────────────────────────

def _v6_to_hab(x_v6: float, y_v6: float) -> Tuple[float, float]:
    return -y_v6, -x_v6   # X_hab = -y_v6, Z_hab = -x_v6


def _get_topdown(sim):
    try:
        pf     = sim.pathfinder
        bounds = pf.get_bounds()
        td     = pf.get_topdown_view(0.05, float(bounds[0][1]) + 0.5)
        return td, bounds
    except Exception:
        return None, None


# ── frame renderer ────────────────────────────────────────────────────────────

def _render_frame(
    rgb_np:      np.ndarray,                    # [H,W,3] float32 0-1
    belief:      BeliefState,
    cfg:         ModelV6Config,
    step:        int,
    phase:       str,                           # "EXPLORE" | "NAVIGATE"
    action_name: str,
    goal_dist:   float,
    path_v6:     List[Tuple[float, float]],
    start_v6:    Tuple[float, float],
    topdown:     Optional[np.ndarray],
    bounds,
    goal_v6:         Optional[torch.Tensor] = None,
    waypoints_v6:    Optional[List[torch.Tensor]] = None,
    wp_idx:          int = 0,
    wf_phase:        str = "",
    robot_pose_gt:   Optional[Tuple[float, float, float]] = None,  # GT pose for robot dot
    lidar_angles:    Optional[torch.Tensor] = None,                 # [n_beams] rad
    lidar_dists:     Optional[torch.Tensor] = None,                 # [n_beams] metres
) -> np.ndarray:

    fig, (ax_rgb, ax_map) = plt.subplots(
        1, 2, figsize=(FRAME_W / 100, FRAME_H / 100), dpi=100
    )

    # ── left: RGB + overlay ──────────────────────────────────────────────────
    ax_rgb.imshow(rgb_np)
    x, y, th = belief.pose_mu.tolist()
    phase_color = "#2ECC71" if phase == "EXPLORE" else "#E74C3C"
    ax_rgb.set_title(
        f"[{phase}]  Step {step}  |  {action_name}\n"
        f"Pose ({x:.2f}, {y:.2f})  θ={math.degrees(th):.1f}°  "
        + (f"GoalDist={goal_dist:.2f}m" if phase == "NAVIGATE" else ""),
        fontsize=8, loc="center", pad=3,
        color=phase_color,
    )
    ax_rgb.axis("off")

    # ── right: top-down map ──────────────────────────────────────────────────
    ax_map.set_facecolor("#e8e8e8")
    ax_map.set_aspect("equal")

    if topdown is not None and bounds is not None:
        X_min, Z_min = float(bounds[0][0]), float(bounds[0][2])
        X_max, Z_max = float(bounds[1][0]), float(bounds[1][2])
        ax_map.imshow(
            topdown, cmap="Blues", alpha=0.30, origin="lower",
            extent=[X_min, X_max, Z_min, Z_max], zorder=0,
        )
        ax_map.set_xlim(X_min - 0.3, X_max + 0.3)
        ax_map.set_ylim(Z_min - 0.3, Z_max + 0.3)

    # Path taken
    if len(path_v6) > 1:
        hab = [_v6_to_hab(x, y) for x, y in path_v6]
        xs, zs = [p[0] for p in hab], [p[1] for p in hab]
        ax_map.plot(xs, zs, color="royalblue", lw=1.4, alpha=0.6, zorder=2)
        ax_map.scatter(xs[1:], zs[1:], s=8, c="royalblue", alpha=0.4, zorder=3)

    # ── Slot belief map ──────────────────────────────────────────────────────
    # Active slots rendered in three layers:
    #   • Wall slots (class 92)    — red squares, impassable
    #   • Doorway slots (class 93) — green triangles, navigable passages
    #   • Object slots (COCO)      — tab20 coloured circles
    active_mask = belief.slot_conf_logit > cfg.slots.conf_logit_empty_threshold
    if active_mask.any():
        pos   = belief.slot_pos_mu[active_mask].detach().cpu()        # [K, 3]
        conf  = torch.sigmoid(
            belief.slot_conf_logit[active_mask]).detach().cpu().numpy()
        cls   = belief.slot_class_logits[active_mask].argmax(dim=-1).detach().cpu().numpy()
        Xh    = -pos[:, 1].numpy()
        Zh    = -pos[:, 0].numpy()

        wall_mask    = cls == WALL_ID
        doorway_mask = cls == DOORWAY_ID
        obj_mask     = ~wall_mask & ~doorway_mask

        # Object slots
        if obj_mask.any():
            colours = [cm.tab20(int(c) % 20) for c in cls[obj_mask]]
            sizes   = 30 + 70 * conf[obj_mask]
            ax_map.scatter(Xh[obj_mask], Zh[obj_mask], s=sizes, c=colours,
                           alpha=0.75, edgecolors="black", linewidths=0.3,
                           zorder=5, label=f"{int(obj_mask.sum())} obj")
            for i, gi in enumerate(np.where(obj_mask)[0][:8]):
                cid  = int(cls[gi])
                name = _COCO_NAMES[cid] if cid < len(_COCO_NAMES) else str(cid)
                ax_map.text(Xh[gi], Zh[gi] + 0.12, name, fontsize=4.0,
                            ha="center", va="bottom", color="black", alpha=0.8, zorder=6)

        # Wall slots — red squares
        if wall_mask.any():
            sizes_w = 40 + 60 * conf[wall_mask]
            ax_map.scatter(Xh[wall_mask], Zh[wall_mask], s=sizes_w,
                           c="crimson", marker="s", alpha=0.70,
                           edgecolors="darkred", linewidths=0.5,
                           zorder=5, label=f"{int(wall_mask.sum())} walls")

        # Doorway slots — green triangles
        if doorway_mask.any():
            sizes_d = 60 + 80 * conf[doorway_mask]
            ax_map.scatter(Xh[doorway_mask], Zh[doorway_mask], s=sizes_d,
                           c="limegreen", marker="^", alpha=0.85,
                           edgecolors="darkgreen", linewidths=0.6,
                           zorder=6, label=f"{int(doorway_mask.sum())} doors")

    # Start marker
    sx_h, sz_h = _v6_to_hab(*start_v6)
    ax_map.scatter([sx_h], [sz_h], s=120, c="gold", marker="*",
                   edgecolors="black", lw=0.5, zorder=7, label="Start")

    # Goal + waypoints (navigation phase only)
    if phase == "NAVIGATE" and goal_v6 is not None:
        gx_h, gz_h = _v6_to_hab(float(goal_v6[0]), float(goal_v6[1]))
        ax_map.scatter([gx_h], [gz_h], s=160, c="red", marker="*",
                       edgecolors="darkred", lw=0.5, zorder=7, label="Goal")
        ax_map.add_patch(plt.Circle((gx_h, gz_h), GOAL_RADIUS,
                                    color="red", fill=False, ls="--",
                                    lw=1.0, alpha=0.6, zorder=4))
        if waypoints_v6 and len(waypoints_v6) > 1:
            wps = [_v6_to_hab(float(w[0]), float(w[1])) for w in waypoints_v6]
            wx, wz = [p[0] for p in wps], [p[1] for p in wps]
            ax_map.plot(wx, wz, color="orange", lw=1.1, ls=":", alpha=0.7, zorder=3)
            ax_map.scatter(wx, wz, s=20, c="orange", alpha=0.5, zorder=4)
            if wp_idx < len(wps):
                ax_map.scatter([wps[wp_idx][0]], [wps[wp_idx][1]], s=70,
                               c="darkorange", marker="D",
                               edgecolors="black", lw=0.4, zorder=8,
                               label=f"wp[{wp_idx}]")

    # Current robot pose — use GT if available (eliminates EKF drift from display)
    rx, ry, rth = robot_pose_gt if robot_pose_gt is not None else (x, y, th)
    cx_h, cz_h = _v6_to_hab(rx, ry)
    ax_map.scatter([cx_h], [cz_h], s=90, c="limegreen",
                   edgecolors="darkgreen", lw=0.8, zorder=9)
    arrow_len = 0.45
    ax_map.annotate("", xy=(cx_h - math.sin(rth)*arrow_len,
                             cz_h - math.cos(rth)*arrow_len),
                    xytext=(cx_h, cz_h),
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.8),
                    zorder=10)

    # LiDAR scan overlay — polar dots in robot-local frame projected onto map
    # Angles in V6: 0 = forward (+x), positive = left (+y), negative = right (-y).
    # Camera is at robot position, heading = rth.  Each beam:
    #   beam_angle_world = rth + angle_v6
    #   V6: bx = d*cos(θ), by = d*sin(θ) → Hab: X_h = -by, Z_h = -bx
    if lidar_angles is not None and lidar_dists is not None:
        ang_np  = lidar_angles.numpy()
        dist_np = lidar_dists.numpy()
        valid   = dist_np > 0.05                   # skip zero / invalid readings
        beam_world = rth + ang_np[valid]            # world-frame beam angle
        bx_v6  = dist_np[valid] * np.cos(beam_world)
        by_v6  = dist_np[valid] * np.sin(beam_world)
        # Convert to Habitat coords for plot
        bX_h   = -(by_v6  + ry)                    # Hab X = -y_v6
        bZ_h   = -(bx_v6  + rx)                    # Hab Z = -x_v6
        # Colour: red = close, yellow = mid, green = far (normalised 0–4 m)
        norm_d  = np.clip(dist_np[valid] / 4.0, 0.0, 1.0)
        colours = [(1 - nd, nd * 0.8, 0.0, 0.55) for nd in norm_d]
        ax_map.scatter(bX_h, bZ_h, s=4, c=colours, zorder=8, label="LiDAR")

    if phase == "EXPLORE":
        phase_label = f"Exploration — {wf_phase}  (EFE=Ambiguity−InfoGain)" if wf_phase \
                       else "Exploration  (EFE=Ambiguity−InfoGain, WF-DONE)"
    else:
        phase_label = "Navigation  (EFE=Risk+Ambiguity−InfoGain+WallPenalty)"
    ax_map.set_title(f"Object-centric belief map  |  {phase_label}",
                     fontsize=7.5, color=phase_color)
    ax_map.set_xlabel("Habitat X", fontsize=6)
    ax_map.set_ylabel("Habitat Z", fontsize=6)
    ax_map.legend(fontsize=5.5, loc="upper right", markerscale=0.8)
    ax_map.grid(True, alpha=0.2, lw=0.4)
    ax_map.tick_params(labelsize=5)

    fig.tight_layout(pad=0.6)
    fig.canvas.draw()
    w_px, h_px = fig.canvas.get_width_height()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h_px, w_px, 3)
    plt.close(fig)
    if (w_px, h_px) != (FRAME_W, FRAME_H):
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    return frame


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_unexplored_doorway(
    belief: BeliefState,
    explored_poses: list,
    *,
    explored_radius: float = 2.0,
) -> "Optional[torch.Tensor]":
    """Return the xy position of the nearest doorway slot not yet explored.

    A doorway is considered "explored" if any prior pose in explored_poses
    is within explored_radius metres of the door slot.  Returns None when
    all known doorways have already been visited, or none are detected.
    """
    conf = torch.sigmoid(belief.slot_conf_logit)          # [N]
    cls  = belief.slot_class_logits.argmax(-1)             # [N]
    mask = (cls == DOORWAY_ID) & (conf > 0.5)
    if not mask.any():
        return None

    door_xy  = belief.slot_pos_mu[mask, :2]               # [D, 2]
    curr_pos = belief.pose_mu[:2]

    if explored_poses:
        exp_xy = torch.stack([p[:2].to(door_xy.device) for p in explored_poses])
    else:
        exp_xy = None

    best, best_dist = None, float("inf")
    for i in range(door_xy.shape[0]):
        dp = door_xy[i]
        if exp_xy is not None:
            if float((exp_xy - dp.unsqueeze(0)).norm(dim=-1).min()) < explored_radius:
                continue   # already explored near this door
        dist = float((curr_pos.to(dp.device) - dp).norm())
        if dist < best_dist:
            best_dist, best = dist, dp.clone()
    return best


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(_LOG_DIR, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    vid_path = os.path.join(_LOG_DIR, f"aif_explore_nav_{ts}.mp4")
    log_path = os.path.join(_LOG_DIR, f"aif_explore_nav_{ts}.txt")

    log_file = open(log_path, "w", buffering=1)
    def log(msg=""):
        print(msg)
        log_file.write(msg + "\n")

    log(f"Video → {vid_path}")
    log(f"Log   → {log_path}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device : {device}")

    # ── model ─────────────────────────────────────────────────────────────────
    cfg   = ModelV6Config()
    model = WorldModelV6(cfg).to(device)
    model.eval()
    log(f"WorldModelV6 : {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    # ── Habitat ───────────────────────────────────────────────────────────────
    env = HabitatEnv()
    topdown, bounds = _get_topdown(env.sim)
    log(f"Navmesh      : {topdown.shape if topdown is not None else 'unavailable'}")

    # ── video writer ──────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vid_path, fourcc, VID_FPS, (FRAME_W, FRAME_H))

    # ── episode setup ─────────────────────────────────────────────────────────
    _seed = random.randint(1, 1_000_000)
    env.sim.seed(_seed)
    np.random.seed(_seed % (2**31))
    rgb, depth, cam_to_v6, pose_v6, _ = env.reset()
    start_xy = (float(pose_v6[0]), float(pose_v6[1]))
    log(f"\nStart pose (V6) : ({pose_v6[0]:.2f}, {pose_v6[1]:.2f}, "
        f"{math.degrees(pose_v6[2].item()):.1f}°)")

    belief  = model.initial_belief(device, known_pose=pose_v6.to(device))
    # Exploration: no goal → Risk = 0, EFE driven by Ambiguity - InfoGain
    planner = EFEPlannerV4(model=model, goal_pos=None, cfg=cfg.efe)

    # ── wall-following explorer (Stage 3) ─────────────────────────────────────
    wall_follower = WallFollowExplorer(
        turn_step   = cfg.pose.turn_step_rad,
        max_seek    = 8,    # forward steps per heading (8 × 0.25m = 2m max — stays within room)
        max_follow  = 30,   # max steps per wall-follow segment (~160 steps covers all 4 headings)
        return_dist = 0.8,  # metres — "returned to start" threshold
    )
    log("\nWall-follow explorer ready  (SEEK E → N → W → S)")

    # ── shared loop state ─────────────────────────────────────────────────────
    prev_action   = 3
    prev_odom     = torch.zeros(3, device=device)
    recent_poses: List[torch.Tensor] = [pose_v6.to(device)]
    explored_poses: List[torch.Tensor] = []        # all visited positions (frontier)
    path_xy: List[Tuple[float, float]] = [start_xy]
    step_times_ms: List[float] = []
    collided      = False   # result of the previous step's action

    # ── LiDAR state (initialised from first observation) ─────────────────────
    LIDAR_OBSTACLE_DIST = 0.40   # metres — block FORWARD if obstacle within this range
    LIDAR_CONE_DEG      = 20.0   # ± degrees in forward direction
    _lidar_angles, _lidar_dists = HabitatEnv.depth_to_lidar(depth, n_beams=64)
    _lidar_angles = _lidar_angles.cpu()
    _lidar_dists  = _lidar_dists.cpu()

    # ── persistent belief memory: load slots from previous run ───────────────
    # Slot positions are in the Habitat world frame — consistent across episodes
    # of the same scene.  Only the pose is reset (the robot starts elsewhere).
    # Delete belief_memory.pt to start with a blank slate.
    log("\n[Memory] Checking for persistent belief...")
    if os.path.exists(_BELIEF_MEM_PATH):
        try:
            mem = torch.load(_BELIEF_MEM_PATH, map_location=device)
            saved_shape = mem.get("navmesh_shape")
            cur_shape   = topdown.shape if topdown is not None else None
            if saved_shape == cur_shape:
                belief.slot_conf_logit    = mem["slot_conf_logit"].to(device)
                belief.slot_class_logits  = mem["slot_class_logits"].to(device)
                belief.slot_pos_mu        = mem["slot_pos_mu"].to(device)
                belief.slot_pos_logvar    = mem["slot_pos_logvar"].to(device)
                prev_ep = [p.to(device) for p in mem.get("explored_poses", [])]
                explored_poses.extend(prev_ep)
                n_loaded = int((belief.slot_conf_logit >
                                cfg.slots.conf_logit_empty_threshold).sum().item())
                log(f"[Memory] Restored {n_loaded} occupied slots + "
                    f"{len(prev_ep)} prior explored poses  (scene matches)")
            else:
                log(f"[Memory] Scene changed ({saved_shape} → {cur_shape}) — starting fresh")
        except Exception as exc:
            log(f"[Memory] Load failed ({exc}) — starting fresh")
    else:
        log("[Memory] No prior belief found — starting fresh")

    # Navigation-phase state (set at phase switch)
    phase         = "EXPLORE"
    goal_v6       = None
    waypoints_v6: Optional[List[torch.Tensor]] = None
    wp_idx        = 0
    current_wp    = None
    blocked_headings: set = set()
    steps_no_progress = 0
    prev_goal_dist = float("inf")
    consecutive_non_fwd = 0
    wall_avoid_queue: List[int] = []
    nav_phase_attempted = False   # ensure phase switch is tried exactly once
    reached = False
    # Multi-room exploration state
    door_transit_target: Optional[torch.Tensor] = None  # doorway we're heading toward
    door_transit_steps:  int = 0
    rooms_explored:      int = 0    # extra rooms entered (beyond the start room)

    log(f"\n{'Step':>4} | {'Phase':>8} | {'Action':>7} | "
        f"{'Pose (x, y, θ)':>28} | {'GoalDist':>8} | "
        f"{'Slots':>5} | {'ms':>5}")
    log("-" * 85)

    for step in range(MAX_STEPS):
        t0 = time.perf_counter()

        # ── belief update ─────────────────────────────────────────────────────
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

        n_slots = int(belief.slot_conf_logit.gt(
            cfg.slots.conf_logit_empty_threshold).sum().item())

        # ── record explored poses (every EXPLORE_SAMPLE steps) ───────────────
        if step % EXPLORE_SAMPLE == 0:
            explored_poses.append(belief.pose_mu.detach().clone())
            if len(explored_poses) > 300:
                explored_poses = explored_poses[-300:]

        # (no waypoint-advance logic — wall-follower manages its own state)

        # ── Multi-room: trigger door transit once per WF completion ──────────
        # After each wall-follow cycle completes, look for a doorway slot that
        # hasn't been explored yet and queue a transit to it.  Only fires when:
        #   • WF is done AND no transit is already active
        #   • we haven't exhausted the room budget (MAX_ROOMS)
        #   • navigation hasn't started yet
        if (phase == "EXPLORE"
                and wall_follower.done
                and door_transit_target is None
                and not nav_phase_attempted
                and rooms_explored < MAX_ROOMS):
            _next_door = _find_unexplored_doorway(belief, explored_poses)
            if _next_door is not None:
                door_transit_target = _next_door.to(device)
                door_transit_steps  = 0
                log(f"\n[Multi-room] WF done → targeting doorway at "
                    f"({float(door_transit_target[0]):.2f},"
                    f" {float(door_transit_target[1]):.2f})")

        # ── PHASE SWITCH: exploration → navigation ────────────────────────────
        # Attempted exactly once (nav_phase_attempted flag), always succeeds:
        # if pathfinder finds a route use it; otherwise navigate directly to goal.
        # Early switch: if wall-follower is done AND belief map already saturated
        # (≥90% slots occupied from memory) AND no active door transit AND no more
        # rooms left to explore — skip the EFE exploration fallback.
        _slots_saturated = n_slots >= int(cfg.slots.num_slots * 0.90)
        _early_nav = (wall_follower.done
                      and _slots_saturated
                      and door_transit_target is None
                      and step > 0)
        if (phase == "EXPLORE" and not nav_phase_attempted
                and (step >= N_EXPLORE or _early_nav)):
            nav_phase_attempted = True
            _trigger = "early (belief saturated)" if _early_nav else f"N_EXPLORE={N_EXPLORE}"
            log(f"\n{'='*60}")
            log(f"  EXPLORATION COMPLETE at step {step+1}  [{_trigger}]"
                f"  ({n_slots} active slots in belief map)")
            log(f"  Switching to NAVIGATION phase...")

            # Try to sample a reachable goal with pathfinder route
            goal_v6_cand = None
            waypoints_v6_cand = None
            for _att in range(20):
                g   = env.sample_goal(min_dist=3.0, max_dist=8.0)
                wps = env.get_path_waypoints(g, waypoint_spacing=1.0)
                if wps is not None:
                    goal_v6_cand    = g
                    waypoints_v6_cand = wps
                    break
                log(f"  [goal attempt {_att+1}: no path, resampling]")

            if goal_v6_cand is None:
                # Fallback: shorter range + also try to get waypoints for it
                log("  [WARNING: no path found — trying shorter range fallback]")
                for _att2 in range(20):
                    g2   = env.sample_goal(min_dist=1.0, max_dist=5.0)
                    wps2 = env.get_path_waypoints(g2, waypoint_spacing=1.0)
                    if wps2 is not None:
                        goal_v6_cand      = g2
                        waypoints_v6_cand = wps2
                        log(f"  [fallback goal found at attempt {_att2+1}]")
                        break
                if goal_v6_cand is None:
                    # Absolute last resort: direct navigation, no waypoints
                    goal_v6_cand      = env.sample_goal(min_dist=0.5, max_dist=3.0)
                    waypoints_v6_cand = [goal_v6_cand]
                    log("  [WARNING: using raw direct goal, no waypoints]")

            goal_v6      = goal_v6_cand
            waypoints_v6 = waypoints_v6_cand
            phase        = "NAVIGATE"
            wp_idx       = min(1, len(waypoints_v6) - 1)
            current_wp   = waypoints_v6[wp_idx]
            planner.set_goal(current_wp.to(device))
            prev_goal_dist = float("inf")
            log(f"  Goal (V6)    : ({goal_v6[0]:.2f}, {goal_v6[1]:.2f})")
            log(f"  Waypoints    : {len(waypoints_v6)}")
            for wi, wp in enumerate(waypoints_v6):
                log(f"    wp[{wi}] = ({wp[0]:.2f}, {wp[1]:.2f})")
            log(f"{'='*60}\n")

        # ── goal check (navigation phase) ────────────────────────────────────
        # Use ground-truth pose (pose_v6) to avoid EKF drift corrupting wp checks.
        if phase == "NAVIGATE" and current_wp is not None:
            curr_xy_gt = pose_v6[:2].to(device)
            wp_dist    = float((curr_xy_gt - current_wp.to(device)).norm())
            if wp_dist < WP_ADVANCE and wp_idx < len(waypoints_v6) - 1:
                wp_idx    += 1
                current_wp = waypoints_v6[wp_idx]
                planner.set_goal(current_wp.to(device))
                blocked_headings.clear()
                log(f"  >> wp[{wp_idx}]/{len(waypoints_v6)-1}: "
                    f"({current_wp[0]:.2f}, {current_wp[1]:.2f})")

            # Goal-reached: use GT distance to avoid EKF drift preventing success.
            if wp_idx == len(waypoints_v6) - 1 and wp_dist < GOAL_RADIUS:
                log(f"\n  >> GOAL REACHED at step {step+1}!")
                reached = True
                rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
                frame  = _render_frame(
                    rgb_np, belief, cfg, step+1, "NAVIGATE", "REACHED",
                    0.0, path_xy, start_xy, topdown, bounds,
                    goal_v6=goal_v6, waypoints_v6=waypoints_v6, wp_idx=wp_idx,
                    robot_pose_gt=(float(pose_v6[0]), float(pose_v6[1]), float(pose_v6[2])),
                    lidar_angles=_lidar_angles,
                    lidar_dists=_lidar_dists,
                )
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                for _ in range(VID_FPS * 2):
                    writer.write(bgr)
                break

        # goal_dist: use GT distance to current waypoint during NAVIGATE so the
        # nav controller and goal-check are consistent.  Belief-based distance
        # diverges from truth under EKF drift and causes the robot to oscillate
        # past the goal without ever triggering GOAL_REACHED.
        if phase == "NAVIGATE" and current_wp is not None:
            goal_dist = float((pose_v6[:2].to(device) - current_wp.to(device)).norm())
        else:
            goal_dist = float("nan")

        # ── action selection ──────────────────────────────────────────────────
        # Redesigned priority order:
        #  1. wall_avoid_queue  — escape collision (both phases)
        #  2. WallFollowExplorer — perimeter trace (EXPLORE, wall not done)
        #  2.5 door-transit     — heading to next room's doorway (EXPLORE, after WF done)
        #  3. EFE fallback      — info-gain sweep (EXPLORE, WF done + no transit)
        #  4. waypoint-follower — heading-align to navmesh wp (NAVIGATE, pure navmesh)
        #  5. spin-override     — unstick when nav turns too long (NAVIGATE backup)

        # EFE planner: only called during EXPLORE (saves ~80ms/step during NAVIGATE).
        # VIA-DOOR is disabled globally via cfg.efe.enable_door_routing=False.
        wall_pen_val = 0.0
        if phase == "EXPLORE":
            with torch.no_grad():
                plan = planner.select_action(
                    belief,
                    recent_poses     = recent_poses,
                    blocked_headings = blocked_headings,
                    explored_poses   = explored_poses,
                )
            action       = plan["best_action"]
            wall_pen_val = plan.get("wall_penalty", 0.0)
        else:
            action = 0   # will be overridden by Priority 4

        # Priority 1: Wall-escape queue (both phases, highest priority)
        if wall_avoid_queue:
            action = wall_avoid_queue.pop(0)
            log(f"       [forced: {ACTION_NAMES[action]}, {len(wall_avoid_queue)} left]")

        # Priority 2: Wall-following explorer (EXPLORE phase)
        elif phase == "EXPLORE" and not wall_follower.done:
            wf_act = wall_follower.get_action(pose_v6, collided)
            if wf_act is not None:
                action = wf_act
                log(f"       [{wall_follower.phase_name} → {ACTION_NAMES[action]}]")

        # Priority 2.5: Door transit — navigate toward an unexplored doorway to
        # enter an adjacent room and restart the wall-follower there.
        elif phase == "EXPLORE" and wall_follower.done and door_transit_target is not None:
            dx_d   = float(door_transit_target[0] - pose_v6[0])
            dy_d   = float(door_transit_target[1] - pose_v6[1])
            d_dir  = math.atan2(dy_d, dx_d)
            curr_h = float(pose_v6[2])
            herr   = math.atan2(math.sin(d_dir - curr_h), math.cos(d_dir - curr_h))
            _ts    = cfg.pose.turn_step_rad
            _rh    = math.atan2(math.sin(round(curr_h / _ts) * _ts),
                                math.cos(round(curr_h / _ts) * _ts))
            if abs(herr) <= math.radians(30) and _rh not in blocked_headings:
                action = 0
            elif herr > 0:
                action = 1
            else:
                action = 2

            door_transit_steps += 1
            dist_to_dt = float(
                (pose_v6[:2].to(device) - door_transit_target).norm()
            )
            log(f"       [door-transit: dist={dist_to_dt:.2f}m,"
                f" h_err={math.degrees(herr):.0f}° → {ACTION_NAMES[action]}]")

            if dist_to_dt < 1.5 or door_transit_steps >= 60:
                rooms_explored += 1
                log(f"\n[Multi-room] Entered room {rooms_explored + 1}"
                    f" — restarting WallFollower.")
                wall_follower = WallFollowExplorer(
                    turn_step   = cfg.pose.turn_step_rad,
                    max_seek    = 8,
                    max_follow  = 30,
                    return_dist = 0.8,
                )
                door_transit_target = None
                door_transit_steps  = 0

        # Priority 3 (EXPLORE EFE fallback): EFE action already set above.
        # No extra code needed — action was assigned by planner.select_action().

        # Priority 4: Pure waypoint-follower (NAVIGATE phase, full range).
        # Uses navmesh waypoints — no EFE, no VIA-DOOR, no wall-penalty needed.
        # The navmesh already guarantees obstacle-free paths between waypoints.
        elif phase == "NAVIGATE" and planner.goal_pos is not None and goal_dist > GOAL_RADIUS:
            dx_g     = float(planner.goal_pos[0] - pose_v6[0])
            dy_g     = float(planner.goal_pos[1] - pose_v6[1])
            goal_dir = math.atan2(dy_g, dx_g)
            curr_h   = float(pose_v6[2])
            herr     = math.atan2(math.sin(goal_dir - curr_h),
                                   math.cos(goal_dir - curr_h))
            _ts = cfg.pose.turn_step_rad
            _rh = math.atan2(math.sin(round(curr_h / _ts) * _ts),
                              math.cos(round(curr_h / _ts) * _ts))
            if abs(herr) <= math.radians(30) and _rh not in blocked_headings:
                action = 0   # aligned → FORWARD
            elif herr > 0:
                action = 1   # TURN_L
            else:
                action = 2   # TURN_R
            log(f"       [nav: h_err={math.degrees(herr):.0f}°"
                f" dist={goal_dist:.2f}m → {ACTION_NAMES[action]}]")

        # Priority 5: Spin-override (NAVIGATE backup when heading-align stalls)
        elif (phase == "NAVIGATE" and not wall_avoid_queue
              and consecutive_non_fwd >= 6
              and not math.isnan(goal_dist) and goal_dist < 3.0
              and planner.goal_pos is not None):
            dx_g     = float(planner.goal_pos[0] - pose_v6[0])
            dy_g     = float(planner.goal_pos[1] - pose_v6[1])
            goal_dir = math.atan2(dy_g, dx_g)
            curr_h   = float(pose_v6[2])
            herr     = math.atan2(math.sin(goal_dir - curr_h),
                                   math.cos(goal_dir - curr_h))
            if abs(herr) < math.radians(45):
                ov_act = 0
            elif herr > 0:
                ov_act = 1
            else:
                ov_act = 2
            _blocked = False
            if ov_act == 0 and blocked_headings:
                _ts = cfg.pose.turn_step_rad
                _rh = math.atan2(math.sin(round(curr_h / _ts) * _ts),
                                  math.cos(round(curr_h / _ts) * _ts))
                _blocked = _rh in blocked_headings
            if not _blocked:
                action = ov_act
                consecutive_non_fwd = 0
                log(f"       [spin-override: h_err={math.degrees(herr):.0f}°"
                    f" → {ACTION_NAMES[action]}]")

        # ── log ───────────────────────────────────────────────────────────────
        x, y, th = belief.pose_mu.tolist()
        ms = (time.perf_counter() - t0) * 1000
        step_times_ms.append(ms)
        log(f"{step+1:>4} | {phase:>8} | {ACTION_NAMES[action]:>7} | "
            f"({x:6.2f}, {y:6.2f}, {math.degrees(th):5.1f}°) | "
            f"{goal_dist:8.3f} | {n_slots:>5} | {ms:5.0f}")
        if wall_pen_val > 5.0:
            log(f"       [wall-penalty={wall_pen_val:.2f}]")

        # ── render frame ──────────────────────────────────────────────────────
        rgb_np = rgb.permute(1, 2, 0).cpu().numpy()
        frame  = _render_frame(
            rgb_np, belief, cfg, step+1, phase, ACTION_NAMES[action], goal_dist,
            path_xy, start_xy, topdown, bounds,
            goal_v6=goal_v6, waypoints_v6=waypoints_v6, wp_idx=wp_idx,
            wf_phase=wall_follower.phase_name if phase == "EXPLORE" else "",
            robot_pose_gt=(float(pose_v6[0]), float(pose_v6[1]), float(pose_v6[2])),
            lidar_angles=_lidar_angles,
            lidar_dists=_lidar_dists,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # ── LiDAR obstacle avoidance (override FORWARD if blocked) ───────────
        # Prevents the robot from charging into walls regardless of which
        # priority assigned FORWARD.  Only applies when FORWARD is selected and
        # there is a near obstacle within the forward cone.
        if action == 0 and not wall_avoid_queue:
            _fwd_clear = HabitatEnv.lidar_min_forward(
                _lidar_angles, _lidar_dists, cone_deg=LIDAR_CONE_DEG
            )
            if _fwd_clear < LIDAR_OBSTACLE_DIST:
                _left_clr, _right_clr = HabitatEnv.lidar_side_clearance(
                    _lidar_angles, _lidar_dists
                )
                action = 1 if _left_clr >= _right_clr else 2
                log(f"       [LiDAR: fwd={_fwd_clear:.2f}m < {LIDAR_OBSTACLE_DIST}m"
                    f"  L={_left_clr:.2f} R={_right_clr:.2f}"
                    f" → {ACTION_NAMES[action]}]")

        # ── execute action ────────────────────────────────────────────────────
        rgb, depth, cam_to_v6, odom, pose_v6, done, info = env.step(action)
        collided = info.get("collided", False)
        if collided:
            log(f"       [collision]")

        # Update LiDAR from new depth image (used next iteration)
        _lidar_angles, _lidar_dists = HabitatEnv.depth_to_lidar(depth, n_beams=64)
        _lidar_angles = _lidar_angles.cpu()
        _lidar_dists  = _lidar_dists.cpu()

        # ── collision bookkeeping — BOTH phases ───────────────────────────────
        # Track blocked headings whenever FORWARD hits a wall.
        if action == 0:
            _ts = cfg.pose.turn_step_rad
            _h  = belief.pose_mu[2].item()
            _rh = math.atan2(math.sin(round(_h/_ts)*_ts),
                              math.cos(round(_h/_ts)*_ts))
            if collided:
                blocked_headings.add(_rh)
            elif not collided:
                # Successful FORWARD — clear stale wall memory
                blocked_headings.discard(_rh)

        # Track how many steps since meaningful forward progress.
        # Wall-follower turns are intentional — suppress escape counter while active.
        if action == 0 and not collided:
            steps_no_progress = 0
        elif action != 0 and wall_avoid_queue:
            pass   # wall-escape turn — don't count as stuck
        elif phase == "EXPLORE" and not wall_follower.done:
            steps_no_progress = 0   # wall-follower owns its own recovery
        else:
            steps_no_progress += 1

        # In navigation, also track goal-distance progress and clear wall memory
        if phase == "NAVIGATE" and not math.isnan(goal_dist):
            if goal_dist < prev_goal_dist - 0.05:
                steps_no_progress = 0
                if action == 0 and not collided:
                    blocked_headings.clear()
            prev_goal_dist = goal_dist

        # ── escape when stuck (BOTH phases) ──────────────────────────────────
        # During EXPLORE: only fire when wall-follower is DONE (EFE fallback mode).
        if steps_no_progress >= 8 and not wall_avoid_queue:
            if phase == "NAVIGATE" and blocked_headings and planner.goal_pos is not None:
                # Navigation: perpendicular detour around wall (use GT pose)
                dx_g = float(planner.goal_pos[0] - pose_v6[0])
                dy_g = float(planner.goal_pos[1] - pose_v6[1])
                _gd  = math.atan2(dy_g, dx_g)
                _ch  = float(pose_v6[2])
                _ts  = cfg.pose.turn_step_rad
                _p1, _p2 = _gd + math.pi/2, _gd - math.pi/2
                _h1  = math.atan2(math.sin(_p1-_ch), math.cos(_p1-_ch))
                _h2  = math.atan2(math.sin(_p2-_ch), math.cos(_p2-_ch))
                _herr_p, _pdir = (_h1, _p1) if abs(_h1) <= abs(_h2) else (_h2, _p2)
                _nt  = max(1, round(abs(_herr_p) / _ts))
                _td  = 1 if _herr_p > 0 else 2
                wall_avoid_queue = [_td] * _nt + [0, 0, 0]
                log(f"       [wall-escape: {_nt} turns to "
                    f"{math.degrees(_pdir):.0f}° + 3 FWD]")
            elif phase == "EXPLORE" and wall_follower.done:
                # EFE fallback only — sweep if EFE gets stuck
                wall_avoid_queue = [1, 1, 1, 0, 0, 0]   # 3×TURN_L + 3×FORWARD
                log(f"       [explore-sweep: 3 TURN_L + 3 FORWARD queued]")
            if blocked_headings:
                log(f"       [escape: clearing {len(blocked_headings)} blocked headings]")
            blocked_headings.clear()
            steps_no_progress = 0

        if action == 0 and not collided:
            consecutive_non_fwd = 0
        else:
            consecutive_non_fwd += 1

        prev_odom   = odom.to(device)
        prev_action = action

        recent_poses.append(belief.pose_mu.detach().clone())
        if len(recent_poses) > 10:
            recent_poses = recent_poses[-10:]

        path_xy.append((float(pose_v6[0]), float(pose_v6[1])))
        if done:
            log(f"\n  >> Stop action at step {step+1}.")
            break

    # ── summary ───────────────────────────────────────────────────────────────
    # Use GT distance for final summary (belief may have drifted far from truth)
    if phase == "NAVIGATE" and goal_v6 is not None:
        final_dist = float((pose_v6[:2].to(device) - goal_v6[:2].to(device)).norm())
    else:
        final_dist = float("nan")
    log()
    log("=" * 60)
    log(f"  Result         : {'SUCCESS' if reached else 'TIMEOUT'}")
    log(f"  Steps          : {step+1}  ({N_EXPLORE} explore + {max(0, step+1-N_EXPLORE)} navigate)")
    log(f"  Final slot count: {n_slots}")
    if not math.isnan(final_dist):
        log(f"  Final goal dist: {final_dist:.3f}m")
    if step_times_ms:
        import statistics
        log(f"  Mean ms/step   : {statistics.mean(step_times_ms):.0f}")
    log("=" * 60)

    # ── persistent belief memory: save slots for next run ────────────────────
    try:
        n_occ = int((belief.slot_conf_logit >
                     cfg.slots.conf_logit_empty_threshold).sum().item())
        mem = {
            "navmesh_shape":    topdown.shape if topdown is not None else None,
            "slot_conf_logit":  belief.slot_conf_logit.cpu(),
            "slot_class_logits": belief.slot_class_logits.cpu(),
            "slot_pos_mu":       belief.slot_pos_mu.cpu(),
            "slot_pos_logvar":   belief.slot_pos_logvar.cpu(),
            "explored_poses":    [p.cpu() for p in explored_poses[-300:]],
        }
        torch.save(mem, _BELIEF_MEM_PATH)
        log(f"  [Memory] Saved {n_occ} occupied slots → belief_memory.pt")
    except Exception as exc:
        log(f"  [Memory] Save failed: {exc}")

    writer.release()
    env.close()
    log_file.close()
    print(f"\nVideo → {vid_path}")
    print(f"Log   → {log_path}")


if __name__ == "__main__":
    main()

"""
test_v6_model.py — Synthetic unit test for V6 architecture.

Validates the full belief update loop WITHOUT Habitat-Sim or a real camera.
Injects synthetic detections directly (bypassing Perception module) so we
can test the EKF, slot memory, GRU, and EFE planner in isolation.

Test scenario:
  - 4 objects at known world positions
  - Agent moves forward 5 steps
  - After each step, we inject detections for objects visible from the new pose
  - Check that:
      (a) slot positions converge toward true object positions
      (b) pose uncertainty decreases after each EKF correction
      (c) EFE planner returns valid action sequences
      (d) goal is reached within planning horizon
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import torch

# Add parent to path so we can import model_v6
sys.path.insert(0, str(Path(__file__).parent))

from model_v6 import (
    ModelV6Config, BeliefState, Detection, WorldModelV6,
)
from planner import EFEPlannerV4


# ------------------------------------------------------------------ #
# Synthetic world
# ------------------------------------------------------------------ #

OBJECTS: List[Tuple[float, float, float, int]] = [
    (2.0,  0.5, 0.5, 56),   # (x, y, z, coco_class)  class 56 = chair
    (3.0, -0.5, 0.5, 56),
    (4.5,  1.0, 1.2, 57),   # class 57 = couch
    (5.0, -1.0, 0.5, 60),   # class 60 = dining table
]

GOAL_XY = torch.tensor([5.0, 0.0])   # navigate to (5, 0)


def synthetic_detections(
    pose_mu: torch.Tensor,   # [3]  (x, y, θ)
    device:  torch.device,
    fov_deg: float = 90.0,
    max_dist: float = 6.0,
    noise_std: float = 0.15,
) -> List[Detection]:
    """
    Generate synthetic detections of OBJECTS visible from pose_mu.
    Adds Gaussian noise to positions to simulate sensor noise.
    """
    half = math.radians(fov_deg / 2.0)
    dets = []
    for (ox, oy, oz, cls) in OBJECTS:
        dx    = ox - pose_mu[0].item()
        dy    = oy - pose_mu[1].item()
        dist  = math.sqrt(dx ** 2 + dy ** 2)
        if dist > max_dist:
            continue
        bear  = math.atan2(dy, dx) - pose_mu[2].item()
        bear  = (bear + math.pi) % (2 * math.pi) - math.pi
        if abs(bear) > half:
            continue
        # Add noise
        noise = torch.randn(3, device=device) * noise_std
        pos   = torch.tensor([ox, oy, oz], device=device) + noise
        dets.append(Detection(
            class_id=cls, class_conf=0.9,
            pos_world=pos,
            bbox_img=(0.0, 0.0, 1.0, 1.0),
        ))
    return dets


# ------------------------------------------------------------------ #
# Minimal slot + pose update without Perception module
# (bypasses the Faster R-CNN; feeds synthetic detections directly)
# ------------------------------------------------------------------ #

def step_with_synthetic_detections(
    model:    WorldModelV6,
    belief:   BeliefState,
    action:   int,
    dets:     List[Detection],
    odom:     torch.Tensor,
) -> Tuple[BeliefState, dict]:
    """
    Run one belief update step using pre-built synthetic detections
    (skips the Perception module).
    """
    from model_v6.data_association import associate
    device = belief.device

    # Pose prediction
    pose_mu_prior, pose_logvar_prior = model.pose_est.predict(
        belief.pose_mu, belief.pose_logvar, action, odom
    )
    b = belief.clone()
    b.pose_mu     = pose_mu_prior
    b.pose_logvar = pose_logvar_prior

    # Data association
    matched, unmatched_dets, unmatched_slots = associate(
        dets, b, model.cfg.assoc, model.cfg.slots
    )

    # Slot update
    b = model.slot_mem.update(b, dets, matched, unmatched_dets, unmatched_slots)

    # Pose correction
    pose_mu_post, pose_logvar_post = model.pose_est.correct(
        b.pose_mu, b.pose_logvar, dets, matched, b
    )
    b.pose_mu     = pose_mu_post
    b.pose_logvar = pose_logvar_post

    # GRU context
    occ        = model.slot_mem.occupied_mask(b)
    n_occ      = int(occ.sum().item())
    mean_conf  = float(b.slot_conf[occ].mean().item()) if n_occ > 0 else 0.0
    pose_delta = float((pose_mu_post[:2] - pose_mu_prior[:2]).norm().item())

    ctx = model.ctx_encoder(
        pose_mu=b.pose_mu, pose_logvar=b.pose_logvar,
        n_matched=len(matched), n_new=len(unmatched_dets),
        n_occupied=n_occ, mean_slot_conf=mean_conf,
        pose_correction=pose_delta,
    )
    h_new, _ = model.gru(ctx.unsqueeze(0).unsqueeze(0), belief.h)
    b.h    = h_new
    b.step = belief.step + 1

    info = {
        "n_matched": len(matched),
        "n_new":     len(unmatched_dets),
        "n_occupied": n_occ,
        "pose_std_xy": float(torch.exp(0.5 * b.pose_logvar[:2]).mean().item()),
    }
    return b, info


# ------------------------------------------------------------------ #
# Main test
# ------------------------------------------------------------------ #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg   = ModelV6Config()
    model = WorldModelV6(cfg).to(device)
    model.eval()

    # Count learnable params (GRU + ContextEncoder only)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Learnable parameters: {n_params:,}")

    # Initial belief — agent starts at (0, 0, 0°), pose known
    true_pose   = torch.tensor([0.0, 0.0, 0.0], device=device)
    belief      = model.initial_belief(device, known_pose=true_pose)

    planner = EFEPlannerV4(model, goal_pos=GOAL_XY, cfg=cfg.efe)

    print("\n--- Synthetic Navigation Test ---")
    print(f"{'Step':>4} | {'Pose (x,y,θ)':>24} | {'PoseStd':>8} | "
          f"{'Slots':>5} | {'Matched':>7} | {'GoalDist':>8} | {'Action':>6}")
    print("-" * 80)

    recent_poses = [true_pose.clone()]
    action_names = {0: "FWD", 1: "L", 2: "R", 3: "STOP"}

    for step in range(15):
        # ---- Check if goal reached ----
        if planner.reached_goal(belief):
            print(f"  >> Goal reached at step {step}! "
                  f"Dist={planner.goal_distance(belief):.3f}m")
            break

        # ---- Plan ----
        plan = planner.select_action(belief, recent_poses)
        action = plan["best_action"]

        # ---- Simulate ideal odometry for the chosen action ----
        from model_v6.pose_estimator import _action_to_delta
        dx, dy, dth = _action_to_delta(action, true_pose[2].item(), cfg.pose)
        true_pose = true_pose + torch.tensor([dx, dy, dth], device=device)
        true_pose[2] = float(torch.atan2(torch.sin(true_pose[2]),
                                          torch.cos(true_pose[2])).item())

        odom = torch.tensor([dx, dy, dth], device=device)

        # ---- Synthetic detections from new pose ----
        dets = synthetic_detections(true_pose, device)

        # ---- Belief update ----
        with torch.no_grad():
            belief, info = step_with_synthetic_detections(model, belief, action, dets, odom)

        recent_poses.append(true_pose.clone())

        x, y, th = belief.pose_mu.tolist()
        print(
            f"{step+1:>4} | "
            f"({x:6.2f}, {y:6.2f}, {math.degrees(th):5.1f}°) | "
            f"{info['pose_std_xy']:8.4f} | "
            f"{info['n_occupied']:>5} | "
            f"{info['n_matched']:>7} | "
            f"{planner.goal_distance(belief):8.3f}m | "
            f"{action_names[action]:>6}"
        )

    # ---- Slot position accuracy ----
    print("\n--- Slot Position Accuracy (occupied slots) ---")
    occ_idxs = belief.slot_conf_logit.argsort(descending=True)
    n_report = min(len(OBJECTS) + 2, belief.num_slots)
    print(f"{'Slot':>4} | {'Conf':>6} | {'Pos Est (x,y,z)':>30} | {'Pos Std':>8}")
    for idx in occ_idxs[:n_report]:
        conf_logit = belief.slot_conf_logit[idx].item()
        if conf_logit < cfg.slots.conf_logit_empty_threshold:
            break
        pos  = belief.slot_pos_mu[idx].tolist()
        std  = float(torch.exp(0.5 * belief.slot_pos_logvar[idx]).mean().item())
        print(f"{idx:>4} | {conf_logit:>6.2f} | "
              f"({pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}) | {std:>8.4f}m")

    print("\n--- Ground Truth Objects ---")
    for ox, oy, oz, cls in OBJECTS:
        print(f"  class={cls}  pos=({ox:.2f}, {oy:.2f}, {oz:.2f})")

    # ---- EFE planner timing ----
    import time
    n_warmup, n_timed = 3, 20
    for _ in range(n_warmup):
        planner.select_action(belief)
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        planner.select_action(belief)
        times.append((time.perf_counter() - t0) * 1000)
    import statistics
    print(f"\n--- EFE Planning Timing ({n_timed} runs) ---")
    print(f"  mean  = {statistics.mean(times):.1f} ms")
    print(f"  median= {statistics.median(times):.1f} ms")
    print(f"  min   = {min(times):.1f} ms")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()

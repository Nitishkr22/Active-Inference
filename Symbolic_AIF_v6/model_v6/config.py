"""
config.py — All hyperparameters for WorldModelV6.

Design philosophy (V6.0):
  - Analytical-first: EKF belief update, no neural training required
  - Pluggable perception: Faster R-CNN by default, swap in YOLOv8 later
  - Continuous pose: (x, y, θ) Gaussian, not discrete grid
  - Object-centric map: N=64 slots, each = class + 3D position + confidence
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class PerceptionConfig:
    # Image dimensions fed to detector
    img_h: int = 256
    img_w: int = 256

    # Camera intrinsics (Habitat default: HFOV=90°, 256×256)
    # fx = img_w / (2 * tan(HFOV/2)) = 256 / 2 = 128
    fx: float = 128.0
    fy: float = 128.0
    cx: float = 128.0   # img_w / 2
    cy: float = 128.0   # img_h / 2

    # Depth clip (metres)
    depth_min: float = 0.1
    depth_max: float = 8.0

    # Object detector
    detector_backbone: str = "fasterrcnn"   # "fasterrcnn" | "yolov8"
    detector_conf_threshold: float = 0.40
    detector_max_det: int = 20              # cap per frame

    # COCO class ids to keep (indoor-navigation relevant subset)
    # 0-indexed as returned by torchvision COCO detector.
    # None = keep all 91 classes.
    allowed_coco_ids: None = None


@dataclass
class SlotConfig:
    num_slots: int = 64
    # COCO has 91 categories; we keep 91 + 1 background
    num_classes: int = 92

    # Initial slot confidence logit (sigmoid(-2) ≈ 0.12 → "empty")
    conf_logit_init: float = -4.0
    # Logit added each time a slot is observed
    conf_logit_obs_increment: float = 2.5
    # Logit subtracted each step the slot is NOT observed
    conf_logit_decay: float = 0.05
    # Slots below this logit are treated as empty (available for reuse)
    conf_logit_empty_threshold: float = -3.0

    # Measurement noise std (metres) for EKF slot position update
    obs_noise_std: float = 0.30

    # Initial slot position logvar (log(σ²)) — large = high uncertainty
    pos_logvar_init: float = 2.0    # σ ≈ 7.4 m — very uncertain at birth


@dataclass
class PoseConfig:
    # 2D navigation plane: state = (x, y, θ)
    dim: int = 3

    # Initial pose uncertainty (log σ²)
    init_logvar_xy: float = -4.0    # σ ≈ 0.18 m — tight prior at episode start
    init_logvar_theta: float = -4.0

    # Process noise added per action step (log σ²)
    process_logvar_xy: float = -4.6    # σ ≈ 0.10 m odometry drift per step
    process_logvar_theta: float = -5.8  # σ ≈ 0.05 rad

    # Habitat action parameters
    forward_step_m: float = 0.25    # metres per MOVE_FORWARD
    turn_step_rad: float = 0.5236   # 30° per TURN_LEFT / TURN_RIGHT


@dataclass
class AssociationConfig:
    # Weighted cost = w_pos * L2_pos + w_cls * class_mismatch
    w_pos: float = 1.0
    w_cls: float = 0.5

    # Max cost to allow a match (above → unmatched, new slot)
    threshold: float = 2.5   # in normalised cost units

    # Minimum slot confidence to be a matching candidate
    min_conf_logit: float = -2.0


@dataclass
class GRUConfig:
    hidden_dim: int = 256
    input_dim: int = 64     # encoded context fed to GRU each step


@dataclass
class EFEConfig:
    horizon: int = 5
    # Actions during imagination: 0=forward, 1=turn_left, 2=turn_right
    # (stop action excluded from planning — only valid at execution time)
    num_plan_actions: int = 3    # 3^5 = 243 candidates

    # Weights
    w_risk: float = 8.0
    w_ambiguity: float = 0.5
    w_info_gain: float = 1.0
    w_action_cost: float = 0.1
    w_stay_penalty: float = 2.0
    # Penalty for sequences that open with an inverse turn pair (L→R or R→L).
    # These waste K-2 useful steps and cause the planner to oscillate between
    # two headings instead of committing to FORWARD near the goal.
    w_inverse_pair_penalty: float = 1.5

    # Goal preference: goal_radius defines the "success zone"
    goal_radius: float = 0.5    # metres

    # Field-of-view for slot visibility (degrees, symmetric)
    fov_deg: float = 90.0
    max_view_dist: float = 6.0  # metres


@dataclass
class ModelV6Config:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    slots: SlotConfig = field(default_factory=SlotConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    assoc: AssociationConfig = field(default_factory=AssociationConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    efe: EFEConfig = field(default_factory=EFEConfig)

    # VFE loss weights (used when training the GRU context network)
    w_recon_pos: float = 1.0
    w_recon_cls: float = 0.5
    w_kl_pose: float = 0.1
    w_kl_slots: float = 0.05
    w_sup_pose: float = 2.0     # supervised pose loss (when GT pose available)

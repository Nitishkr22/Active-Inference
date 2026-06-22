"""
belief_state.py — BeliefState and Detection dataclasses.

BeliefState encodes what the agent currently believes about:
  1. Its own pose     — continuous Gaussian over (x, y, θ)
  2. The object map   — N=64 slots, each a Gaussian over 3D position + class distribution
  3. Temporal context — GRU hidden state for planning

Coordinate system:
  World frame: right-handed 2D nav plane.
    x = forward,  y = left,  θ = yaw counter-clockwise from +x.
  Slot positions: 3D (x, y, z) in world frame (z = height, useful for tall objects).
  For 2D navigation, only (x, y) are used in risk/planning; z is maintained for
  reconstruction and future 3D extension.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from .config import ModelV6Config


@dataclass
class Detection:
    """One object detection lifted to 3D world-frame coordinates."""
    class_id: int           # COCO class id (0-indexed)
    class_conf: float       # detector confidence
    pos_world: torch.Tensor # [3]  (x, y, z) in world frame
    bbox_img: Tuple[float, float, float, float]  # (x1, y1, x2, y2) pixels

    def to(self, device: torch.device) -> "Detection":
        return Detection(
            class_id=self.class_id,
            class_conf=self.class_conf,
            pos_world=self.pos_world.to(device),
            bbox_img=self.bbox_img,
        )


@dataclass
class BeliefState:
    """
    Full belief at one timestep.  All tensors live on the same device.

    Shapes:
      pose_mu            [3]       (x, y, θ)
      pose_logvar        [3]       log variance per dim
      slot_pos_mu        [N, 3]   mean 3D position per slot
      slot_pos_logvar    [N, 3]   log variance per slot per dim
      slot_class_logits  [N, C]   unnormalised log-probs over C COCO classes
      slot_conf_logit    [N]      logit of "slot is occupied"
      h                  [1, 1, H] GRU hidden state (num_layers=1, batch=1)
      step               int      episode step counter
    """
    pose_mu:           torch.Tensor   # [3]
    pose_logvar:       torch.Tensor   # [3]
    slot_pos_mu:       torch.Tensor   # [N, 3]
    slot_pos_logvar:   torch.Tensor   # [N, 3]
    slot_class_logits: torch.Tensor   # [N, C]
    slot_conf_logit:   torch.Tensor   # [N]
    h:                 torch.Tensor   # [1, 1, H_gru]
    step:              int = 0

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        return self.pose_mu.device

    @property
    def num_slots(self) -> int:
        return self.slot_pos_mu.shape[0]

    @property
    def num_classes(self) -> int:
        return self.slot_class_logits.shape[1]

    @property
    def slot_conf(self) -> torch.Tensor:
        """Slot occupancy probability  [N]."""
        return torch.sigmoid(self.slot_conf_logit)

    @property
    def occupied_mask(self) -> torch.Tensor:
        """Boolean mask of slots with conf_logit > threshold  [N]."""
        # threshold is stored on config but we expose it here as a property
        # using a sensible default.  Callers should use SlotMemory.occupied_mask
        # for the config-aware version.
        return self.slot_conf_logit > -3.0

    @property
    def pose_std(self) -> torch.Tensor:
        """Pose standard deviation  [3]."""
        return torch.exp(0.5 * self.pose_logvar)

    # ------------------------------------------------------------------ #
    # Device / copy helpers
    # ------------------------------------------------------------------ #

    def to(self, device: torch.device) -> "BeliefState":
        return BeliefState(
            pose_mu=self.pose_mu.to(device),
            pose_logvar=self.pose_logvar.to(device),
            slot_pos_mu=self.slot_pos_mu.to(device),
            slot_pos_logvar=self.slot_pos_logvar.to(device),
            slot_class_logits=self.slot_class_logits.to(device),
            slot_conf_logit=self.slot_conf_logit.to(device),
            h=self.h.to(device),
            step=self.step,
        )

    def clone(self) -> "BeliefState":
        return BeliefState(
            pose_mu=self.pose_mu.clone(),
            pose_logvar=self.pose_logvar.clone(),
            slot_pos_mu=self.slot_pos_mu.clone(),
            slot_pos_logvar=self.slot_pos_logvar.clone(),
            slot_class_logits=self.slot_class_logits.clone(),
            slot_conf_logit=self.slot_conf_logit.clone(),
            h=self.h.clone(),
            step=self.step,
        )

    # ------------------------------------------------------------------ #
    # Entropy helpers (used by EFE planner)
    # ------------------------------------------------------------------ #

    def pose_entropy_2d(self) -> float:
        """Differential entropy of the 2D position marginal (x, y)."""
        # H[N(μ, σ²)] = 0.5 * log(2πe σ²)  — sum over x, y
        return float(0.5 * (self.pose_logvar[:2].sum() + 2 * math.log(2 * math.pi * math.e)).item())

    def pose_entropy_full(self) -> float:
        """Differential entropy of the full (x, y, θ) Gaussian."""
        return float(0.5 * (self.pose_logvar.sum() + 3 * math.log(2 * math.pi * math.e)).item())

    def slot_uncertainty(self, slot_idx: int) -> float:
        """Mean positional std for one slot (metres)."""
        return float(torch.exp(0.5 * self.slot_pos_logvar[slot_idx]).mean().item())

    # ------------------------------------------------------------------ #
    # Factory: initial belief at episode start
    # ------------------------------------------------------------------ #

    @staticmethod
    def initial(cfg: ModelV6Config, device: torch.device,
                known_pose: Optional[torch.Tensor] = None) -> "BeliefState":
        """
        Create a fresh belief at the start of an episode.

        known_pose: [3] (x, y, θ).  If provided, pose uncertainty is set tight.
        """
        N = cfg.slots.num_slots
        C = cfg.slots.num_classes
        H = cfg.gru.hidden_dim

        if known_pose is not None:
            pose_mu = known_pose.to(device).float()
            pose_logvar = torch.full((3,), cfg.pose.init_logvar_xy, device=device)
        else:
            pose_mu = torch.zeros(3, device=device)
            # Large initial uncertainty when pose is unknown
            pose_logvar = torch.tensor([2.0, 2.0, 1.0], device=device)

        return BeliefState(
            pose_mu=pose_mu,
            pose_logvar=pose_logvar,
            slot_pos_mu=torch.zeros(N, 3, device=device),
            slot_pos_logvar=torch.full((N, 3), cfg.slots.pos_logvar_init, device=device),
            slot_class_logits=torch.zeros(N, C, device=device),
            slot_conf_logit=torch.full((N,), cfg.slots.conf_logit_init, device=device),
            h=torch.zeros(1, 1, H, device=device),
            step=0,
        )

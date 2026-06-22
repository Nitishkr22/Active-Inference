"""
slot_memory.py — Bayesian EKF update for the slot-based object map.

Each slot tracks one object via:
  - 3D position: Gaussian (μ, σ²) updated with Kalman gain
  - Class distribution: logit-space Bayesian update (softmax prior × likelihood)
  - Confidence: logit updated by +increment on observation, -decay otherwise

Update rules:
  Matched slot j, detection i:
    K = σ²_j / (σ²_j + σ²_obs)              # Kalman gain per dim  [3]
    μ_j  ← μ_j  + K ⊙ (pos_i − μ_j)        # position update
    σ²_j ← (1 − K) ⊙ σ²_j                  # covariance shrink
    cls_j ← cls_j + Δ_cls(class_id_i)       # logit bump for observed class
    conf_j ← conf_j + increment              # more confident

  Unmatched occupied slot j:
    conf_j ← conf_j − decay                 # slowly fade unseen slots

  Unmatched detection i → write to emptiest available slot:
    new slot: μ = pos_i, σ² = init, cls = one-hot bump, conf = init + increment
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from .belief_state import BeliefState, Detection
from .config import ModelV6Config


class SlotMemory(nn.Module):
    """
    Slot-based map memory.  No learnable parameters in V6.0 — pure Bayesian EKF.
    Inherits nn.Module so it integrates cleanly into WorldModelV6.
    """

    def __init__(self, cfg: ModelV6Config):
        super().__init__()
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # Main update
    # ------------------------------------------------------------------ #

    def update(
        self,
        belief: BeliefState,
        detections: List[Detection],
        matched_pairs:   List[Tuple[int, int]],   # (det_idx, slot_idx)
        unmatched_dets:  List[int],
        unmatched_slots: List[int],
    ) -> BeliefState:
        """Return an updated clone of belief after applying all observations."""
        b = belief.clone()
        sc = self.cfg.slots

        # ---- 1. Update matched slots ----
        matched_slots = set()
        for det_idx, slot_idx in matched_pairs:
            det = detections[det_idx]
            self._ekf_pos_update(b, slot_idx, det.pos_world, sc.obs_noise_std)
            self._class_update(b, slot_idx, det.class_id)
            b.slot_conf_logit[slot_idx] = b.slot_conf_logit[slot_idx] + sc.conf_logit_obs_increment
            matched_slots.add(slot_idx)

        # ---- 2. Decay unmatched occupied slots ----
        for slot_idx in unmatched_slots:
            b.slot_conf_logit[slot_idx] = b.slot_conf_logit[slot_idx] - sc.conf_logit_decay

        # ---- 3. Spawn new slots for unmatched detections ----
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            free = self._find_free_slot(b)
            if free is None:
                continue  # map is full — silently drop (could evict lowest-conf)
            b.slot_pos_mu[free]      = det.pos_world.clone()
            b.slot_pos_logvar[free]  = torch.full((3,), sc.pos_logvar_init, device=b.device)
            b.slot_class_logits[free] = torch.zeros(sc.num_classes, device=b.device)
            self._class_update(b, free, det.class_id)
            b.slot_conf_logit[free] = sc.conf_logit_init + sc.conf_logit_obs_increment

        return b

    # ------------------------------------------------------------------ #
    # EKF position update (Kalman)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ekf_pos_update(
        belief: BeliefState,
        slot_idx: int,
        obs_pos: torch.Tensor,   # [3] observed world-frame position
        obs_noise_std: float,
    ) -> None:
        """In-place EKF update of slot position."""
        prior_var = torch.exp(belief.slot_pos_logvar[slot_idx])   # [3]
        obs_var   = torch.full((3,), obs_noise_std ** 2, device=belief.device)

        K = prior_var / (prior_var + obs_var)                     # Kalman gain  [3]
        innovation = obs_pos - belief.slot_pos_mu[slot_idx]       # [3]

        belief.slot_pos_mu[slot_idx]     = belief.slot_pos_mu[slot_idx] + K * innovation
        posterior_var                     = (1.0 - K) * prior_var
        belief.slot_pos_logvar[slot_idx] = torch.log(posterior_var.clamp_min(1e-8))

    # ------------------------------------------------------------------ #
    # Class logit update (Bayesian: add log-likelihood for observed class)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _class_update(
        belief: BeliefState,
        slot_idx: int,
        class_id: int,
        strength: float = 2.0,
    ) -> None:
        """
        Bayesian class update in logit space.
        Equivalent to multiplying prior by a likelihood that puts weight on class_id.
        """
        belief.slot_class_logits[slot_idx, class_id] = (
            belief.slot_class_logits[slot_idx, class_id] + strength
        )

    # ------------------------------------------------------------------ #
    # Find empty slot
    # ------------------------------------------------------------------ #

    def _find_free_slot(self, belief: BeliefState) -> int | None:
        """Return index of the lowest-confidence slot below the empty threshold."""
        sc = self.cfg.slots
        below = belief.slot_conf_logit < sc.conf_logit_empty_threshold
        if not below.any():
            # All slots occupied — evict the one with lowest confidence
            return int(belief.slot_conf_logit.argmin().item())
        # Among free slots, pick the most negative (most empty)
        conf = belief.slot_conf_logit.clone()
        conf[~below] = float("inf")
        return int(conf.argmin().item())

    # ------------------------------------------------------------------ #
    # Query helpers (used by EFE planner)
    # ------------------------------------------------------------------ #

    def visible_slots(
        self,
        belief: BeliefState,
        pose_mu: torch.Tensor,  # [3]  (x, y, θ)
    ) -> torch.Tensor:
        """
        Boolean mask [N] of slots that are within the camera's FoV from pose_mu.
        Only considers occupied slots.
        """
        efe_cfg = self.cfg.efe
        occupied = belief.slot_conf_logit > self.cfg.assoc.min_conf_logit  # [N]

        # 2D distance from agent to each slot (x, y only)
        dx = belief.slot_pos_mu[:, 0] - pose_mu[0]    # [N]
        dy = belief.slot_pos_mu[:, 1] - pose_mu[1]    # [N]
        dist = torch.sqrt(dx ** 2 + dy ** 2)           # [N]

        # Bearing from agent to slot, relative to heading θ
        bearing = torch.atan2(dy, dx) - pose_mu[2]    # [N]
        bearing = _wrap_angle(bearing)

        half_fov = math.radians(efe_cfg.fov_deg / 2.0)
        in_fov   = (bearing.abs() <= half_fov) & (dist <= efe_cfg.max_view_dist)

        return occupied & in_fov

    def map_uncertainty(self, belief: BeliefState, visible_mask: torch.Tensor) -> float:
        """
        Sum of positional uncertainty for visible slots.
        Used as info-gain proxy: exploring toward uncertain regions is epistemic.
        """
        if not visible_mask.any():
            return 0.0
        logvars = belief.slot_pos_logvar[visible_mask]    # [K, 3]
        return float(torch.exp(0.5 * logvars).sum().item())

    def occupied_mask(self, belief: BeliefState) -> torch.Tensor:
        """Boolean mask [N] using config threshold."""
        return belief.slot_conf_logit > self.cfg.slots.conf_logit_empty_threshold


# ------------------------------------------------------------------ #
# Angle util
# ------------------------------------------------------------------ #

def _wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-π, π]."""
    return (a + math.pi) % (2 * math.pi) - math.pi

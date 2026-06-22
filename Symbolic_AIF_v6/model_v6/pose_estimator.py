"""
pose_estimator.py — Continuous pose belief: EKF prediction + correction.

State: (x, y, θ) in world frame.  Belief: diagonal Gaussian (μ, log σ²).

Two-phase update per timestep:
  1. predict()  — propagate pose forward using wheel odometry
                  (uncertainty grows by process noise)
  2. correct()  — reduce uncertainty using matched detection residuals
                  (weighted average of EKF corrections from each matched pair)

Action semantics (Habitat conventions, overridable via PoseConfig):
  0 = MOVE_FORWARD  → x += cos(θ)*d,  y += sin(θ)*d
  1 = TURN_LEFT     → θ += Δθ
  2 = TURN_RIGHT    → θ -= Δθ
  3 = STOP          → no change
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .belief_state import BeliefState, Detection
from .config import ModelV6Config


class PoseEstimator(nn.Module):
    """
    EKF-style 2D pose estimator.  No learnable parameters in V6.0.
    """

    def __init__(self, cfg: ModelV6Config):
        super().__init__()
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # Odometry prediction  (called before observations arrive)
    # ------------------------------------------------------------------ #

    def predict(
        self,
        pose_mu:     torch.Tensor,   # [3]
        pose_logvar: torch.Tensor,   # [3]
        action:      int,
        odom:        Optional[torch.Tensor] = None,  # [3] measured (dx, dy, dθ)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate pose through one action step.

        If odom is provided (from wheel odometry sensor), it is used directly.
        Otherwise, the nominal action kinematics are used (ideal execution).
        Returns (pose_mu_prior, pose_logvar_prior).
        """
        pc = self.cfg.pose

        if odom is not None:
            # Use measured odometry: rotate displacement into world frame
            dx, dy, dtheta = odom[0].item(), odom[1].item(), odom[2].item()
        else:
            dx, dy, dtheta = _action_to_delta(action, pose_mu[2].item(), pc)

        delta = torch.tensor([dx, dy, dtheta], dtype=torch.float32, device=pose_mu.device)
        new_mu = pose_mu + delta
        new_mu[2] = _wrap_scalar(new_mu[2].item())

        # Process noise: logvar increases after each action
        process_noise = torch.tensor(
            [pc.process_logvar_xy, pc.process_logvar_xy, pc.process_logvar_theta],
            dtype=torch.float32, device=pose_mu.device,
        )
        new_logvar = _log_add_variances(pose_logvar, process_noise)

        return new_mu, new_logvar

    # ------------------------------------------------------------------ #
    # Correction from matched detections
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def correct(
        self,
        pose_mu:      torch.Tensor,           # [3]  pose prior from predict()
        pose_logvar:  torch.Tensor,           # [3]
        detections:   List[Detection],
        matched_pairs: List[Tuple[int, int]], # (det_idx, slot_idx)
        belief:       BeliefState,            # provides slot positions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        EKF correction step: pull pose toward positions consistent with
        seeing the matched slots at the detected 3D positions.

        For each matched (detection_i, slot_j):
          predicted observation = slot_j.pos_world (known from map)
          actual observation    = detection_i.pos_world (measured from RGBD)
          residual              = det.pos − slot.pos  (in world frame)
          → the agent must have moved by ~residual to reconcile the two

        We compute a per-pair pose correction and take the uncertainty-weighted
        mean across all pairs.

        Returns (pose_mu_posterior, pose_logvar_posterior).
        """
        if not matched_pairs:
            return pose_mu, pose_logvar

        corrections_x, corrections_y = [], []
        weights = []

        for det_idx, slot_idx in matched_pairs:
            det       = detections[det_idx]
            slot_mu   = belief.slot_pos_mu[slot_idx]   # [3]
            slot_var  = torch.exp(belief.slot_pos_logvar[slot_idx])  # [3]

            # Residual in (x, y) only
            res_x = det.pos_world[0] - slot_mu[0]
            res_y = det.pos_world[1] - slot_mu[1]

            # Weight by inverse of total uncertainty (pose + slot)
            pose_var = torch.exp(pose_logvar[:2])  # [2]
            total_var_x = pose_var[0] + slot_var[0]
            total_var_y = pose_var[1] + slot_var[1]

            # Kalman gain for pose (treating slot position as known measurement)
            K_x = pose_var[0] / total_var_x
            K_y = pose_var[1] / total_var_y

            corrections_x.append(K_x * res_x)
            corrections_y.append(K_y * res_y)
            # Weight for averaging: inverse of residual uncertainty
            w = 1.0 / (total_var_x + total_var_y + 1e-6)
            weights.append(w)

        # Weighted mean correction
        W = torch.stack(weights)
        W = W / W.sum()
        cx = sum(W[i] * corrections_x[i] for i in range(len(corrections_x)))
        cy = sum(W[i] * corrections_y[i] for i in range(len(corrections_y)))

        new_mu = pose_mu.clone()
        new_mu[0] = pose_mu[0] + cx
        new_mu[1] = pose_mu[1] + cy

        # Variance reduction: each matched pair reduces x,y uncertainty
        n = len(matched_pairs)
        pose_var  = torch.exp(pose_logvar)
        shrink    = (1.0 - W.mean()) ** n  # approximate; stays >0
        new_var   = pose_var.clone()
        new_var[0] = (pose_var[0] * shrink).clamp_min(1e-6)
        new_var[1] = (pose_var[1] * shrink).clamp_min(1e-6)
        new_logvar = torch.log(new_var)

        return new_mu, new_logvar

    # ------------------------------------------------------------------ #
    # Imagination: pose transition for EFE rollout (no odometry available)
    # ------------------------------------------------------------------ #

    def imagine_step(
        self,
        pose_mu:     torch.Tensor,  # [..., 3]  supports batched [N, 3]
        pose_logvar: torch.Tensor,  # [..., 3]
        action:      torch.Tensor,  # [...] int action per sequence
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched pose propagation for EFE planning.
        No odometry correction — ideal kinematics + process noise.

        action values: 0=forward, 1=turn_left, 2=turn_right
        """
        pc = self.cfg.pose
        device = pose_mu.device
        shape  = pose_mu.shape[:-1]   # leading batch dims

        x     = pose_mu[..., 0]
        y     = pose_mu[..., 1]
        theta = pose_mu[..., 2]

        fwd  = (action == 0).float()
        tl   = (action == 1).float()
        tr   = (action == 2).float()

        dx     = fwd * pc.forward_step_m * torch.cos(theta)
        dy     = fwd * pc.forward_step_m * torch.sin(theta)
        dtheta = tl * pc.turn_step_rad - tr * pc.turn_step_rad

        new_mu = torch.stack([x + dx, y + dy, theta + dtheta], dim=-1)
        # Wrap θ to [-π, π]
        new_mu[..., 2] = torch.atan2(torch.sin(new_mu[..., 2]),
                                      torch.cos(new_mu[..., 2]))

        noise = torch.tensor(
            [pc.process_logvar_xy, pc.process_logvar_xy, pc.process_logvar_theta],
            dtype=torch.float32, device=device,
        )
        new_logvar = _log_add_variances(pose_logvar, noise)

        return new_mu, new_logvar


# ------------------------------------------------------------------ #
# Utilities
# ------------------------------------------------------------------ #

def _action_to_delta(action: int, theta: float, pc) -> Tuple[float, float, float]:
    """Ideal kinematics: action int → (dx, dy, dθ) in world frame."""
    if action == 0:   # forward
        return (pc.forward_step_m * math.cos(theta),
                pc.forward_step_m * math.sin(theta),
                0.0)
    elif action == 1:  # turn left
        return (0.0, 0.0, pc.turn_step_rad)
    elif action == 2:  # turn right
        return (0.0, 0.0, -pc.turn_step_rad)
    else:              # stop or unknown
        return (0.0, 0.0, 0.0)


def _wrap_scalar(theta: float) -> float:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def _log_add_variances(
    logvar1: torch.Tensor,
    logvar2: torch.Tensor,
) -> torch.Tensor:
    """log(σ₁² + σ₂²) = log(exp(logvar1) + exp(logvar2))."""
    # Works for any broadcast-compatible shapes (e.g. [N,3] + [3]).
    return torch.log(torch.exp(logvar1) + torch.exp(logvar2))

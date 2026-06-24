"""
world_model.py — WorldModelV6: integrates perception, slot memory, pose estimator,
and GRU context into a single forward_step interface.

Per-step forward pass:
  1. Perception        : RGBD → detections in world frame
  2. Pose prediction   : carry pose forward with odometry (EKF predict)
  3. Data association  : Hungarian match detections ↔ slots
  4. Slot update       : EKF Bayesian update of matched slots; spawn new slots
  5. Pose correction   : EKF correct pose from matched (det, slot) pairs
  6. GRU context update: encode current observation summary → h_t
  7. VFE computation   : return reconstruction + KL losses for training

Belief rollout (for EFE planning):
  - Imagination: propagate pose forward for each candidate action sequence
  - No real observations — uses current slot map as the imagined world
  - Computes EFE terms: risk + ambiguity - info_gain

V6.0 design: EKF equations are analytical (no learnable params).
Only the GRU context network has learnable parameters.
The belief update is principled and works at step 1 without any training.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .belief_state import BeliefState, Detection
from .config import ModelV6Config
from .data_association import associate
from .perception import Perception
from .pose_estimator import PoseEstimator
from .slot_memory import SlotMemory
from .structural_detector import StructuralDetector


# ------------------------------------------------------------------ #
# GRU context encoder  (the only learned component in V6.0)
# ------------------------------------------------------------------ #

class ContextEncoder(nn.Module):
    """
    Encodes the current observation summary into a fixed-size vector for the GRU.

    Input features (concatenated):
      - pose_mu       [3]
      - pose_logvar   [3]        — how uncertain is the pose?
      - n_matched     [1]        — how many detections matched existing slots?
      - n_new         [1]        — how many new objects discovered?
      - n_occupied    [1]        — total occupied slots
      - mean_slot_conf[1]        — mean confidence of occupied slots
      - pose_correction_norm [1] — magnitude of EKF correction this step
    Total: 11 → projected to gru.input_dim
    """

    def __init__(self, cfg: ModelV6Config):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(11, cfg.gru.input_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        pose_mu:          torch.Tensor,   # [3]
        pose_logvar:      torch.Tensor,   # [3]
        n_matched:        int,
        n_new:            int,
        n_occupied:       int,
        mean_slot_conf:   float,
        pose_correction:  float,
    ) -> torch.Tensor:                    # [input_dim]
        feat = torch.cat([
            pose_mu,
            pose_logvar,
            torch.tensor([n_matched, n_new, n_occupied,
                          mean_slot_conf, pose_correction],
                         dtype=torch.float32, device=pose_mu.device),
        ], dim=0)  # [11]
        return self.proj(feat)


# ------------------------------------------------------------------ #
# WorldModelV6
# ------------------------------------------------------------------ #

class WorldModelV6(nn.Module):
    """
    Full V6 world model.  Call forward_step() at each timestep.
    """

    def __init__(self, cfg: ModelV6Config):
        super().__init__()
        self.cfg         = cfg
        self.perception  = Perception(cfg)
        self.struct_det  = StructuralDetector(cfg)
        self.slot_mem    = SlotMemory(cfg)
        self.pose_est    = PoseEstimator(cfg)
        self.ctx_encoder = ContextEncoder(cfg)
        self.gru         = nn.GRU(
            input_size=cfg.gru.input_dim,
            hidden_size=cfg.gru.hidden_dim,
            num_layers=1,
            batch_first=False,
        )

    # ------------------------------------------------------------------ #
    # Per-step update
    # ------------------------------------------------------------------ #

    def forward_step(
        self,
        rgb:          torch.Tensor,                   # [3, H, W]  float32  0–1
        depth:        torch.Tensor,                   # [1, H, W]  float32  metres
        action:       int,                            # action taken BEFORE this obs
        cam_to_world: torch.Tensor,                   # [4, 4]  camera→world
        prev_belief:  BeliefState,
        odom:         Optional[torch.Tensor] = None,  # [3] measured (dx,dy,dθ)
        gt_pose:      Optional[torch.Tensor] = None,  # [3] for supervised training
    ) -> Tuple[BeliefState, Dict[str, torch.Tensor]]:
        """
        Run one belief update step.

        Returns:
          new_belief : updated BeliefState
          aux        : dict with VFE loss components (for training)
        """
        device = prev_belief.device

        # ---- 1. Perception: RGB objects + depth structural elements ----
        detections: List[Detection] = self.perception(rgb, depth, cam_to_world)
        detections = [d.to(device) for d in detections]

        struct_dets = self.struct_det.detect(depth, cam_to_world)
        detections  = detections + [d.to(device) for d in struct_dets]

        # ---- 2. Pose prediction (odometry) ----
        pose_mu_prior, pose_logvar_prior = self.pose_est.predict(
            prev_belief.pose_mu, prev_belief.pose_logvar, action, odom
        )

        # Create working copy of belief with predicted pose
        b = prev_belief.clone()
        b.pose_mu     = pose_mu_prior
        b.pose_logvar = pose_logvar_prior

        # ---- 3. Data association ----
        matched_pairs, unmatched_dets, unmatched_slots = associate(
            detections, b, self.cfg.assoc, self.cfg.slots
        )

        # ---- 4. Slot update ----
        b = self.slot_mem.update(b, detections, matched_pairs,
                                 unmatched_dets, unmatched_slots)

        # ---- 5. Pose correction ----
        pose_mu_post, pose_logvar_post = self.pose_est.correct(
            b.pose_mu, b.pose_logvar, detections, matched_pairs, b
        )
        pose_delta = float((pose_mu_post[:2] - b.pose_mu[:2]).norm().item())
        b.pose_mu     = pose_mu_post
        b.pose_logvar = pose_logvar_post

        # ---- 6. GRU context ----
        occ_mask     = self.slot_mem.occupied_mask(b)
        n_occupied   = int(occ_mask.sum().item())
        mean_conf    = float(b.slot_conf[occ_mask].mean().item()) if n_occupied > 0 else 0.0

        ctx_input = self.ctx_encoder(
            pose_mu=b.pose_mu,
            pose_logvar=b.pose_logvar,
            n_matched=len(matched_pairs),
            n_new=len(unmatched_dets),
            n_occupied=n_occupied,
            mean_slot_conf=mean_conf,
            pose_correction=pose_delta,
        )  # [input_dim]

        h_new, _ = self.gru(
            ctx_input.unsqueeze(0).unsqueeze(0),   # [1, 1, input_dim]
            prev_belief.h,                          # [1, 1, H_gru]
        )
        b.h    = h_new
        b.step = prev_belief.step + 1

        # ---- 7. VFE components ----
        aux = self._compute_vfe(
            detections=detections,
            matched_pairs=matched_pairs,
            n_unmatched_dets=len(unmatched_dets),
            belief=b,
            pose_mu_prior=pose_mu_prior,
            pose_logvar_prior=pose_logvar_prior,
            gt_pose=gt_pose,
        )

        return b, aux

    # ------------------------------------------------------------------ #
    # VFE loss computation
    # ------------------------------------------------------------------ #

    def _compute_vfe(
        self,
        detections:        List[Detection],
        matched_pairs:     List[Tuple[int, int]],
        n_unmatched_dets:  int,
        belief:            BeliefState,
        pose_mu_prior:     torch.Tensor,
        pose_logvar_prior: torch.Tensor,
        gt_pose:           Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        device = belief.device
        cfg    = self.cfg

        # ---- Reconstruction loss (NLL of detection positions under slot Gaussians) ----
        recon_pos_loss  = torch.tensor(0.0, device=device)
        recon_cls_loss  = torch.tensor(0.0, device=device)

        for det_idx, slot_idx in matched_pairs:
            det      = detections[det_idx]
            mu       = belief.slot_pos_mu[slot_idx]       # [3]
            logvar   = belief.slot_pos_logvar[slot_idx]   # [3]
            cls_log  = belief.slot_class_logits[slot_idx] # [C]

            # Gaussian NLL: 0.5 * ((x-μ)²/σ² + log σ² + log 2π)
            diff      = det.pos_world - mu
            nll_pos   = 0.5 * ((diff ** 2) / logvar.exp() + logvar).sum()
            recon_pos_loss = recon_pos_loss + nll_pos

            # Cross-entropy on class
            target = torch.tensor([det.class_id], dtype=torch.long, device=device)
            nll_cls = F.cross_entropy(cls_log.unsqueeze(0), target)
            recon_cls_loss = recon_cls_loss + nll_cls

        if matched_pairs:
            n = len(matched_pairs)
            recon_pos_loss = recon_pos_loss / n
            recon_cls_loss = recon_cls_loss / n

        # ---- KL pose: q(pose) || N(pose_prior, σ²_prior) ----
        kl_pose = _kl_diagonal_gaussians(
            belief.pose_mu,     belief.pose_logvar,
            pose_mu_prior,      pose_logvar_prior,
        )

        # ---- KL slots (occupied slots vs N(0,I)) ----
        occ     = self.slot_mem.occupied_mask(belief)
        kl_slots = torch.tensor(0.0, device=device)
        if occ.any():
            kl_slots = _kl_diagonal_gaussians(
                belief.slot_pos_mu[occ],
                belief.slot_pos_logvar[occ],
                torch.zeros_like(belief.slot_pos_mu[occ]),
                torch.zeros_like(belief.slot_pos_logvar[occ]),
            )

        # ---- Supervised pose loss (if GT available) ----
        sup_pose_loss = torch.tensor(0.0, device=device)
        if gt_pose is not None:
            sup_pose_loss = F.mse_loss(belief.pose_mu, gt_pose.to(device).float())

        # ---- Total VFE ----
        vfe = (
            cfg.w_recon_pos  * recon_pos_loss
            + cfg.w_recon_cls  * recon_cls_loss
            + cfg.w_kl_pose    * kl_pose
            + cfg.w_kl_slots   * kl_slots
            + cfg.w_sup_pose   * sup_pose_loss
        )

        return {
            "vfe":              vfe,
            "recon_pos":        recon_pos_loss,
            "recon_cls":        recon_cls_loss,
            "kl_pose":          kl_pose,
            "kl_slots":         kl_slots,
            "sup_pose":         sup_pose_loss,
            "n_matched":        torch.tensor(float(len(matched_pairs))),
            "n_new_slots":      torch.tensor(float(n_unmatched_dets)),
        }

    # ------------------------------------------------------------------ #
    # Imagination rollout for EFE planning  (batched, no real observations)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def imagine_rollout(
        self,
        belief:      BeliefState,
        action_seqs: torch.Tensor,   # [N, K]  int   N sequences of length K
    ) -> Dict[str, torch.Tensor]:
        """
        Propagate pose belief forward for N candidate action sequences.

        Slot map is held fixed (objects don't move during imagination).
        Returns per-sequence EFE component tensors of shape [N, K].
        """
        N, K   = action_seqs.shape
        device = belief.device

        # Expand pose belief to N copies
        pose_mu     = belief.pose_mu.unsqueeze(0).expand(N, -1).clone()      # [N, 3]
        pose_logvar = belief.pose_logvar.unsqueeze(0).expand(N, -1).clone()  # [N, 3]

        risk_list, ambiguity_list, info_gain_list = [], [], []

        for k in range(K):
            acts = action_seqs[:, k]   # [N]

            pose_mu, pose_logvar = self.pose_est.imagine_step(
                pose_mu, pose_logvar, acts
            )

            # Risk: squared Mahalanobis distance from predicted pose to goal
            # Goal is set externally on the planner; here we return raw pose stats.
            # risk_k = None (goal is injected by EFEPlannerV4)

            # Ambiguity: differential entropy of 2D position Gaussian  [N]
            entropy_k = 0.5 * (pose_logvar[:, 0] + pose_logvar[:, 1])   # [N]
            ambiguity_list.append(entropy_k)

            # Info gain proxy: how many uncertain slots become visible at pred pose?
            # Approximate per sequence by checking each pose independently.
            ig_k = self._batch_info_gain(belief, pose_mu)   # [N]
            info_gain_list.append(ig_k)

        ambiguity_roll  = torch.stack(ambiguity_list,  dim=1)   # [N, K]
        info_gain_roll  = torch.stack(info_gain_list,  dim=1)   # [N, K]
        pose_mu_roll    = torch.stack([pose_mu], dim=1)          # [N, 1, 3] final only
        pose_logvar_roll = torch.stack([pose_logvar], dim=1)     # [N, 1, 3] final only

        return {
            "pose_mu_final":     pose_mu,          # [N, 3]  after K steps
            "pose_logvar_final": pose_logvar,       # [N, 3]
            "ambiguity_roll":    ambiguity_roll,    # [N, K]
            "info_gain_roll":    info_gain_roll,    # [N, K]
        }

    def _batch_info_gain(
        self,
        belief:    BeliefState,
        pose_mu:   torch.Tensor,   # [N, 3]
    ) -> torch.Tensor:
        """
        Vectorised info gain for N poses: sum of slot uncertainty for visible slots.
        Fully batched — no Python loop over N.  [N] tensor.
        """
        efe  = self.cfg.efe
        occ  = self.slot_mem.occupied_mask(belief)   # [S_total]

        if not occ.any():
            return torch.zeros(pose_mu.shape[0], device=pose_mu.device)

        slot_pos = belief.slot_pos_mu[occ]      # [S, 3]
        slot_lv  = belief.slot_pos_logvar[occ]  # [S, 3]
        slot_unc = torch.exp(0.5 * slot_lv[:, :2]).sum(dim=-1)  # [S]

        # [N, S] relative positions
        dx   = slot_pos[:, 0].unsqueeze(0) - pose_mu[:, 0].unsqueeze(1)  # [N, S]
        dy   = slot_pos[:, 1].unsqueeze(0) - pose_mu[:, 1].unsqueeze(1)  # [N, S]
        dist = torch.sqrt(dx ** 2 + dy ** 2)                              # [N, S]

        bear = torch.atan2(dy, dx) - pose_mu[:, 2].unsqueeze(1)          # [N, S]
        bear = torch.atan2(torch.sin(bear), torch.cos(bear))              # wrap

        half   = math.radians(efe.fov_deg / 2.0)
        in_fov = (bear.abs() <= half) & (dist <= efe.max_view_dist)       # [N, S]

        return (slot_unc.unsqueeze(0) * in_fov.float()).sum(-1)           # [N]

    # ------------------------------------------------------------------ #
    # Episode reset
    # ------------------------------------------------------------------ #

    def initial_belief(
        self,
        device: torch.device,
        known_pose: Optional[torch.Tensor] = None,
    ) -> BeliefState:
        return BeliefState.initial(self.cfg, device, known_pose)


# ------------------------------------------------------------------ #
# KL divergence between two diagonal Gaussians
# ------------------------------------------------------------------ #

def _kl_diagonal_gaussians(
    mu1:     torch.Tensor,
    logvar1: torch.Tensor,
    mu0:     torch.Tensor,
    logvar0: torch.Tensor,
) -> torch.Tensor:
    """
    KL(N(μ₁,σ₁²) || N(μ₀,σ₀²)) for diagonal Gaussians.
    Returns scalar.
    """
    var1 = logvar1.exp()
    var0 = logvar0.exp()
    kl   = 0.5 * (
        logvar0 - logvar1
        - 1.0
        + var1 / (var0 + 1e-8)
        + (mu1 - mu0) ** 2 / (var0 + 1e-8)
    )
    return kl.sum()

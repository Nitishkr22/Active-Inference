"""
efe_planner_v4.py — EFE planner for continuous-space navigation (V6).

Replaces V3's discrete 9×9 grid with continuous pose Gaussian.
Uses WorldModelV6.imagine_rollout() for batched GPU imagination.

EFE = Risk + Ambiguity - InfoGain + ActionCost

Risk       : expected negative log-preference at the goal.
             P*(s) ∝ exp(-||pose[:2] - goal||² / (2 * r²))
             Risk_k = ||μ_k[:2] - goal||² / (2 * r²)

Ambiguity  : differential entropy of pose Gaussian at each step.
             Ambiguity_k = 0.5 * (logvar_x + logvar_y)

InfoGain   : expected uncertainty reduction from exploring visible slots.
             Proxy: sum of positional std for visible, uncertain slots.

ActionCost : small penalty to prefer efficient actions.

Sequences  : 3^K (forward=0, turn_left=1, turn_right=2) — same count as V5.
             K=5, N=243.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model_v6 import BeliefState, WorldModelV6
from model_v6.config import EFEConfig


def _round_heading(theta: float, turn_step: float) -> float:
    """Round theta to the nearest multiple of turn_step, normalised to [-π, π]."""
    n = round(theta / turn_step)
    return math.atan2(math.sin(n * turn_step), math.cos(n * turn_step))


def _wrap(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


class EFEPlannerV4:
    """
    Stateless planner — receives a belief and goal, returns the best action.
    """

    def __init__(
        self,
        model: WorldModelV6,
        goal_pos: Optional[torch.Tensor] = None,   # [2]  (x, y) world frame
        cfg: Optional[EFEConfig] = None,
    ):
        self.model    = model
        self.goal_pos = goal_pos   # set/update via set_goal()
        self.cfg      = cfg or model.cfg.efe
        self._candidates: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # Goal management
    # ------------------------------------------------------------------ #

    def set_goal(self, goal_pos: torch.Tensor) -> None:
        """Update goal position (x, y) in world frame."""
        self.goal_pos = goal_pos.float()

    # ------------------------------------------------------------------ #
    # Main planning interface
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def select_action(
        self,
        belief:              BeliefState,
        recent_poses:        Optional[List[torch.Tensor]] = None,
        blocked_headings:    Optional[Set[float]] = None,  # headings where FORWARD is known to be blocked
    ) -> Dict[str, Any]:
        """
        Return the best first action for the current belief.

        Returns dict with:
          best_action  : int (0=forward, 1=turn_left, 2=turn_right)
          best_sequence: [int, ...]
          best_efe     : float
          all_efe      : [N] tensor
        """
        device     = belief.device
        candidates = self._get_candidates(device)   # [N, K]
        N, K       = candidates.shape

        # ---- Batched imagination rollout ----
        roll = self.model.imagine_rollout(belief, candidates)

        pose_mu_final     = roll["pose_mu_final"]      # [N, 3]
        ambiguity_roll    = roll["ambiguity_roll"]      # [N, K]
        info_gain_roll    = roll["info_gain_roll"]      # [N, K]

        # ---- Discount weights ----
        discount = torch.tensor(
            [0.9 ** k for k in range(K)],
            dtype=torch.float32, device=device,
        )  # [K]

        # ---- Risk: distance to goal at final step ----
        if self.goal_pos is not None:
            gp = self.goal_pos.to(device)
            dist_sq = ((pose_mu_final[:, :2] - gp.unsqueeze(0)) ** 2).sum(-1)  # [N]
            risk = dist_sq / (2.0 * self.cfg.goal_radius ** 2)                  # [N]
        else:
            risk = torch.zeros(N, device=device)

        # ---- Discounted ambiguity sum ----
        ambiguity = (discount.unsqueeze(0) * ambiguity_roll).sum(-1)   # [N]

        # ---- Discounted info gain sum ----
        info_gain = (discount.unsqueeze(0) * info_gain_roll).sum(-1)   # [N]

        # ---- Action cost (prefer forward over turning) ----
        fwd_count  = (candidates == 0).float().sum(-1)   # [N]
        action_cost = self.cfg.w_action_cost * (K - fwd_count)

        # ---- Stay / loop penalty ----
        stay_penalty = torch.zeros(N, device=device)
        if recent_poses and len(recent_poses) >= 2:
            last_xy = recent_poses[-1].to(device)[:2]
            pred_xy = pose_mu_final[:, :2]
            near_last = ((pred_xy - last_xy.unsqueeze(0)).norm(dim=-1) < 0.3).float()
            stay_penalty = self.cfg.w_stay_penalty * near_last

        # ---- Wall-collision penalty ----
        # For each candidate sequence, simulate the heading at the point the first
        # FORWARD is attempted (after any leading turns).  If that heading is in
        # blocked_headings (known wall), assign 1e6 to prevent oscillation between
        # two adjacent blocked headings.  The set clears when FORWARD succeeds.
        blocked_penalty = torch.zeros(N, device=device)
        if blocked_headings:
            turn_step  = self.model.cfg.pose.turn_step_rad
            curr_theta = belief.pose_mu[2].item()
            for i, seq in enumerate(candidates.tolist()):
                theta = curr_theta
                for act in seq:
                    if act == 0:   # FORWARD: check heading
                        h = _round_heading(theta, turn_step)
                        if h in blocked_headings:
                            blocked_penalty[i] = 1e6
                        break
                    elif act == 1:  # TURN_L
                        theta = _wrap(theta + turn_step)
                    elif act == 2:  # TURN_R
                        theta = _wrap(theta - turn_step)

        # If every sequence containing a FORWARD is blocked, disable the penalty
        # so the planner can still pick a forward direction instead of oscillating
        # between pure-turn sequences indefinitely.
        if blocked_headings:
            has_fwd = (candidates == 0).any(dim=-1)   # [N]
            if has_fwd.any() and (blocked_penalty[has_fwd] > 0).all():
                blocked_penalty = torch.zeros(N, device=device)

        # ---- Inverse-pair penalty ----
        # Sequences that open with L→R or R→L waste two steps on a do-undo turn.
        # The planner exploits them near the goal (the K-2 FORWARDs that follow
        # end up close to the target), but executing one step at a time means the
        # agent never commits: it alternates between the two turns forever.
        inverse_pair_penalty = torch.zeros(N, device=device)
        if K >= 2:
            first  = candidates[:, 0]   # [N]
            second = candidates[:, 1]   # [N]
            bad = ((first == 1) & (second == 2)) | ((first == 2) & (second == 1))
            inverse_pair_penalty[bad] = self.cfg.w_inverse_pair_penalty

        # ---- Total EFE ----
        efe = (
            self.cfg.w_risk       * risk
            + self.cfg.w_ambiguity  * ambiguity
            - self.cfg.w_info_gain  * info_gain
            + action_cost
            + stay_penalty
            + blocked_penalty
            + inverse_pair_penalty
        )

        best_idx = int(efe.argmin().item())
        best_seq = candidates[best_idx].cpu().tolist()

        return {
            "best_action":   int(best_seq[0]),
            "best_sequence": best_seq,
            "best_efe":      float(efe[best_idx].item()),
            "all_efe":       efe,
            "risk":          float(risk[best_idx].item()),
            "ambiguity":     float(ambiguity[best_idx].item()),
            "info_gain":     float(info_gain[best_idx].item()),
        }

    # ------------------------------------------------------------------ #
    # Enumerate all action sequences
    # ------------------------------------------------------------------ #

    def _get_candidates(self, device: torch.device) -> torch.Tensor:
        """Cache and return [N, K] tensor of all action sequences."""
        if self._candidates is not None:
            return self._candidates.to(device)
        K = self.cfg.horizon
        A = self.cfg.num_plan_actions
        seqs = list(itertools.product(range(A), repeat=K))
        self._candidates = torch.tensor(seqs, dtype=torch.long)
        return self._candidates.to(device)

    # ------------------------------------------------------------------ #
    # Distance to goal (for external logging)
    # ------------------------------------------------------------------ #

    def goal_distance(self, belief: BeliefState) -> float:
        """Euclidean distance from current pose estimate to goal."""
        if self.goal_pos is None:
            return float("inf")
        return float(
            (belief.pose_mu[:2] - self.goal_pos.to(belief.device)).norm().item()
        )

    def reached_goal(self, belief: BeliefState) -> bool:
        return self.goal_distance(belief) <= self.cfg.goal_radius

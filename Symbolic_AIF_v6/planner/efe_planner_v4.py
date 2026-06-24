"""
efe_planner_v4.py — EFE planner for continuous-space navigation (V6).

Replaces V3's discrete 9×9 grid with continuous pose Gaussian.
Uses WorldModelV6.imagine_rollout() for batched GPU imagination.

EFE = Risk + Ambiguity - InfoGain + ActionCost + WallPenalty

Risk       : expected negative log-preference at the goal.
             P*(s) ∝ exp(-||pose[:2] - goal||² / (2 * r²))
             Risk_k = ||μ_k[:2] - goal||² / (2 * r²)

Ambiguity  : differential entropy of pose Gaussian at each step.
             Ambiguity_k = 0.5 * (logvar_x + logvar_y)

InfoGain   : expected uncertainty reduction from exploring visible slots.
             Proxy: sum of positional std for visible, uncertain slots.

ActionCost : small penalty to prefer efficient actions.

WallPenalty: Gaussian repulsion from wall slots (Stage 2).
             Routes the agent through doorway slots when direct path blocked.

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
from model_v6 import BeliefState, WorldModelV6, WALL_ID, DOORWAY_ID
from model_v6.config import EFEConfig


# ── Module-level helpers ───────────────────────────────────────────────────────

def _round_heading(theta: float, turn_step: float) -> float:
    """Round theta to the nearest multiple of turn_step, normalised to [-π, π]."""
    n = round(theta / turn_step)
    return math.atan2(math.sin(n * turn_step), math.cos(n * turn_step))


def _wrap(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def _extract_structural_slots(
    belief:   BeliefState,
    slot_cfg,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Extract wall and doorway slot positions (XY) from the belief map.

    Returns:
        wall_xy : [W, 2] float32  or  None if no wall slots
        door_xy : [D, 2] float32  or  None if no doorway slots
    """
    occupied = belief.slot_conf_logit > slot_cfg.conf_logit_empty_threshold
    if not occupied.any():
        return None, None

    cls = belief.slot_class_logits[occupied].argmax(dim=-1)   # [M]
    pos = belief.slot_pos_mu[occupied]                         # [M, 3]

    wall_mask = cls == WALL_ID
    door_mask = cls == DOORWAY_ID

    wall_xy = pos[wall_mask, :2] if wall_mask.any() else None
    door_xy = pos[door_mask, :2] if door_mask.any() else None
    return wall_xy, door_xy


def _is_path_blocked(
    curr_xy:  torch.Tensor,   # [2]
    goal_xy:  torch.Tensor,   # [2]
    wall_xy:  torch.Tensor,   # [W, 2]
    margin:   float,
) -> bool:
    """
    Return True if any wall slot lies within `margin` metres of the line segment
    curr_xy → goal_xy (only the interior portion, t ∈ [0.1, 0.9]).
    """
    if wall_xy is None or wall_xy.numel() == 0:
        return False
    W      = wall_xy.shape[0]
    device = wall_xy.device
    A = curr_xy.to(device)
    B = goal_xy.to(device)
    BA = (B - A).unsqueeze(0).expand(W, 2)        # [W, 2]
    PA = wall_xy - A.unsqueeze(0)                  # [W, 2]
    t  = ((PA * BA).sum(-1) / ((BA * BA).sum(-1) + 1e-8)).clamp(0.1, 0.9)   # [W]
    proj = A.unsqueeze(0) + t.unsqueeze(-1) * BA   # [W, 2]
    dist = (wall_xy - proj).norm(dim=-1)            # [W]
    return bool((dist < margin).any().item())


def _find_best_doorway(
    curr_xy:  torch.Tensor,            # [2]
    goal_xy:  torch.Tensor,            # [2]
    door_xy:  Optional[torch.Tensor],  # [D, 2]  or None
) -> Optional[torch.Tensor]:
    """
    Among known doorway slots pick the one that minimises total detour:
        cost = dist(agent → door) + dist(door → goal)
    Returns [2] tensor or None if no doorways.
    """
    if door_xy is None or door_xy.numel() == 0:
        return None
    device    = door_xy.device
    d_to_door = (door_xy - curr_xy.to(device)).norm(dim=-1)    # [D]
    d_to_goal = (door_xy - goal_xy.to(device)).norm(dim=-1)    # [D]
    best_idx  = int((d_to_door + d_to_goal).argmin().item())
    return door_xy[best_idx]   # [2]


def _compute_wall_penalty(
    final_xy:  torch.Tensor,            # [N, 2]
    curr_xy:   torch.Tensor,            # [2]
    wall_xy:   Optional[torch.Tensor],  # [W, 2]  or None
    margin:    float,
    weight:    float,
    device:    torch.device,
) -> torch.Tensor:
    """
    Vectorised Gaussian repulsion from wall slots.

    For each candidate sequence i, compute the closest approach of the path
    segment (curr_xy → final_xy[i]) to every wall slot j.  Sum Gaussian
    penalties over all walls.

    Returns [N] penalty tensor (zero when no wall slots present).
    """
    if wall_xy is None or wall_xy.numel() == 0:
        return torch.zeros(final_xy.shape[0], device=device)

    N = final_xy.shape[0]
    W = wall_xy.shape[0]

    A = curr_xy.to(device).unsqueeze(0).unsqueeze(0).expand(N, W, 2)    # [N, W, 2]
    B = final_xy.unsqueeze(1).expand(N, W, 2)                            # [N, W, 2]
    P = wall_xy.to(device).unsqueeze(0).expand(N, W, 2)                  # [N, W, 2]

    BA  = B - A                                                            # [N, W, 2]
    PA  = P - A                                                            # [N, W, 2]
    t   = ((PA * BA).sum(-1) / ((BA * BA).sum(-1) + 1e-8)).clamp(0, 1)  # [N, W]
    proj = A + t.unsqueeze(-1) * BA                                        # [N, W, 2]
    dist = (P - proj).norm(dim=-1)                                         # [N, W]

    penalty = weight * torch.exp(-dist ** 2 / (2.0 * margin ** 2))        # [N, W]
    # max() over walls: penalty equals closest-wall repulsion only.
    # sum() caused pile-up with many mapped walls (14-20 everywhere), overwhelming risk.
    return penalty.max(-1).values                                           # [N]


# ── Planner class ──────────────────────────────────────────────────────────────

class EFEPlannerV4:
    """
    EFE planner for goal-directed navigation with structural-slot awareness.

    Stage 2 additions:
      - Wall repulsion: penalises imagined paths that pass near wall slots.
      - Doorway routing: when wall slots block the direct path to goal, the
        effective goal is redirected to the best doorway slot.  After the agent
        passes through the doorway, the original goal is restored.
    """

    def __init__(
        self,
        model:    WorldModelV6,
        goal_pos: Optional[torch.Tensor] = None,   # [2]  (x, y) world frame
        cfg:      Optional[EFEConfig]    = None,
    ):
        self.model    = model
        self.cfg      = cfg or model.cfg.efe
        self._candidates: Optional[torch.Tensor] = None

        # Effective goal (may temporarily point to a doorway)
        self.goal_pos:     Optional[torch.Tensor] = None
        # True final navigation target — never redirected
        self._nav_goal:    Optional[torch.Tensor] = None
        # Doorway-routing state
        self._via_door:          bool                   = False
        self._door_target:       Optional[torch.Tensor] = None
        self._via_door_steps:    int                    = 0   # steps spent in VIA-DOOR mode
        self._via_door_cooldown: int                    = 0   # steps to block re-activation

        if goal_pos is not None:
            self.set_goal(goal_pos)

    # ------------------------------------------------------------------ #
    # Goal management
    # ------------------------------------------------------------------ #

    def set_goal(self, goal_pos: torch.Tensor) -> None:
        """Set (or update) the final navigation goal. Resets door-routing state."""
        goal_pos             = goal_pos.float()
        self.goal_pos        = goal_pos
        self._nav_goal       = goal_pos.clone()
        self._via_door          = False
        self._door_target       = None
        self._via_door_steps    = 0
        self._via_door_cooldown = 0

    @property
    def is_via_door(self) -> bool:
        """True while the effective goal is a doorway waypoint, not the final goal."""
        return self._via_door

    # ------------------------------------------------------------------ #
    # Distance helpers
    # ------------------------------------------------------------------ #

    def goal_distance(self, belief: BeliefState) -> float:
        """Distance to the EFFECTIVE goal (doorway when routing, else final goal)."""
        if self.goal_pos is None:
            return float("inf")
        return float((belief.pose_mu[:2] - self.goal_pos.to(belief.device)).norm().item())

    def nav_goal_distance(self, belief: BeliefState) -> float:
        """Distance to the FINAL navigation goal regardless of door-routing state."""
        ref = self._nav_goal if self._nav_goal is not None else self.goal_pos
        if ref is None:
            return float("inf")
        return float((belief.pose_mu[:2] - ref.to(belief.device)).norm().item())

    def reached_goal(self, belief: BeliefState) -> bool:
        """True when the FINAL navigation goal has been reached."""
        return self.nav_goal_distance(belief) <= self.cfg.goal_radius

    # ------------------------------------------------------------------ #
    # Main planning interface
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def select_action(
        self,
        belief:           BeliefState,
        recent_poses:     Optional[List[torch.Tensor]] = None,
        blocked_headings: Optional[Set[float]]         = None,
        explored_poses:   Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Return the best first action for the current belief.

        Returns dict with:
          best_action  : int (0=forward, 1=turn_left, 2=turn_right)
          best_sequence: [int, ...]
          best_efe     : float
          all_efe      : [N] tensor
          risk / ambiguity / info_gain: float (for logging)
          wall_penalty : float (for logging)
          via_door     : bool  (True when routing through a doorway)
        """
        device     = belief.device
        candidates = self._get_candidates(device)   # [N, K]
        N, K       = candidates.shape

        # ---- Extract structural slots from current belief ----
        wall_xy, door_xy = _extract_structural_slots(belief, self.model.cfg.slots)

        # ---- Doorway-routing mode management (with hysteresis) ----
        # Only active when cfg.enable_door_routing is True.
        # When False (default), navmesh waypoints already route correctly through
        # doorways — adding a separate slot-based routing layer causes oscillation.
        if not self.cfg.enable_door_routing:
            pass   # VIA-DOOR disabled — skip entirely
        elif self._via_door and self._door_target is not None:
            self._via_door_steps += 1
            dist_to_door = (
                belief.pose_mu[:2].to(device) - self._door_target.to(device)
            ).norm()
            # Exit VIA-DOOR: passed through door, OR stuck too long
            if (dist_to_door <= self.cfg.door_pass_radius_m
                    or self._via_door_steps >= self.cfg.door_max_routing_steps):
                self._via_door          = False
                self._door_target       = None
                self._via_door_steps    = 0
                self._via_door_cooldown = 30  # block re-activation for 30 steps
                if self._nav_goal is not None:
                    self.goal_pos = self._nav_goal
        elif (
            self._nav_goal is not None
            and not self._via_door
            and wall_xy is not None
        ):
            # Decrement cooldown each step — VIA-DOOR cannot re-activate while > 0.
            # This prevents the robot from oscillating between "through the door" and
            # "path is blocked, go back to the door" when it is standing right next to
            # the doorway slot.
            if self._via_door_cooldown > 0:
                self._via_door_cooldown -= 1
            else:
                curr_xy  = belief.pose_mu[:2].to(device)
                nav_dist = float((curr_xy - self._nav_goal.to(device)).norm().item())
                # Only activate via-door when still far enough from the final goal.
                # At close range EFE + wall_penalty are sufficient; door-routing
                # at close range tends to oscillate because the same wall slots keep
                # triggering _is_path_blocked after the agent "passes through".
                if nav_dist > self.cfg.door_routing_min_dist_m:
                    if _is_path_blocked(curr_xy, self._nav_goal.to(device),
                                        wall_xy, self.cfg.door_block_margin_m):
                        best_door = _find_best_doorway(
                            curr_xy, self._nav_goal.to(device), door_xy
                        )
                        if best_door is not None:
                            self._via_door          = True
                            self._door_target       = best_door.to(device)
                            self.goal_pos           = best_door.to(device)
                            self._via_door_steps    = 0
                            self._via_door_cooldown = 0

        # ---- Batched imagination rollout ----
        roll = self.model.imagine_rollout(belief, candidates)

        pose_mu_final  = roll["pose_mu_final"]   # [N, 3]
        ambiguity_roll = roll["ambiguity_roll"]   # [N, K]
        info_gain_roll = roll["info_gain_roll"]   # [N, K]

        # ---- Discount weights ----
        discount = torch.tensor(
            [0.9 ** k for k in range(K)],
            dtype=torch.float32, device=device,
        )  # [K]

        # ---- Risk: distance to effective goal at final step ----
        if self.goal_pos is not None:
            gp      = self.goal_pos.to(device)
            dist_sq = ((pose_mu_final[:, :2] - gp.unsqueeze(0)) ** 2).sum(-1)  # [N]
            risk    = dist_sq / (2.0 * self.cfg.goal_radius ** 2)               # [N]
        else:
            risk = torch.zeros(N, device=device)

        # ---- Discounted ambiguity sum ----
        ambiguity = (discount.unsqueeze(0) * ambiguity_roll).sum(-1)   # [N]

        # ---- Discounted info gain sum ----
        info_gain = (discount.unsqueeze(0) * info_gain_roll).sum(-1)   # [N]

        # ---- Action cost (prefer forward over turning) ----
        fwd_count   = (candidates == 0).float().sum(-1)   # [N]
        action_cost = self.cfg.w_action_cost * (K - fwd_count)

        # ---- Stay / loop penalty ----
        stay_penalty = torch.zeros(N, device=device)
        if recent_poses and len(recent_poses) >= 2:
            last_xy = recent_poses[-1].to(device)[:2]
            pred_xy = pose_mu_final[:, :2]
            near_last    = ((pred_xy - last_xy.unsqueeze(0)).norm(dim=-1) < 0.3).float()
            stay_penalty = self.cfg.w_stay_penalty * near_last

        # ---- Wall-collision penalty ----
        blocked_penalty = torch.zeros(N, device=device)
        if blocked_headings:
            turn_step  = self.model.cfg.pose.turn_step_rad
            curr_theta = belief.pose_mu[2].item()
            for i, seq in enumerate(candidates.tolist()):
                theta = curr_theta
                for act in seq:
                    if act == 0:
                        h = _round_heading(theta, turn_step)
                        if h in blocked_headings:
                            blocked_penalty[i] = 1e6
                        break
                    elif act == 1:
                        theta = _wrap(theta + turn_step)
                    elif act == 2:
                        theta = _wrap(theta - turn_step)

        # Disable penalty if ALL forward-containing sequences are blocked
        if blocked_headings:
            has_fwd = (candidates == 0).any(dim=-1)
            if has_fwd.any() and (blocked_penalty[has_fwd] > 0).all():
                blocked_penalty = torch.zeros(N, device=device)

        # ---- Frontier penalty ----
        frontier_penalty = torch.zeros(N, device=device)
        if explored_poses and len(explored_poses) > 0:
            exp_xy   = torch.stack([p.to(device)[:2] for p in explored_poses], dim=0)
            pred_xy  = pose_mu_final[:, :2]
            dists    = torch.cdist(pred_xy, exp_xy)
            min_dist = dists.min(dim=-1).values
            near     = (min_dist < self.cfg.frontier_radius).float()
            frontier_penalty = self.cfg.w_frontier_penalty * near

        # ---- Inverse-pair penalty ----
        inverse_pair_penalty = torch.zeros(N, device=device)
        if K >= 2:
            first  = candidates[:, 0]
            second = candidates[:, 1]
            bad = ((first == 1) & (second == 2)) | ((first == 2) & (second == 1))
            inverse_pair_penalty[bad] = self.cfg.w_inverse_pair_penalty

        # ---- Wall proximity penalty (Stage 2) ----
        # Penalise any candidate whose imagined path segment (curr → final_xy)
        # passes close to a known wall slot.  Works in both phases.
        wall_penalty = _compute_wall_penalty(
            pose_mu_final[:, :2],
            belief.pose_mu[:2].to(device),
            wall_xy,
            self.cfg.wall_margin_m,
            self.cfg.w_wall_penalty,
            device,
        )

        # ---- Total EFE ----
        efe = (
            self.cfg.w_risk       * risk
            + self.cfg.w_ambiguity  * ambiguity
            - self.cfg.w_info_gain  * info_gain
            + action_cost
            + stay_penalty
            + blocked_penalty
            + inverse_pair_penalty
            + frontier_penalty
            + wall_penalty
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
            "wall_penalty":  float(wall_penalty[best_idx].item()),
            "via_door":      self._via_door,
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

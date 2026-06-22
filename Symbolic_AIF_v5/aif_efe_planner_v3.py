"""
aif_efe_planner_v3.py — Batched GPU EFE planner.

V2 (sequential):  243 rollout calls × 5 steps = 1215 transition model calls  → ~1883 ms/step
V3 (batched):     1  rollout call  × 5 steps, batch=243                      → target ~20 ms/step

Key idea: instead of looping over each candidate action sequence and calling
rollout_from_filtered_state with B=1 each time, expand the current belief to
B=243 copies and call rollout_from_filtered_state once with B=243.  All EFE
scoring is then done with vectorized tensor operations — no Python loop over
candidates.

The decoder is skipped during rollout (skip_recon=True) since recon_roll is
never used in planning, saving 243×5=1215 decoder forward passes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from aif_efe_planner_v2 import EFEPlannerV2, EFEPlannerConfig
from aif_planner_v3 import HEADING_TO_IDX, most_likely_state_from_belief


class EFEPlannerV3(EFEPlannerV2):
    """
    Drop-in replacement for EFEPlannerV2.  Only score_action_sequences_pure
    is overridden — everything else (config, helpers, rollout_sequence, etc.)
    is inherited unchanged so the planning logic stays identical.
    """

    @torch.no_grad()
    def score_action_sequences_pure(
        self,
        belief: Dict[str, torch.Tensor],
        recent_true_states: List[Tuple[int, int, str]],
    ) -> Dict[str, Any]:
        """
        Batched GPU implementation.

        All N candidate sequences are rolled out in a single model call and
        scored with vectorized tensor ops.  The result dict has the same keys
        as V2 so callers need no changes.
        """
        candidates = self.enumerate_all_action_sequences(self.cfg.horizon)
        if not candidates:
            return {
                "reference_sequence": [],
                "best_sequence": [],
                "best_first_action": None,
                "best_score": float("inf"),
                "all_details": [],
                "belief": belief,
            }

        N = len(candidates)
        K = len(candidates[0])
        device = belief["row_probs"].device
        eps = self.cfg.eps

        # ------------------------------------------------------------------ #
        # 1.  Single batched rollout  (B = N)
        # ------------------------------------------------------------------ #
        row_b    = belief["row_probs"].expand(N, -1).contiguous()      # [N, R]
        col_b    = belief["col_probs"].expand(N, -1).contiguous()      # [N, C]
        hdg_b    = belief["heading_probs"].expand(N, -1).contiguous()  # [N, Hd]
        ctx_b    = belief["context"].expand(N, -1).contiguous()        # [N, U]
        acts_t   = torch.tensor(candidates, dtype=torch.long, device=device)  # [N, K]

        roll = self.model.rollout_from_filtered_state(
            row_probs_start=row_b,
            col_probs_start=col_b,
            heading_probs_start=hdg_b,
            context_start=ctx_b,
            action_seq=acts_t,
            skip_recon=True,          # decoder not needed for planning
        )

        row_roll  = roll["row_probs_roll"]        # [N, K, R]
        col_roll  = roll["col_probs_roll"]        # [N, K, C]
        hdg_roll  = roll["heading_probs_roll"]    # [N, K, Hd]
        coll_roll = roll["collision_logits_roll"] # [N, K, 2]

        reach = self.reachable_mask.to(device)    # [R, C]
        nlp   = self.neg_log_pref.to(device)      # [R, C]

        # ------------------------------------------------------------------ #
        # 2.  Risk  E[-log P(o*)] under joint (row, col) belief  [N, K]
        # ------------------------------------------------------------------ #
        # Outer product: [N, K, R, C]
        joint = row_roll.unsqueeze(-1) * col_roll.unsqueeze(-2)
        joint = joint * reach.unsqueeze(0).unsqueeze(0)
        valid_mass = joint.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)
        joint = joint / valid_mass
        risk  = (joint * nlp.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))  # [N, K]

        # ------------------------------------------------------------------ #
        # 3.  Ambiguity (entropy of pose belief)  [N, K]
        # ------------------------------------------------------------------ #
        def _ent(p: torch.Tensor) -> torch.Tensor:
            return -(p * p.clamp_min(eps).log()).sum(dim=-1)

        ambiguity = _ent(row_roll) + _ent(col_roll) + _ent(hdg_roll)  # [N, K]

        # ------------------------------------------------------------------ #
        # 4.  Collision probability  [N, K]
        # ------------------------------------------------------------------ #
        coll_prob = torch.softmax(coll_roll, dim=-1)[:, :, 1]  # [N, K]

        # ------------------------------------------------------------------ #
        # 5.  Info gain  max(H_current - H_k, 0)  [N, K]
        # ------------------------------------------------------------------ #
        row0, col0, hdg0 = belief["row_probs"][0], belief["col_probs"][0], belief["heading_probs"][0]
        current_entropy = self.state_entropy_from_belief(row0, col0, hdg0)
        info_gain = (current_entropy - ambiguity).clamp_min(0.0)  # [N, K]

        # ------------------------------------------------------------------ #
        # Adaptive precision: scale risk DOWN and epistemic terms UP when
        # belief entropy is high (agent is lost / uncertain).
        # ------------------------------------------------------------------ #
        if self.cfg.adaptive_precision and self.cfg.adaptive_entropy_threshold > 0.0:
            h_norm = min(current_entropy / self.cfg.adaptive_entropy_threshold, 1.0)
        else:
            h_norm = 0.0
        eff_w_risk             = self.cfg.w_risk             * (1.0 - (1.0 - self.cfg.adaptive_risk_min_scale) * h_norm)
        eff_w_terminal_risk    = self.cfg.w_terminal_risk    * (1.0 - 0.25 * h_norm)
        eff_w_context_unc      = self.cfg.w_context_uncertainty * (1.0 + (self.cfg.adaptive_epistemic_boost - 1.0) * h_norm)
        eff_w_info_gain        = self.cfg.w_info_gain        * (1.0 + 2.0 * h_norm)

        # ------------------------------------------------------------------ #
        # 6.  Context uncertainty  [N, K]  (optional)
        # ------------------------------------------------------------------ #
        if self.cfg.use_context_uncertainty and "context_logvar_roll" in roll and roll["context_logvar_roll"] is not None:
            ctx_unc = torch.exp(roll["context_logvar_roll"]).mean(dim=-1)  # [N, K]
        else:
            ctx_unc = torch.zeros(N, K, device=device)

        # ------------------------------------------------------------------ #
        # 7.  Discount weights  [K]
        # ------------------------------------------------------------------ #
        discount = torch.tensor(
            [self.cfg.discount ** k for k in range(K)],
            dtype=torch.float32, device=device,
        )

        # ------------------------------------------------------------------ #
        # 8.  Discounted sums  [N]
        # ------------------------------------------------------------------ #
        risk_sum      = (discount * risk).sum(-1)      * eff_w_risk
        ambiguity_sum = (discount * ambiguity).sum(-1) * self.cfg.w_ambiguity
        coll_sum      = (discount * coll_prob).sum(-1) * self.cfg.w_collision
        info_gain_sum = (discount * info_gain).sum(-1) * eff_w_info_gain
        ctx_unc_sum   = (discount * ctx_unc).sum(-1)   * eff_w_context_unc

        terminal_risk = eff_w_terminal_risk * risk[:, -1]      # [N]
        min_risk      = self.cfg.w_min_risk * risk.min(dim=-1).values  # [N]

        # ------------------------------------------------------------------ #
        # 9.  Action costs  [N]
        # ------------------------------------------------------------------ #
        cost_table = torch.tensor(
            [self.cfg.cost_forward, self.cfg.cost_backward,
             self.cfg.cost_turn,    self.cfg.cost_turn],
            dtype=torch.float32, device=device,
        )
        action_cost = cost_table[acts_t].sum(dim=-1)  # [N]

        # ------------------------------------------------------------------ #
        # 10.  Inverse-action penalties  [N]
        # ------------------------------------------------------------------ #
        if K > 1:
            prev_a = acts_t[:, :-1]
            curr_a = acts_t[:, 1:]
            fb_inv   = ((prev_a == 0) & (curr_a == 1)) | ((prev_a == 1) & (curr_a == 0))
            turn_inv = ((prev_a == 2) & (curr_a == 3)) | ((prev_a == 3) & (curr_a == 2))
            inverse_cost = (
                fb_inv.float()   * self.cfg.inverse_fb_penalty
                + turn_inv.float() * self.cfg.inverse_turn_penalty
            ).sum(dim=-1)  # [N]
        else:
            inverse_cost = torch.zeros(N, device=device)

        # ------------------------------------------------------------------ #
        # 11.  Loop / same-state penalties  [N]
        # Predicted most-likely state per (n, k) via argmax per factor.
        # ------------------------------------------------------------------ #
        pred_r = row_roll.argmax(dim=-1)   # [N, K]
        pred_c = col_roll.argmax(dim=-1)   # [N, K]
        pred_h = hdg_roll.argmax(dim=-1)   # [N, K]
        pred_states = torch.stack([pred_r, pred_c, pred_h], dim=-1)  # [N, K, 3]

        # Same consecutive state
        stay_cost = (
            (pred_states[:, 1:, :] == pred_states[:, :-1, :]).all(dim=-1).float().sum(dim=-1)
            * self.cfg.imagination_same_state_penalty
        )  # [N]

        # Any repeated state (loop)
        loop_cost = torch.zeros(N, device=device)
        for k in range(1, K):
            curr = pred_states[:, k:k+1, :]   # [N, 1, 3]
            prev = pred_states[:, :k, :]       # [N, k, 3]
            appeared = (curr == prev).all(dim=-1).any(dim=-1).float()  # [N]
            loop_cost = loop_cost + appeared * self.cfg.imagination_loop_penalty

        # ------------------------------------------------------------------ #
        # 12.  Recent-state revisit penalty  [N]
        # ------------------------------------------------------------------ #
        recent_revisit = torch.zeros(N, device=device)
        tail = recent_true_states[-6:]
        if tail:
            tail_r = torch.tensor([t[0] for t in tail], dtype=torch.long, device=device)
            tail_c = torch.tensor([t[1] for t in tail], dtype=torch.long, device=device)
            tail_h = torch.tensor([HEADING_TO_IDX[t[2]] for t in tail], dtype=torch.long, device=device)
            M = len(tail)
            for k in range(K):
                ps = pred_states[:, k, :]  # [N, 3]
                for idx in range(M):
                    match = (
                        (ps[:, 0] == tail_r[idx])
                        & (ps[:, 1] == tail_c[idx])
                        & (ps[:, 2] == tail_h[idx])
                    ).float()  # [N]
                    penalty = (
                        self.cfg.immediate_recent_state_penalty
                        if idx == M - 1
                        else self.cfg.recent_state_revisit_penalty
                    )
                    recent_revisit = recent_revisit + match * penalty

        # ------------------------------------------------------------------ #
        # 13.  Reference prefix cost  [N]  (usually 0.0 in pure_efe mode)
        # ------------------------------------------------------------------ #
        if self.cfg.reference_prefix_penalty > 0.0:
            ml_r, ml_c, ml_h = most_likely_state_from_belief(row0, col0, hdg0, reach)
            from aif_planner_v3 import build_reference_action_sequence
            ref_seq = build_reference_action_sequence(
                r=ml_r, c=ml_c, h=ml_h,
                dist_t=self.dist_t.to(device),
                reachable_mask=reach,
                horizon=K,
            )
            if ref_seq:
                ref_t = torch.tensor(ref_seq[:K], dtype=torch.long, device=device)
                ref_decay = torch.tensor(
                    [self.cfg.reference_prefix_decay ** k for k in range(K)],
                    dtype=torch.float32, device=device,
                )
                deviates = (acts_t != ref_t.unsqueeze(0)).float()   # [N, K]
                ref_cost = (deviates * ref_decay.unsqueeze(0)).sum(-1) * self.cfg.reference_prefix_penalty
            else:
                ref_cost = torch.zeros(N, device=device)
        else:
            ref_cost = torch.zeros(N, device=device)

        # ------------------------------------------------------------------ #
        # 14.  Total EFE  [N]
        # ------------------------------------------------------------------ #
        efe = (
            risk_sum
            + terminal_risk
            + min_risk
            + ambiguity_sum
            + coll_sum
            - info_gain_sum
            - ctx_unc_sum
            + action_cost
            + inverse_cost
            + ref_cost
            + loop_cost
            + stay_cost
            + recent_revisit
        )

        best_idx = int(efe.argmin().item())
        best_seq = candidates[best_idx]

        return {
            "reference_sequence": [],
            "best_sequence":      best_seq,
            "best_first_action":  int(best_seq[0]),
            "best_score":         float(efe[best_idx].item()),
            "all_details":        [],   # not populated in batched mode
            "belief":             belief,
        }

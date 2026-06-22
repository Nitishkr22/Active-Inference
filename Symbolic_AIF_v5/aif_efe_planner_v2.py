# aif_efe_planner_v2.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import math
import torch
import torch.nn.functional as F

from aif_planner_v3 import (
    ACTION_NAMES,
    HEADING_TO_IDX,
    IDX_TO_HEADING,
    entropy_categorical,
    collision_probability_from_logits,
    masked_joint_from_belief,
    most_likely_state_from_belief,
    build_reference_action_sequence,
    action_inverse,
)


@dataclass
class EFEPlannerConfig:
    horizon: int = 5
    max_candidates: int = 128
    allow_backward: bool = True

    # Pragmatic value (risk)
    w_risk: float = 4.0
    w_terminal_risk: float = 10.0
    w_min_risk: float = 4.0

    # Epistemic value
    w_ambiguity: float = 0.10
    # info_gain rewards entropy *reduction* along the rollout horizon:
    #   info_gain_k = max(H(s_0) - H(s_k), 0)
    # A positive w_info_gain subtracts from EFE (lower EFE = better),
    # rewarding sequences that lead to more certain beliefs.
    # For pure exploration you would *add* ambiguity instead of subtracting info_gain.
    w_info_gain: float = 0.25

    # Context uncertainty (V4-specific epistemic term)
    # The VAE context logvar tells us where the model is uncertain about the
    # unstructured aspects of the scene.  High context variance → high epistemic
    # value → agent is attracted to that region to reduce uncertainty.
    w_context_uncertainty: float = 0.15
    use_context_uncertainty: bool = True

    # Decoder-based epistemic value (expensive, off by default)
    # When enabled, the planner samples a hypothetical observation from the
    # decoder and estimates how much it would sharpen the belief — a closer
    # approximation to the true mutual-information epistemic term.
    use_decoder_epistemic: bool = False
    w_decoder_epistemic: float = 0.20

    # Collision
    w_collision: float = 18.0

    preference_precision: float = 1.00
    preference_floor: float = 1e-8
    discount: float = 0.90

    cost_forward: float = 0.00
    cost_backward: float = 1.80
    cost_turn: float = 0.20

    inverse_fb_penalty: float = 4.00
    inverse_turn_penalty: float = 2.00

    imagination_loop_penalty: float = 5.00
    imagination_same_state_penalty: float = 6.00
    recent_state_revisit_penalty: float = 4.00
    immediate_recent_state_penalty: float = 7.00

    reference_prefix_penalty: float = 2.5
    reference_prefix_decay: float = 0.75

    eps: float = 1e-8

    # Adaptive precision weighting (V4d+)
    adaptive_precision: bool = True
    adaptive_entropy_threshold: float = 0.10
    adaptive_risk_min_scale: float = 0.50
    adaptive_epistemic_boost: float = 6.0


class EFEPlannerV2:
    """
    V2 EFE planner — extends EFEPlannerV1 with uncertainty signals from WorldModelV4.

    Additional epistemic terms vs V1:

    1. Context uncertainty bonus:
         The VAE context logvar from V4 reveals where the model is uncertain
         about the continuous, unstructured part of the state.  Policies that
         lead to high-variance context regions are rewarded (exploration drive).

    2. Decoder epistemic bonus (optional, flag use_decoder_epistemic):
         For each imagined future state, sample a hypothetical observation
         from the decoder, re-encode it, and measure how much the belief
         would sharpen.  This approximates the true expected information gain
         I(s_future ; o_future | pi) more accurately than entropy difference.

    The model reference passed to __init__ should be a WorldModelV4 instance
    (or any model with forward_filter / rollout_from_filtered_state).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dist_t: torch.Tensor,
        reachable_mask: torch.Tensor,
        cfg: Optional[EFEPlannerConfig] = None,
    ) -> None:
        self.model = model
        self.dist_t = dist_t
        self.reachable_mask = reachable_mask
        self.cfg = cfg if cfg is not None else EFEPlannerConfig()

        self.neg_log_pref = self._build_negative_log_preferences(
            dist_t=dist_t,
            reachable_mask=reachable_mask,
        )

    def _build_negative_log_preferences(
        self,
        dist_t: torch.Tensor,
        reachable_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build C-like state preferences from shortest-path distance.
        Preferred states near goal get high probability; risk = -log preference.
        """
        device = dist_t.device

        pref_logits = -self.cfg.preference_precision * dist_t
        pref_logits = pref_logits.masked_fill(reachable_mask <= 0.5, -1e9)

        pref = torch.softmax(pref_logits.reshape(-1), dim=0).reshape_as(dist_t)
        pref = pref.clamp_min(self.cfg.preference_floor)

        neg_log_pref = -torch.log(pref)
        return neg_log_pref.to(device)

    @torch.no_grad()
    def infer_current_belief(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if hasattr(self.model, "forward_filter"):
            out = self.model.forward_filter(observations, actions)
        elif hasattr(self.model, "filter_sequence"):
            out = self.model.filter_sequence(observations, actions)
        else:
            raise AttributeError("Model must provide forward_filter() or filter_sequence().")

        belief = {
            "row_probs": out["row_probs_seq"][:, -1, :],
            "col_probs": out["col_probs_seq"][:, -1, :],
            "heading_probs": out["heading_probs_seq"][:, -1, :],
        }

        if "context_seq" in out:
            belief["context"] = out["context_seq"][:, -1, :]

        # Include VAE uncertainty from V4 if available
        if "context_logvar_seq" in out:
            belief["context_logvar"] = out["context_logvar_seq"][:, -1, :]

        return belief

    def enumerate_action_sequences_from_reference(
        self,
        ref_seq: List[int],
    ) -> List[List[int]]:
        H = len(ref_seq)
        candidates = set()
        candidates.add(tuple(ref_seq))

        action_set = [0, 2, 3]
        if self.cfg.allow_backward:
            action_set = [0, 1, 2, 3]

        for i in range(H):
            for a in action_set:
                if a != ref_seq[i]:
                    seq = list(ref_seq)
                    seq[i] = a
                    candidates.add(tuple(seq))

        for i in range(H - 1):
            for a1 in action_set:
                for a2 in action_set:
                    seq = list(ref_seq)
                    seq[i] = a1
                    seq[i + 1] = a2
                    candidates.add(tuple(seq))

        if H >= 1:
            candidates.add(tuple(([0] + ref_seq[1:])[:H]))
            candidates.add(tuple(([2] + ref_seq[1:])[:H]))
            candidates.add(tuple(([3] + ref_seq[1:])[:H]))
            if self.cfg.allow_backward:
                candidates.add(tuple(([1] + ref_seq[1:])[:H]))

        cand_list = [list(x) for x in candidates]
        cand_list.sort()

        if len(cand_list) > self.cfg.max_candidates:
            cand_list = cand_list[: self.cfg.max_candidates]

        return cand_list

    def enumerate_all_action_sequences(self, horizon: int) -> List[List[int]]:
        """Pure EFE candidate generation — enumerates action sequences directly."""
        action_set = [0, 2, 3]
        if self.cfg.allow_backward:
            action_set = [0, 1, 2, 3]

        candidates: List[List[int]] = [[]]
        for _ in range(horizon):
            new_candidates = []
            for seq in candidates:
                for a in action_set:
                    if len(seq) > 0 and action_inverse(seq[-1], a):
                        continue
                    new_candidates.append(seq + [a])
            candidates = new_candidates

        candidates.sort()
        if len(candidates) > self.cfg.max_candidates:
            candidates = candidates[: self.cfg.max_candidates]

        return candidates

    def state_entropy_from_belief(
        self,
        row_probs: torch.Tensor,
        col_probs: torch.Tensor,
        heading_probs: torch.Tensor,
    ) -> float:
        e = (
            entropy_categorical(row_probs, self.cfg.eps)
            + entropy_categorical(col_probs, self.cfg.eps)
            + entropy_categorical(heading_probs, self.cfg.eps)
        )
        return float(e.item())

    def expected_risk(
        self,
        row_probs: torch.Tensor,
        col_probs: torch.Tensor,
    ) -> float:
        joint, valid_mass = masked_joint_from_belief(
            row_probs=row_probs,
            col_probs=col_probs,
            reachable_mask=self.reachable_mask.to(row_probs.device),
        )

        if valid_mass <= 1e-8:
            r = int(torch.argmax(row_probs).item())
            c = int(torch.argmax(col_probs).item())
            return float(self.neg_log_pref[r, c].item())

        return float((joint * self.neg_log_pref.to(row_probs.device)).sum().item())

    def _context_uncertainty_bonus(
        self,
        context_logvar_roll: Optional[torch.Tensor],
        k: int,
    ) -> float:
        """Mean predicted context variance at rollout step k → epistemic bonus."""
        if not self.cfg.use_context_uncertainty:
            return 0.0
        if context_logvar_roll is None:
            return 0.0
        # context_logvar_roll: [K, context_dim]  (single batch item)
        var = torch.exp(context_logvar_roll[k]).mean().item()
        return var

    def _decoder_epistemic_bonus(
        self,
        row_probs_k: torch.Tensor,   # [1, R]
        col_probs_k: torch.Tensor,   # [1, C]
        heading_probs_k: torch.Tensor,  # [1, Hd]
        context_k: torch.Tensor,        # [1, U]
    ) -> float:
        """
        Approximate epistemic value via decoder sampling.

        Sample a hypothetical observation o' from P(o|s_k), then estimate
        how much that observation would sharpen belief by comparing entropy
        of p(s_k) vs the re-encoded posterior.  Requires model.decoder and
        model.factor_heads to be accessible.
        """
        if not self.cfg.use_decoder_epistemic:
            return 0.0

        decoder = getattr(self.model, "decoder", None)
        encoder = getattr(self.model, "encoder", None)
        factor_heads = getattr(self.model, "factor_heads", None)

        if decoder is None or encoder is None or factor_heads is None:
            return 0.0

        try:
            # Sample hypothetical obs
            sampled_obs = decoder(row_probs_k, col_probs_k, heading_probs_k, context_k)
            # Re-encode
            feat = encoder(sampled_obs)
            facts = factor_heads(feat)
            post_row = torch.softmax(facts["row_logits"], dim=-1)
            post_col = torch.softmax(facts["col_logits"], dim=-1)
            post_heading = torch.softmax(facts["heading_logits"], dim=-1)

            prior_entropy = self.state_entropy_from_belief(
                row_probs_k[0], col_probs_k[0], heading_probs_k[0]
            )
            post_entropy = self.state_entropy_from_belief(
                post_row[0], post_col[0], post_heading[0]
            )
            # Positive when observation would sharpen belief
            return max(prior_entropy - post_entropy, 0.0)
        except Exception:
            return 0.0

    @torch.no_grad()
    def rollout_sequence(
        self,
        belief: Dict[str, torch.Tensor],
        action_seq: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        device = belief["row_probs"].device
        seq_t = torch.tensor(action_seq, dtype=torch.long, device=device).unsqueeze(0)

        if hasattr(self.model, "rollout_from_filtered_state"):
            return self.model.rollout_from_filtered_state(
                row_probs_start=belief["row_probs"],
                col_probs_start=belief["col_probs"],
                heading_probs_start=belief["heading_probs"],
                context_start=belief.get("context", None),
                action_seq=seq_t,
            )

        if hasattr(self.model, "rollout_from_belief"):
            inputs = {
                "row_probs_start": belief["row_probs"],
                "col_probs_start": belief["col_probs"],
                "heading_probs_start": belief["heading_probs"],
                "action_seq": seq_t,
            }
            if "context" in belief:
                inputs["context_start"] = belief["context"]
            return self.model.rollout_from_belief(**inputs)

        raise AttributeError(
            "Model must provide rollout_from_filtered_state() or rollout_from_belief()."
        )

    @torch.no_grad()
    def score_sequence(
        self,
        belief: Dict[str, torch.Tensor],
        action_seq: Sequence[int],
        recent_true_states: List[Tuple[int, int, str]],
    ) -> Dict[str, Any]:

        risk_values = []
        loop_cost_sum = 0.0
        stay_cost_sum = 0.0
        recent_revisit_cost_sum = 0.0
        seen_pred_states = []
        context_uncertainty_sum = 0.0
        decoder_epistemic_sum = 0.0

        device = belief["row_probs"].device

        roll = self.rollout_sequence(belief, action_seq)

        row_roll = roll["row_probs_roll"][0]
        col_roll = roll["col_probs_roll"][0]
        hdg_roll = roll["heading_probs_roll"][0]
        coll_roll = roll["collision_logits_roll"][0]

        # Context logvar from V4 rollout (may not exist for V3 models)
        context_logvar_roll: Optional[torch.Tensor] = None
        if "context_logvar_roll" in roll:
            context_logvar_roll = roll["context_logvar_roll"][0]  # [K, context_dim]

        row0 = belief["row_probs"][0]
        col0 = belief["col_probs"][0]
        hdg0 = belief["heading_probs"][0]

        current_entropy = self.state_entropy_from_belief(row0, col0, hdg0)

        ml_r0, ml_c0, ml_h0 = most_likely_state_from_belief(
            row0, col0, hdg0,
            self.reachable_mask.to(device),
        )

        ref_seq = build_reference_action_sequence(
            r=ml_r0, c=ml_c0, h=ml_h0,
            dist_t=self.dist_t.to(device),
            reachable_mask=self.reachable_mask.to(device),
            horizon=len(action_seq),
        )

        risk_sum = 0.0
        ambiguity_sum = 0.0
        collision_sum = 0.0
        info_gain_sum = 0.0
        action_cost_sum = 0.0
        inverse_cost_sum = 0.0
        ref_cost_sum = 0.0

        final_risk = 0.0
        prev_action: Optional[int] = None

        for k, a in enumerate(action_seq):
            rk = row_roll[k]
            ck = col_roll[k]
            hk = hdg_roll[k]

            discount = self.cfg.discount ** k

            risk = self.expected_risk(rk, ck)
            ambiguity = self.state_entropy_from_belief(rk, ck, hk)
            coll_prob = float(collision_probability_from_logits(coll_roll[k]).item())

            # info_gain rewards entropy reduction: model becomes more certain
            # about future states — good for goal-directed navigation.
            # (For pure curiosity-driven exploration you would reward high
            #  ambiguity instead, i.e. seek uncertain states.)
            info_gain = max(current_entropy - ambiguity, 0.0)

            risk_sum += discount * self.cfg.w_risk * risk
            ambiguity_sum += discount * self.cfg.w_ambiguity * ambiguity
            collision_sum += discount * self.cfg.w_collision * coll_prob
            info_gain_sum += discount * self.cfg.w_info_gain * info_gain

            # Context uncertainty epistemic bonus (V4)
            ctx_unc = self._context_uncertainty_bonus(context_logvar_roll, k)
            context_uncertainty_sum += discount * self.cfg.w_context_uncertainty * ctx_unc

            # Decoder-based epistemic bonus (optional, expensive)
            context_k = roll["context_roll"][0, k:k+1, :]
            dec_ep = self._decoder_epistemic_bonus(
                rk.unsqueeze(0), ck.unsqueeze(0), hk.unsqueeze(0), context_k
            )
            decoder_epistemic_sum += discount * self.cfg.w_decoder_epistemic * dec_ep

            final_risk = risk
            risk_values.append(risk)

            rr, cc, hh = most_likely_state_from_belief(
                rk, ck, hk,
                self.reachable_mask.to(device),
            )
            pred_state = (rr, cc, hh)

            if len(seen_pred_states) > 0 and pred_state == seen_pred_states[-1]:
                stay_cost_sum += self.cfg.imagination_same_state_penalty

            if pred_state in seen_pred_states:
                loop_cost_sum += self.cfg.imagination_loop_penalty

            seen_pred_states.append(pred_state)

            tail = recent_true_states[-6:]
            for idx, tstate in enumerate(tail):
                tr, tc, th = tstate
                th_idx = HEADING_TO_IDX[th]
                if pred_state == (tr, tc, th_idx):
                    if idx == len(tail) - 1:
                        recent_revisit_cost_sum += self.cfg.immediate_recent_state_penalty
                    else:
                        recent_revisit_cost_sum += self.cfg.recent_state_revisit_penalty

            if a == 0:
                action_cost_sum += self.cfg.cost_forward
            elif a == 1:
                action_cost_sum += self.cfg.cost_backward
            else:
                action_cost_sum += self.cfg.cost_turn

            if prev_action is not None:
                if (prev_action == 0 and a == 1) or (prev_action == 1 and a == 0):
                    inverse_cost_sum += self.cfg.inverse_fb_penalty
                if (prev_action == 2 and a == 3) or (prev_action == 3 and a == 2):
                    inverse_cost_sum += self.cfg.inverse_turn_penalty

            if k < len(ref_seq) and a != ref_seq[k]:
                ref_cost_sum += self.cfg.reference_prefix_penalty * (
                    self.cfg.reference_prefix_decay ** k
                )

            prev_action = a

        terminal_risk = self.cfg.w_terminal_risk * final_risk
        min_risk = self.cfg.w_min_risk * min(risk_values) if len(risk_values) > 0 else 0.0

        efe = (
            risk_sum
            + terminal_risk
            + min_risk
            + ambiguity_sum
            + collision_sum
            - info_gain_sum
            - context_uncertainty_sum   # attracted to uncertain regions
            - decoder_epistemic_sum     # attracted to info-rich regions
            + action_cost_sum
            + inverse_cost_sum
            + ref_cost_sum
            + loop_cost_sum
            + stay_cost_sum
            + recent_revisit_cost_sum
        )

        return {
            "sequence": list(action_seq),
            "score": float(efe),
            "efe": float(efe),
            "risk": float(risk_sum),
            "terminal_risk": float(terminal_risk),
            "ambiguity": float(ambiguity_sum),
            "collision": float(collision_sum),
            "info_gain": float(info_gain_sum),
            "context_uncertainty": float(context_uncertainty_sum),
            "decoder_epistemic": float(decoder_epistemic_sum),
            "action_cost": float(action_cost_sum),
            "inverse_cost": float(inverse_cost_sum),
            "ref_cost": float(ref_cost_sum),
            "ref_seq": ref_seq,
            "rollout": roll,
            "min_risk": float(min_risk),
            "loop_cost": float(loop_cost_sum),
            "stay_cost": float(stay_cost_sum),
            "recent_revisit_cost": float(recent_revisit_cost_sum),
        }

    @torch.no_grad()
    def score_action_sequences(
        self,
        belief: Dict[str, torch.Tensor],
        recent_true_states: List[Tuple[int, int, str]],
    ) -> Dict[str, Any]:
        device = belief["row_probs"].device

        r, c, h = most_likely_state_from_belief(
            belief["row_probs"][0],
            belief["col_probs"][0],
            belief["heading_probs"][0],
            self.reachable_mask.to(device),
        )

        ref_seq = build_reference_action_sequence(
            r=r, c=c, h=h,
            dist_t=self.dist_t.to(device),
            reachable_mask=self.reachable_mask.to(device),
            horizon=self.cfg.horizon,
        )

        if len(ref_seq) == 0:
            return {
                "reference_sequence": [],
                "best_sequence": [],
                "best_first_action": None,
                "best_score": math.inf,
                "all_details": [],
                "belief": belief,
            }

        candidates = self.enumerate_action_sequences_from_reference(ref_seq)

        all_details = []
        for seq in candidates:
            d = self.score_sequence(
                belief=belief,
                action_seq=seq,
                recent_true_states=recent_true_states,
            )
            if math.isfinite(d["score"]):
                all_details.append(d)

        if len(all_details) == 0:
            return {
                "reference_sequence": ref_seq,
                "best_sequence": [],
                "best_first_action": None,
                "best_score": math.inf,
                "all_details": [],
                "belief": belief,
            }

        all_details.sort(key=lambda x: x["score"])
        best = all_details[0]

        return {
            "reference_sequence": ref_seq,
            "best_sequence": best["sequence"],
            "best_first_action": int(best["sequence"][0]),
            "best_score": float(best["score"]),
            "all_details": all_details,
            "belief": belief,
        }

    @torch.no_grad()
    def score_action_sequences_pure(
        self,
        belief: Dict[str, torch.Tensor],
        recent_true_states: List[Tuple[int, int, str]],
    ) -> Dict[str, Any]:
        """
        Pure EFE scoring — enumerates action sequences directly without
        reference-path bias.
        """
        candidates = self.enumerate_all_action_sequences(self.cfg.horizon)

        if len(candidates) == 0:
            return {
                "reference_sequence": [],
                "best_sequence": [],
                "best_first_action": None,
                "best_score": math.inf,
                "all_details": [],
                "belief": belief,
            }

        all_details = []
        for seq in candidates:
            d = self.score_sequence(
                belief=belief,
                action_seq=seq,
                recent_true_states=recent_true_states,
            )
            if math.isfinite(d["score"]):
                all_details.append(d)

        if len(all_details) == 0:
            return {
                "reference_sequence": [],
                "best_sequence": [],
                "best_first_action": None,
                "best_score": math.inf,
                "all_details": [],
                "belief": belief,
            }

        all_details.sort(key=lambda x: x["score"])
        best = all_details[0]

        return {
            "reference_sequence": [],
            "best_sequence": best["sequence"],
            "best_first_action": int(best["sequence"][0]),
            "best_score": float(best["score"]),
            "all_details": all_details,
            "belief": belief,
        }

    @torch.no_grad()
    def plan(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor],
        recent_true_states: List[Tuple[int, int, str]],
    ) -> Dict[str, Any]:
        belief = self.infer_current_belief(
            observations=observations,
            actions=actions,
        )
        return self.score_action_sequences(
            belief=belief,
            recent_true_states=recent_true_states,
        )

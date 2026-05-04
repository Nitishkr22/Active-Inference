# aif_efe_planner_v1.py

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

    w_risk: float = 4.0
    w_terminal_risk: float = 10.0
    w_min_risk: float = 4.0
    w_ambiguity: float = 0.10
    w_collision: float = 18.0
    w_info_gain: float = 0.25

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


class EFEPlannerV1:
    """
    First active-inference-style planner.

    It uses your learned V3 rollout model, but scores policies using
    EFE-inspired terms:

        G(pi) = risk + ambiguity + collision - information_gain

    where:
      risk             = expected negative log preference
      ambiguity        = predicted state entropy
      information_gain = current entropy - future entropy
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

        Preferred states near goal get high probability.
        Risk uses -log preference.
        """
        device = dist_t.device

        pref_logits = -self.cfg.preference_precision * dist_t
        pref_logits = pref_logits.masked_fill(reachable_mask <= 0.5, -1e9)

        pref = torch.softmax(pref_logits.reshape(-1), dim=0).reshape_as(dist_t)
        pref = pref.clamp_min(self.cfg.preference_floor)

        neg_log_pref = -torch.log(pref)
        neg_log_pref = neg_log_pref.to(device)

        return neg_log_pref

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

        # one-change variants
        for i in range(H):
            for a in action_set:
                if a != ref_seq[i]:
                    seq = list(ref_seq)
                    seq[i] = a
                    candidates.add(tuple(seq))

        # adjacent two-change variants
        for i in range(H - 1):
            for a1 in action_set:
                for a2 in action_set:
                    seq = list(ref_seq)
                    seq[i] = a1
                    seq[i + 1] = a2
                    candidates.add(tuple(seq))

        # conservative variants
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
        """
        Pure EFE candidate generation.

        Does not use graph reference sequence.
        Enumerates action sequences directly.
        """
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

        device = belief["row_probs"].device

        roll = self.rollout_sequence(belief, action_seq)

        row_roll = roll["row_probs_roll"][0]
        col_roll = roll["col_probs_roll"][0]
        hdg_roll = roll["heading_probs_roll"][0]
        coll_roll = roll["collision_logits_roll"][0]

        row0 = belief["row_probs"][0]
        col0 = belief["col_probs"][0]
        hdg0 = belief["heading_probs"][0]

        current_entropy = self.state_entropy_from_belief(row0, col0, hdg0)

        ml_r0, ml_c0, ml_h0 = most_likely_state_from_belief(
            row0,
            col0,
            hdg0,
            self.reachable_mask.to(device),
        )

        ref_seq = build_reference_action_sequence(
            r=ml_r0,
            c=ml_c0,
            h=ml_h0,
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

            info_gain = max(current_entropy - ambiguity, 0.0)

            risk_sum += discount * self.cfg.w_risk * risk
            ambiguity_sum += discount * self.cfg.w_ambiguity * ambiguity
            collision_sum += discount * self.cfg.w_collision * coll_prob
            info_gain_sum += discount * self.cfg.w_info_gain * info_gain

            final_risk = risk

            risk_values.append(risk)

            rr, cc, hh = most_likely_state_from_belief(
                rk,
                ck,
                hk,
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

        # terminal_risk = self.cfg.w_terminal_risk * final_risk
        terminal_risk = self.cfg.w_terminal_risk * final_risk
        min_risk = self.cfg.w_min_risk * min(risk_values) if len(risk_values) > 0 else 0.0

        efe = (
            risk_sum
            + terminal_risk
            + min_risk
            + ambiguity_sum
            + collision_sum
            - info_gain_sum
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
            r=r,
            c=c,
            h=h,
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
        Pure EFE scoring.

        This does not build candidate sequences from graph reference.
        It enumerates action sequences directly.
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
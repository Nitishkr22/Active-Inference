# aif_planner_v3.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Sequence

import math
import torch
import torch.nn.functional as F


# ============================================================
# Action / heading constants
# ============================================================

ACTION_NAMES = ["forward", "backward", "turn_left", "turn_right"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_NAMES)}

HEADING_TO_IDX = {"N": 0, "E": 1, "S": 2, "W": 3}
IDX_TO_HEADING = {v: k for k, v in HEADING_TO_IDX.items()}

HEADING_DELTAS = {
    0: (-1, 0),   # N
    1: (0, 1),    # E
    2: (1, 0),    # S
    3: (0, -1),   # W
}


# ============================================================
# Config
# ============================================================

@dataclass
class AIFPlannerConfig:
    horizon: int = 5

    # reference-centered search
    allow_backward: bool = True
    max_candidates: int = 128

    # rollout scoring
    planning_discount: float = 0.85
    goal_cost_weight: float = 1.25
    terminal_goal_weight: float = 18.0
    min_goal_cost_weight: float = 7.0
    progress_bonus_weight: float = 7.5
    regress_penalty_weight: float = 9.0
    no_progress_penalty: float = 1.5
    collision_penalty: float = 10.0
    entropy_penalty: float = 0.05

    # action regularization
    cost_forward: float = 0.00
    cost_backward: float = 1.80
    cost_turn: float = 0.20
    inverse_fb_penalty: float = 4.00
    inverse_turn_penalty: float = 2.00
    heading_change_penalty: float = 0.10

    # loop penalties
    recent_state_revisit_penalty: float = 4.00
    immediate_recent_state_penalty: float = 7.00
    imagination_loop_penalty: float = 5.00
    imagination_same_state_penalty: float = 6.00

    # reference adherence
    reference_prefix_penalty: float = 2.5
    reference_prefix_decay: float = 0.75

    eps: float = 1e-8


# ============================================================
# Utility helpers
# ============================================================

def entropy_categorical(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def collision_probability_from_logits(collision_logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(collision_logits, dim=-1)
    return probs[..., 1]


def masked_joint_from_belief(
    row_probs: torch.Tensor,
    col_probs: torch.Tensor,
    reachable_mask: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    joint = torch.outer(row_probs, col_probs)
    joint = joint * reachable_mask
    valid_mass = float(joint.sum().item())
    if valid_mass > 1e-8:
        joint = joint / joint.sum()
    return joint, valid_mass


def expected_shortest_path_cost(
    row_probs: torch.Tensor,
    col_probs: torch.Tensor,
    dist_t: torch.Tensor,
    reachable_mask: torch.Tensor,
) -> float:
    joint, valid_mass = masked_joint_from_belief(row_probs, col_probs, reachable_mask)

    if valid_mass <= 1e-8:
        r = int(torch.argmax(row_probs).item())
        c = int(torch.argmax(col_probs).item())
        return float(dist_t[r, c].item())

    return float((joint * dist_t).sum().item())


def most_likely_state_from_belief(
    row_probs: torch.Tensor,
    col_probs: torch.Tensor,
    heading_probs: torch.Tensor,
    reachable_mask: torch.Tensor,
) -> Tuple[int, int, int]:
    joint, valid_mass = masked_joint_from_belief(row_probs, col_probs, reachable_mask)

    if valid_mass <= 1e-8:
        r = int(torch.argmax(row_probs).item())
        c = int(torch.argmax(col_probs).item())
    else:
        flat_idx = int(torch.argmax(joint).item())
        R, C = joint.shape
        r = flat_idx // C
        c = flat_idx % C

    h = int(torch.argmax(heading_probs).item())
    return r, c, h


def forward_cell(r: int, c: int, h: int) -> Tuple[int, int]:
    dr, dc = HEADING_DELTAS[h]
    return r + dr, c + dc


def backward_cell(r: int, c: int, h: int) -> Tuple[int, int]:
    dr, dc = HEADING_DELTAS[h]
    return r - dr, c - dc


def rollout_one_step_state(
    r: int,
    c: int,
    h: int,
    a: int,
    reachable_mask: torch.Tensor,
) -> Tuple[int, int, int]:
    R, C = reachable_mask.shape

    def valid(rr: int, cc: int) -> bool:
        return 0 <= rr < R and 0 <= cc < C and reachable_mask[rr, cc] > 0.5

    if a == 2:
        return r, c, (h - 1) % 4
    if a == 3:
        return r, c, (h + 1) % 4
    if a == 0:
        rr, cc = forward_cell(r, c, h)
        if valid(rr, cc):
            return rr, cc, h
        return r, c, h
    if a == 1:
        rr, cc = backward_cell(r, c, h)
        if valid(rr, cc):
            return rr, cc, h
        return r, c, h
    return r, c, h


def shortest_path_next_cell(
    r: int,
    c: int,
    dist_t: torch.Tensor,
    reachable_mask: torch.Tensor,
) -> Optional[Tuple[int, int]]:
    R, C = dist_t.shape
    cur = float(dist_t[r, c].item())

    best = None
    best_d = cur
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < R and 0 <= cc < C and reachable_mask[rr, cc] > 0.5:
            d = float(dist_t[rr, cc].item())
            if d < best_d:
                best_d = d
                best = (rr, cc)
    return best


def heading_for_move(r: int, c: int, rr: int, cc: int) -> int:
    dr, dc = rr - r, cc - c
    if (dr, dc) == (-1, 0):
        return 0
    if (dr, dc) == (0, 1):
        return 1
    if (dr, dc) == (1, 0):
        return 2
    if (dr, dc) == (0, -1):
        return 3
    raise ValueError(f"Invalid move from {(r, c)} to {(rr, cc)}")


def turn_sequence_to_heading(h: int, target_h: int) -> List[int]:
    left_steps = (h - target_h) % 4
    right_steps = (target_h - h) % 4
    if right_steps <= left_steps:
        return [3] * right_steps
    return [2] * left_steps


def build_reference_action_sequence(
    r: int,
    c: int,
    h: int,
    dist_t: torch.Tensor,
    reachable_mask: torch.Tensor,
    horizon: int,
) -> List[int]:
    ref: List[int] = []
    rr, cc, hh = r, c, h

    for _ in range(horizon * 3):
        if len(ref) >= horizon:
            break

        if float(dist_t[rr, cc].item()) <= 0.0:
            break

        nxt = shortest_path_next_cell(rr, cc, dist_t, reachable_mask)
        if nxt is None:
            break

        tr, tc = nxt
        desired_h = heading_for_move(rr, cc, tr, tc)

        turns = turn_sequence_to_heading(hh, desired_h)
        for a in turns:
            if len(ref) >= horizon:
                break
            ref.append(a)
            rr, cc, hh = rollout_one_step_state(rr, cc, hh, a, reachable_mask)

        if len(ref) >= horizon:
            break

        ref.append(0)
        rr, cc, hh = rollout_one_step_state(rr, cc, hh, 0, reachable_mask)

    return ref[:horizon]


def action_inverse(a1: int, a2: int) -> bool:
    return (
        (a1 == 0 and a2 == 1)
        or (a1 == 1 and a2 == 0)
        or (a1 == 2 and a2 == 3)
        or (a1 == 3 and a2 == 2)
    )


# ============================================================
# Planner
# ============================================================

class AIFPlannerV3:
    """
    Hybrid planner:
      1. build shortest-path reference from most likely filtered pose
      2. enumerate a small candidate set around that reference
      3. use learned rollout only to rank these nearby candidates
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dist_t: torch.Tensor,
        reachable_mask: torch.Tensor,
        cfg: Optional[AIFPlannerConfig] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg if cfg is not None else AIFPlannerConfig()

        self.dist_t = dist_t
        self.reachable_mask = reachable_mask

    # --------------------------------------------------------
    # Belief extraction
    # --------------------------------------------------------

    @torch.no_grad()
    def infer_current_belief(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if hasattr(self.model, "forward_filter"):
            outputs = self.model.forward_filter(observations, actions)
        elif hasattr(self.model, "filter_sequence"):
            outputs = self.model.filter_sequence(observations, actions)
        else:
            raise AttributeError(
                "Model must provide either forward_filter() or filter_sequence()."
            )

        belief = {
            "row_probs": outputs["row_probs_seq"][:, -1, :],
            "col_probs": outputs["col_probs_seq"][:, -1, :],
            "heading_probs": outputs["heading_probs_seq"][:, -1, :],
        }

        if "context_seq" in outputs:
            belief["context"] = outputs["context_seq"][:, -1, :]

        return belief

    # --------------------------------------------------------
    # Candidate generation
    # --------------------------------------------------------

    def enumerate_action_sequences_from_reference(
        self,
        ref_seq: List[int],
    ) -> List[List[int]]:
        H = len(ref_seq)
        candidates = set()

        # exact reference
        candidates.add(tuple(ref_seq))

        # one-change variants
        for i in range(H):
            for a in [0, 1, 2, 3]:
                if not self.cfg.allow_backward and a == 1:
                    continue
                if a != ref_seq[i]:
                    seq = list(ref_seq)
                    seq[i] = a
                    candidates.add(tuple(seq))

        # two-change adjacent variants
        for i in range(H - 1):
            for a1 in [0, 1, 2, 3]:
                for a2 in [0, 1, 2, 3]:
                    if not self.cfg.allow_backward and (a1 == 1 or a2 == 1):
                        continue
                    if a1 == ref_seq[i] and a2 == ref_seq[i + 1]:
                        continue
                    seq = list(ref_seq)
                    seq[i] = a1
                    seq[i + 1] = a2
                    candidates.add(tuple(seq))

        # some conservative variants
        if H >= 1:
            candidates.add(tuple(([0] + ref_seq[1:])[:H]))
            candidates.add(tuple(([2] + ref_seq[1:])[:H]))
            candidates.add(tuple(([3] + ref_seq[1:])[:H]))
        if self.cfg.allow_backward and H >= 1:
            candidates.add(tuple(([1] + ref_seq[1:])[:H]))
        if H >= 2:
            candidates.add(tuple(([ref_seq[0], ref_seq[0]] + ref_seq[2:])[:H]))

        cand_list = [list(x) for x in candidates]

        # deterministic ordering for readability
        cand_list.sort()

        if len(cand_list) > self.cfg.max_candidates:
            cand_list = cand_list[: self.cfg.max_candidates]

        return cand_list

    # --------------------------------------------------------
    # Score one sequence
    # --------------------------------------------------------

    @torch.no_grad()
    def score_sequence(
        self,
        belief: Dict[str, torch.Tensor],
        action_seq: Sequence[int],
        recent_true_states: List[Tuple[int, int, str]],
    ) -> Dict[str, Any]:
        device = belief["row_probs"].device

        seq_t = torch.tensor(action_seq, dtype=torch.long, device=device).unsqueeze(0)

        rollout_inputs = {
            "row_probs_start": belief["row_probs"],
            "col_probs_start": belief["col_probs"],
            "heading_probs_start": belief["heading_probs"],
            "action_seq": seq_t,
        }
        if "context" in belief:
            rollout_inputs["context_start"] = belief["context"]

        if hasattr(self.model, "rollout_from_filtered_state"):
            roll = self.model.rollout_from_filtered_state(
                row_probs_start=belief["row_probs"],
                col_probs_start=belief["col_probs"],
                heading_probs_start=belief["heading_probs"],
                context_start=belief.get("context", None),
                action_seq=seq_t,
            )
        elif hasattr(self.model, "rollout_from_belief"):
            roll = self.model.rollout_from_belief(**rollout_inputs)
        else:
            raise AttributeError(
                "Model must provide rollout_from_filtered_state() or rollout_from_belief()."
            )

        row_roll = roll["row_probs_roll"][0]
        col_roll = roll["col_probs_roll"][0]
        hdg_roll = roll["heading_probs_roll"][0]
        coll_roll = roll["collision_logits_roll"][0]

        row0 = belief["row_probs"][0]
        col0 = belief["col_probs"][0]
        hdg0 = belief["heading_probs"][0]

        current_cost = expected_shortest_path_cost(
            row0, col0, self.dist_t.to(device), self.reachable_mask.to(device)
        )

        ml_r0, ml_c0, ml_h0 = most_likely_state_from_belief(
            row0, col0, hdg0, self.reachable_mask.to(device)
        )
        ref_seq = build_reference_action_sequence(
            r=ml_r0,
            c=ml_c0,
            h=ml_h0,
            dist_t=self.dist_t.to(device),
            reachable_mask=self.reachable_mask.to(device),
            horizon=len(action_seq),
        )

        goal_cost_sum = 0.0
        terminal_goal_cost = 0.0
        min_goal_cost_term = 0.0
        progress_bonus_sum = 0.0
        regress_penalty_sum = 0.0
        collision_cost_sum = 0.0
        entropy_cost_sum = 0.0
        action_cost_sum = 0.0
        reverse_cost_sum = 0.0
        heading_cost_sum = 0.0
        loop_cost_sum = 0.0
        ref_cost_sum = 0.0

        prev_action: Optional[int] = None
        seen_pred_states: List[Tuple[int, int, int]] = []
        min_cost_seen = current_cost
        final_cost = current_cost

        for k, a in enumerate(action_seq):
            rk = row_roll[k]
            ck = col_roll[k]
            hk = hdg_roll[k]

            next_cost = expected_shortest_path_cost(
                rk, ck, self.dist_t.to(device), self.reachable_mask.to(device)
            )

            # discounted path cost
            step_discount = self.cfg.planning_discount ** k
            goal_cost_sum += step_discount * self.cfg.goal_cost_weight * next_cost

            # progress shaping
            delta = current_cost - next_cost
            if delta > 1e-6:
                progress_bonus_sum += self.cfg.progress_bonus_weight * delta
            elif delta < -1e-6:
                regress_penalty_sum += self.cfg.regress_penalty_weight * (-delta)
            else:
                regress_penalty_sum += self.cfg.no_progress_penalty

            current_cost = next_cost
            final_cost = next_cost
            min_cost_seen = min(min_cost_seen, next_cost)

            # collision
            coll_prob = float(collision_probability_from_logits(coll_roll[k]).item())
            collision_cost_sum += self.cfg.collision_penalty * coll_prob

            # entropy
            ent = (
                float(entropy_categorical(rk, self.cfg.eps).item())
                + float(entropy_categorical(ck, self.cfg.eps).item())
                + float(entropy_categorical(hk, self.cfg.eps).item())
            )
            entropy_cost_sum += self.cfg.entropy_penalty * ent

            # action costs
            if a == 0:
                action_cost_sum += self.cfg.cost_forward
            elif a == 1:
                action_cost_sum += self.cfg.cost_backward
            else:
                action_cost_sum += self.cfg.cost_turn

            # inverse penalties
            if prev_action is not None:
                if (prev_action == 0 and a == 1) or (prev_action == 1 and a == 0):
                    reverse_cost_sum += self.cfg.inverse_fb_penalty
                if (prev_action == 2 and a == 3) or (prev_action == 3 and a == 2):
                    reverse_cost_sum += self.cfg.inverse_turn_penalty

            if a in [2, 3]:
                heading_cost_sum += self.cfg.heading_change_penalty

            # predicted argmax state
            rr, cc, hh = most_likely_state_from_belief(
                rk, ck, hk, self.reachable_mask.to(device)
            )
            pred_state = (rr, cc, hh)

            if len(seen_pred_states) > 0 and pred_state == seen_pred_states[-1]:
                loop_cost_sum += self.cfg.imagination_same_state_penalty

            if pred_state in seen_pred_states:
                loop_cost_sum += self.cfg.imagination_loop_penalty

            seen_pred_states.append(pred_state)

            tail = recent_true_states[-6:]
            for idx, tstate in enumerate(tail):
                tr, tc, th = tstate
                th_idx = HEADING_TO_IDX[th]
                if (rr, cc, hh) == (tr, tc, th_idx):
                    if idx == len(tail) - 1:
                        loop_cost_sum += self.cfg.immediate_recent_state_penalty
                    else:
                        loop_cost_sum += self.cfg.recent_state_revisit_penalty

            if k < len(ref_seq) and a != ref_seq[k]:
                ref_cost_sum += self.cfg.reference_prefix_penalty * (
                    self.cfg.reference_prefix_decay ** k
                )

            prev_action = a

        terminal_goal_cost = self.cfg.terminal_goal_weight * final_cost
        min_goal_cost_term = self.cfg.min_goal_cost_weight * min_cost_seen

        score = (
            goal_cost_sum
            + terminal_goal_cost
            + min_goal_cost_term
            - progress_bonus_sum
            + regress_penalty_sum
            + collision_cost_sum
            + entropy_cost_sum
            + action_cost_sum
            + reverse_cost_sum
            + heading_cost_sum
            + loop_cost_sum
            + ref_cost_sum
        )

        return {
            "sequence": list(action_seq),
            "score": float(score),
            "goal": float(goal_cost_sum),
            "term": float(terminal_goal_cost),
            "min_goal": float(min_goal_cost_term),
            "prog": float(progress_bonus_sum),
            "reg": float(regress_penalty_sum),
            "coll": float(collision_cost_sum),
            "ent": float(entropy_cost_sum),
            "act": float(action_cost_sum),
            "rev": float(reverse_cost_sum),
            "hdg": float(heading_cost_sum),
            "loop": float(loop_cost_sum),
            "ref": float(ref_cost_sum),
            "ref_seq": ref_seq,
            "final_cost": float(final_cost),
            "min_cost_seen": float(min_cost_seen),
            "rollout": roll,
        }

    # --------------------------------------------------------
    # Score candidate set
    # --------------------------------------------------------

    @torch.no_grad()
    def score_action_sequences(
        self,
        belief: Dict[str, torch.Tensor],
        recent_true_states: List[Tuple[int, int, str]],
    ) -> Dict[str, Any]:
        device = belief["row_probs"].device

        ml_r0, ml_c0, ml_h0 = most_likely_state_from_belief(
            belief["row_probs"][0],
            belief["col_probs"][0],
            belief["heading_probs"][0],
            self.reachable_mask.to(device),
        )

        ref_seq = build_reference_action_sequence(
            r=ml_r0,
            c=ml_c0,
            h=ml_h0,
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
            }

        candidates = self.enumerate_action_sequences_from_reference(ref_seq)

        all_details: List[Dict[str, Any]] = []
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
            }

        all_details.sort(key=lambda x: x["score"])
        best = all_details[0]

        return {
            "reference_sequence": ref_seq,
            "best_sequence": best["sequence"],
            "best_first_action": int(best["sequence"][0]),
            "best_score": float(best["score"]),
            "all_details": all_details,
        }

    # --------------------------------------------------------
    # Main planning entry
    # --------------------------------------------------------

    @torch.no_grad()
    def plan(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor],
        recent_true_states: List[Tuple[int, int, str]],
    ) -> Dict[str, Any]:
        belief = self.infer_current_belief(observations=observations, actions=actions)
        scored = self.score_action_sequences(
            belief=belief,
            recent_true_states=recent_true_states,
        )
        scored["belief"] = belief
        return scored
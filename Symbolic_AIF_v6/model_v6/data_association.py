"""
data_association.py — Hungarian algorithm matching of detections to map slots.

Cost function:
  cost(detection_i, slot_j) = w_pos * mahalanobis_3d(det.pos, slot.pos_mu, slot.pos_logvar)
                             + w_cls * class_mismatch(det.class_id, slot.class_logits)

The Hungarian algorithm finds the minimum-cost bipartite matching.
Pairs whose cost exceeds cfg.assoc.threshold are discarded (unmatched).

Returns three lists:
  matched_pairs  : [(det_idx, slot_idx), ...]   — cost-feasible matches
  unmatched_dets : [det_idx, ...]               — detections with no slot match
  unmatched_slots: [slot_idx, ...]              — occupied slots not seen this step
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .belief_state import BeliefState, Detection
from .config import AssociationConfig, SlotConfig


def associate(
    detections: List[Detection],
    belief: BeliefState,
    assoc_cfg: AssociationConfig,
    slot_cfg: SlotConfig,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match current detections to occupied map slots.

    Only slots above assoc_cfg.min_conf_logit are considered candidates.
    Unoccupied slots are excluded and returned in unmatched_slots.
    """
    if not detections:
        occupied_idxs = _occupied_slot_idxs(belief, assoc_cfg)
        return [], [], list(occupied_idxs)

    occupied_idxs = _occupied_slot_idxs(belief, assoc_cfg)

    if len(occupied_idxs) == 0:
        # No existing slots — all detections are new
        return [], list(range(len(detections))), []

    D = len(detections)
    S = len(occupied_idxs)

    # ---- Build cost matrix [D × S] ----
    cost = np.full((D, S), fill_value=1e6, dtype=np.float32)

    slot_pos_mu  = belief.slot_pos_mu[occupied_idxs]       # [S, 3]
    slot_pos_var = torch.exp(belief.slot_pos_logvar[occupied_idxs])  # [S, 3]
    slot_cls     = belief.slot_class_logits[occupied_idxs]  # [S, C]
    slot_probs   = torch.softmax(slot_cls, dim=-1)           # [S, C]

    for di, det in enumerate(detections):
        det_pos = det.pos_world.unsqueeze(0)  # [1, 3]

        # Mahalanobis distance in 3D (use only x, y for 2D navigation safety;
        # include z with lower weight to handle height uncertainty)
        diff = det_pos - slot_pos_mu           # [S, 3]
        mah  = (diff ** 2 / (slot_pos_var + 1e-6)).sum(dim=-1)  # [S]
        mah  = torch.sqrt(mah.clamp_min(0.0))                    # [S]

        # Class mismatch: 1 - P(det.class_id | slot)
        cls_match = slot_probs[:, det.class_id]    # [S]
        cls_cost  = 1.0 - cls_match                # [S]

        total = (assoc_cfg.w_pos * mah + assoc_cfg.w_cls * cls_cost).cpu().numpy()
        cost[di, :] = total

    # ---- Hungarian algorithm ----
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pairs:   List[Tuple[int, int]] = []
    matched_det_set  = set()
    matched_slot_set = set()

    for ri, ci in zip(row_ind.tolist(), col_ind.tolist()):
        if cost[ri, ci] <= assoc_cfg.threshold:
            matched_pairs.append((ri, int(occupied_idxs[ci])))
            matched_det_set.add(ri)
            matched_slot_set.add(ci)

    unmatched_dets  = [i for i in range(D)  if i not in matched_det_set]
    unmatched_slots = [int(occupied_idxs[j]) for j in range(S) if j not in matched_slot_set]

    return matched_pairs, unmatched_dets, unmatched_slots


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #

def _occupied_slot_idxs(belief: BeliefState, assoc_cfg: AssociationConfig) -> torch.Tensor:
    """Indices of slots above the minimum confidence threshold."""
    mask = belief.slot_conf_logit > assoc_cfg.min_conf_logit
    return mask.nonzero(as_tuple=False).squeeze(1)

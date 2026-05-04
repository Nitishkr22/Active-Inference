# latent_efe_planner_v1.py

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


ACTION_NAMES = {
    0: "forward",
    1: "backward",
    2: "turn_left",
    3: "turn_right",
}


# ============================================================
# Configs
# ============================================================

@dataclass
class OnlineVFEConfig:
    enabled: bool = True
    num_steps: int = 3
    lr: float = 0.15
    entropy_weight: float = 0.02
    unreachable_weight: float = 5.0
    eps: float = 1e-8


@dataclass
class LatentEFEConfig:
    horizon: int = 5
    allow_backward: bool = False
    max_candidates: int = 128

    # Latent preference terms
    w_latent_risk: float = 12.0
    w_terminal_latent_risk: float = 25.0

    # Graph/path anchoring terms
    # Set these to 0 for pure latent planning.
    w_graph_path: float = 4.0
    w_terminal_graph_path: float = 8.0
    w_graph_progress: float = 40.0

    # Safety / uncertainty
    w_collision: float = 30.0
    w_entropy: float = 0.05
    w_info_gain: float = 1.0
    w_wall_mass: float = 20.0
    w_no_progress: float = 12.0

    # Action regularization
    w_action: float = 0.20
    w_backward: float = 1.50
    w_turn: float = 0.20
    w_inverse: float = 8.00
    w_context_smoothness: float = 0.25

    discount: float = 0.90
    eps: float = 1e-8


# ============================================================
# Helper functions
# ============================================================

def enumerate_action_sequences(
    horizon: int,
    allow_backward: bool = False,
    max_candidates: int = 128,
) -> List[List[int]]:
    actions = [0, 1, 2, 3] if allow_backward else [0, 2, 3]
    seqs = [list(s) for s in product(actions, repeat=horizon)]

    def priority(seq: List[int]) -> Tuple[int, int, int]:
        n_forward = sum(a == 0 for a in seq)
        n_backward = sum(a == 1 for a in seq)
        n_turn = sum(a in [2, 3] for a in seq)
        return (-n_forward, n_backward, n_turn)

    seqs.sort(key=priority)
    return seqs[:max_candidates]


def categorical_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def collision_prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)[..., 1]


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(a, b, dim=-1)


def inverse_action(a: int, b: int) -> bool:
    return (
        (a == 0 and b == 1)
        or (a == 1 and b == 0)
        or (a == 2 and b == 3)
        or (a == 3 and b == 2)
    )


def get_context_rollout(roll: Dict[str, torch.Tensor]) -> torch.Tensor:
    for key in ["context_roll", "context_rollout", "context_probs_roll", "context_seq_roll"]:
        if key in roll:
            return roll[key]
    raise KeyError(
        "No context rollout found. Expected one of: "
        "context_roll, context_rollout, context_probs_roll, context_seq_roll"
    )


def refine_belief_online_vfe(
    model: torch.nn.Module,
    belief: Dict[str, torch.Tensor],
    current_obs: torch.Tensor,          # [B,1,64,64]
    cfg: OnlineVFEConfig,
    reachable_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:

    if not cfg.enabled:
        return belief

    with torch.enable_grad():
        row_logits = torch.log(belief["row_probs"].detach().clamp_min(cfg.eps)).requires_grad_(True)
        col_logits = torch.log(belief["col_probs"].detach().clamp_min(cfg.eps)).requires_grad_(True)
        hdg_logits = torch.log(belief["heading_probs"].detach().clamp_min(cfg.eps)).requires_grad_(True)
        context = belief["context"].detach().clone().requires_grad_(True)

        row_prior = belief["row_probs"].detach()
        col_prior = belief["col_probs"].detach()
        hdg_prior = belief["heading_probs"].detach()
        ctx_prior = belief["context"].detach()

        current_obs = current_obs.detach()

        for _ in range(cfg.num_steps):
            row_probs = torch.softmax(row_logits, dim=-1)
            col_probs = torch.softmax(col_logits, dim=-1)
            hdg_probs = torch.softmax(hdg_logits, dim=-1)

            recon = model.decoder(row_probs, col_probs, hdg_probs, context)

            recon_loss = F.mse_loss(recon, current_obs)

            kl_row = (
                row_probs
                * (row_probs.clamp_min(cfg.eps).log() - row_prior.clamp_min(cfg.eps).log())
            ).sum(dim=-1).mean()

            kl_col = (
                col_probs
                * (col_probs.clamp_min(cfg.eps).log() - col_prior.clamp_min(cfg.eps).log())
            ).sum(dim=-1).mean()

            kl_hdg = (
                hdg_probs
                * (hdg_probs.clamp_min(cfg.eps).log() - hdg_prior.clamp_min(cfg.eps).log())
            ).sum(dim=-1).mean()

            ctx_reg = F.mse_loss(context, ctx_prior)

            entropy = (
                categorical_entropy(row_probs, cfg.eps)
                + categorical_entropy(col_probs, cfg.eps)
                + categorical_entropy(hdg_probs, cfg.eps)
            ).mean()

            unreachable_loss = torch.tensor(0.0, device=current_obs.device)

            if reachable_mask is not None:
                reachable = reachable_mask.to(device=current_obs.device, dtype=row_probs.dtype)
                joint_rc = row_probs.unsqueeze(-1) * col_probs.unsqueeze(-2)
                unreachable_mass = (
                    joint_rc * (1.0 - reachable).unsqueeze(0)
                ).sum(dim=(-2, -1)).mean()
                unreachable_loss = cfg.unreachable_weight * unreachable_mass

            vfe_loss = (
                recon_loss
                + kl_row
                + kl_col
                + kl_hdg
                + ctx_reg
                + cfg.entropy_weight * entropy
                + unreachable_loss
            )

            grads = torch.autograd.grad(
                vfe_loss,
                [row_logits, col_logits, hdg_logits, context],
                retain_graph=False,
                create_graph=False,
            )

            with torch.no_grad():
                row_logits -= cfg.lr * grads[0]
                col_logits -= cfg.lr * grads[1]
                hdg_logits -= cfg.lr * grads[2]
                context -= cfg.lr * grads[3]

            row_logits.requires_grad_(True)
            col_logits.requires_grad_(True)
            hdg_logits.requires_grad_(True)
            context.requires_grad_(True)

        refined = dict(belief)
        refined["row_probs"] = torch.softmax(row_logits.detach(), dim=-1)
        refined["col_probs"] = torch.softmax(col_logits.detach(), dim=-1)
        refined["heading_probs"] = torch.softmax(hdg_logits.detach(), dim=-1)
        refined["context"] = context.detach()
        refined["vfe_loss"] = float(vfe_loss.detach().item())
        refined["vfe_recon_loss"] = float(recon_loss.detach().item())

        return refined


# ============================================================
# Planner
# ============================================================

class LatentEFEPlannerV1:
    def __init__(
        self,
        model: torch.nn.Module,
        dist_t: torch.Tensor,
        reachable_mask: Optional[torch.Tensor] = None,
        cfg: Optional[LatentEFEConfig] = None,
        vfe_cfg: Optional[OnlineVFEConfig] = None,
    ) -> None:
        self.model = model
        self.dist_t = dist_t
        self.reachable_mask = reachable_mask
        self.cfg = cfg if cfg is not None else LatentEFEConfig()
        self.vfe_cfg = vfe_cfg if vfe_cfg is not None else OnlineVFEConfig(enabled=True)

        self.candidate_action_sequences = enumerate_action_sequences(
            horizon=self.cfg.horizon,
            allow_backward=self.cfg.allow_backward,
            max_candidates=self.cfg.max_candidates,
        )

    # @torch.no_grad()
    def infer_current_belief(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:

        if actions is None:
            B = observations.shape[0]
            T = observations.shape[1]

            if T < 2:
                observations = torch.cat([observations, observations], dim=1)
                actions = torch.zeros((B, 1), dtype=torch.long, device=observations.device)
            else:
                actions = torch.zeros((B, T - 1), dtype=torch.long, device=observations.device)

        out = self.model.forward_filter(observations, actions)

        if "context_seq" not in out:
            raise KeyError("Model output does not contain context_seq.")

        belief = {
            "row_probs": out["row_probs_seq"][:, -1, :],
            "col_probs": out["col_probs_seq"][:, -1, :],
            "heading_probs": out["heading_probs_seq"][:, -1, :],
            "context": out["context_seq"][:, -1, :],
            "raw_filter_out": out,
        }

        current_obs = observations[:, -1, :, :, :]  # [B,1,64,64]

        belief = refine_belief_online_vfe(
            model=self.model,
            belief=belief,
            current_obs=current_obs,
            cfg=self.vfe_cfg,
            reachable_mask=self.reachable_mask,
        )

        return belief

    @torch.no_grad()
    def score_action_sequences(
        self,
        belief: Dict[str, torch.Tensor],
        z_goal: torch.Tensor,
    ) -> Dict[str, Any]:

        device = belief["context"].device
        dtype = belief["context"].dtype

        candidate_seqs = self.candidate_action_sequences
        N = len(candidate_seqs)
        H = self.cfg.horizon

        action_seq = torch.tensor(candidate_seqs, dtype=torch.long, device=device)

        row0 = belief["row_probs"].repeat(N, 1)
        col0 = belief["col_probs"].repeat(N, 1)
        hdg0 = belief["heading_probs"].repeat(N, 1)
        ctx0 = belief["context"].repeat(N, 1)

        if z_goal.ndim == 2:
            z_goal = z_goal[0]
        z_goal = z_goal.to(device=device, dtype=dtype)

        roll = self.model.rollout_from_filtered_state(
            row_probs_start=row0,
            col_probs_start=col0,
            heading_probs_start=hdg0,
            context_start=ctx0,
            action_seq=action_seq,
        )

        ctx_roll = get_context_rollout(roll)
        row_roll = roll["row_probs_roll"]
        col_roll = roll["col_probs_roll"]
        hdg_roll = roll["heading_probs_roll"]
        coll_logits = roll["collision_logits_roll"]

        discounts = torch.tensor(
            [self.cfg.discount ** k for k in range(H)],
            device=device,
            dtype=dtype,
        ).view(1, H)

        # -----------------------------
        # Latent risk
        # -----------------------------
        z_goal_expand = z_goal.view(1, 1, -1).expand_as(ctx_roll)
        latent_dist = cosine_distance(ctx_roll, z_goal_expand)

        latent_risk = (latent_dist * discounts).sum(dim=1)
        terminal_risk = latent_dist[:, -1]

        # -----------------------------
        # Entropy / ambiguity
        # -----------------------------
        row_ent = categorical_entropy(row_roll, eps=self.cfg.eps)
        col_ent = categorical_entropy(col_roll, eps=self.cfg.eps)
        hdg_ent = categorical_entropy(hdg_roll, eps=self.cfg.eps)
        state_ent = row_ent + col_ent + hdg_ent
        entropy_cost = (state_ent * discounts).sum(dim=1)

        current_ent = (
            categorical_entropy(belief["row_probs"], eps=self.cfg.eps)
            + categorical_entropy(belief["col_probs"], eps=self.cfg.eps)
            + categorical_entropy(belief["heading_probs"], eps=self.cfg.eps)
        )

        final_ent = state_ent[:, -1]
        info_gain = (current_ent.view(1) - final_ent).clamp_min(0.0)

        # -----------------------------
        # Collision risk
        # -----------------------------
        coll_prob = collision_prob_from_logits(coll_logits)
        collision_cost = (coll_prob * discounts).sum(dim=1)

        # -----------------------------
        # Graph/path anchoring
        # -----------------------------
        joint_rc = row_roll.unsqueeze(-1) * col_roll.unsqueeze(-2)

        dist = self.dist_t.to(device=device, dtype=dtype)
        dist = dist / dist.max().clamp_min(self.cfg.eps)

        graph_cost_per_step = (
            joint_rc * dist.view(1, 1, *dist.shape)
        ).sum(dim=(-2, -1))

        graph_cost = (graph_cost_per_step * discounts).sum(dim=1)
        terminal_graph_cost = graph_cost_per_step[:, -1]

        row0_joint = belief["row_probs"].unsqueeze(-1) * belief["col_probs"].unsqueeze(-2)
        current_graph_cost = (
            row0_joint * dist.view(1, *dist.shape)
        ).sum(dim=(-2, -1))

        graph_progress = (
            current_graph_cost.view(1) - terminal_graph_cost
        ).clamp_min(0.0)

        no_progress_penalty = (graph_progress <= 1e-5).to(dtype)

        # -----------------------------
        # Wall mass penalty
        # -----------------------------
        if self.reachable_mask is not None:
            reachable = self.reachable_mask.to(device=device, dtype=dtype)
            wall_mass_per_step = (
                joint_rc * (1.0 - reachable).view(1, 1, *reachable.shape)
            ).sum(dim=(-2, -1))
            wall_mass_cost = (wall_mass_per_step * discounts).sum(dim=1)
        else:
            wall_mass_cost = torch.zeros(N, device=device, dtype=dtype)

        # -----------------------------
        # Latent smoothness
        # -----------------------------
        ctx_prev = torch.cat([ctx0[:, None, :], ctx_roll[:, :-1, :]], dim=1)
        smooth_cost = F.mse_loss(ctx_roll, ctx_prev, reduction="none").mean(dim=-1)
        smooth_cost = (smooth_cost * discounts).sum(dim=1)

        # -----------------------------
        # Action regularization
        # -----------------------------
        action_cost = torch.zeros(N, device=device, dtype=dtype)
        inverse_cost = torch.zeros(N, device=device, dtype=dtype)

        for i, seq in enumerate(candidate_seqs):
            c = 0.0
            inv = 0.0
            prev = None

            for a in seq:
                c += self.cfg.w_action

                if a == 1:
                    c += self.cfg.w_backward
                elif a in [2, 3]:
                    c += self.cfg.w_turn

                if prev is not None and inverse_action(prev, a):
                    inv += self.cfg.w_inverse

                prev = a

            action_cost[i] = c
            inverse_cost[i] = inv

        # -----------------------------
        # Total EFE-like score
        # -----------------------------
        total_score = (
            self.cfg.w_latent_risk * latent_risk
            + self.cfg.w_terminal_latent_risk * terminal_risk
            + self.cfg.w_graph_path * graph_cost
            + self.cfg.w_terminal_graph_path * terminal_graph_cost
            - self.cfg.w_graph_progress * graph_progress
            + self.cfg.w_wall_mass * wall_mass_cost
            + self.cfg.w_no_progress * no_progress_penalty
            + self.cfg.w_collision * collision_cost
            + self.cfg.w_entropy * entropy_cost
            - self.cfg.w_info_gain * info_gain
            + action_cost
            + inverse_cost
            + self.cfg.w_context_smoothness * smooth_cost
        )

        best_idx = int(torch.argmin(total_score).item())
        best_seq = candidate_seqs[best_idx]
        best_first_action = int(best_seq[0])

        details: List[Dict[str, Any]] = []
        for i, seq in enumerate(candidate_seqs):
            details.append(
                {
                    "sequence": seq,
                    "score": float(total_score[i].item()),
                    "latent_risk": float(latent_risk[i].item()),
                    "terminal_risk": float(terminal_risk[i].item()),
                    "graph": float(graph_cost[i].item()),
                    "terminal_graph": float(terminal_graph_cost[i].item()),
                    "graph_progress": float(graph_progress[i].item()),
                    "wall_mass": float(wall_mass_cost[i].item()),
                    "no_progress": float(no_progress_penalty[i].item()),
                    "collision": float(collision_cost[i].item()),
                    "entropy": float(entropy_cost[i].item()),
                    "info_gain": float(info_gain[i].item()),
                    "action": float(action_cost[i].item()),
                    "inverse": float(inverse_cost[i].item()),
                    "smooth": float(smooth_cost[i].item()),
                    "final_latent_dist": float(latent_dist[i, -1].item()),
                }
            )

        details.sort(key=lambda x: x["score"])

        return {
            "best_sequence": best_seq,
            "best_first_action": best_first_action,
            "best_score": float(total_score[best_idx].item()),
            "all_details": details,
            "rollout": roll,
            "action_seq_tensor": action_seq,
        }

    def plan(
        self,
        observations: torch.Tensor,
        actions: Optional[torch.Tensor],
        z_goal: torch.Tensor,
    ) -> Dict[str, Any]:
        belief = self.infer_current_belief(observations, actions)
        scored = self.score_action_sequences(
            belief=belief,
            z_goal=z_goal,
        )
        scored["belief"] = belief
        return scored
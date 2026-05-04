# online_vfe_refinement_v1.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class OnlineVFEConfig:
    enabled: bool = True
    num_iters: int = 8
    lr: float = 0.05

    # keep refined belief close to neural filter output
    w_kl: float = 1.0

    # sharpen uncertain belief
    w_entropy: float = 0.05

    # discourage impossible wall states
    w_invalid: float = 3.0

    # keep context close to original context
    w_context: float = 0.05

    eps: float = 1e-8


def categorical_entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)


def categorical_kl(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    q = q.clamp_min(eps)
    p = p.clamp_min(eps)
    return (q * (q.log() - p.log())).sum(dim=-1)


def refine_belief_online_vfe(
    belief: Dict[str, torch.Tensor],
    reachable_mask: Optional[torch.Tensor] = None,
    cfg: Optional[OnlineVFEConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Online VFE-like belief refinement.

    This does NOT retrain the model.
    It refines the current amortized posterior from the neural filter.

    Refined variables:
      q(row), q(col), q(heading), context

    Objective:
      F ≈ KL(q_refined || q_filter)
          + entropy(q_refined)
          + invalid-map-mass
          + context-drift
    """

    if cfg is None:
        cfg = OnlineVFEConfig()

    if not cfg.enabled:
        belief["vfe_loss"] = torch.tensor(0.0, device=belief["context"].device)
        return belief

    row_init = belief["row_probs"].detach()
    col_init = belief["col_probs"].detach()
    hdg_init = belief["heading_probs"].detach()
    ctx_init = belief["context"].detach()

    device = ctx_init.device
    dtype = ctx_init.dtype

    row_logits = torch.log(row_init.clamp_min(cfg.eps)).detach().clone().requires_grad_(True)
    col_logits = torch.log(col_init.clamp_min(cfg.eps)).detach().clone().requires_grad_(True)
    hdg_logits = torch.log(hdg_init.clamp_min(cfg.eps)).detach().clone().requires_grad_(True)
    ctx = ctx_init.detach().clone().requires_grad_(True)

    params = [row_logits, col_logits, hdg_logits, ctx]
    opt = torch.optim.Adam(params, lr=cfg.lr)

    if reachable_mask is not None:
        reachable_mask = reachable_mask.to(device=device, dtype=dtype)

    final_loss = None

    for _ in range(cfg.num_iters):
        opt.zero_grad()

        row_q = F.softmax(row_logits, dim=-1)
        col_q = F.softmax(col_logits, dim=-1)
        hdg_q = F.softmax(hdg_logits, dim=-1)

        kl_loss = (
            categorical_kl(row_q, row_init, cfg.eps)
            + categorical_kl(col_q, col_init, cfg.eps)
            + categorical_kl(hdg_q, hdg_init, cfg.eps)
        ).mean()

        entropy_loss = (
            categorical_entropy(row_q, cfg.eps)
            + categorical_entropy(col_q, cfg.eps)
            + categorical_entropy(hdg_q, cfg.eps)
        ).mean()

        if reachable_mask is not None:
            joint = row_q.unsqueeze(-1) * col_q.unsqueeze(-2)  # [B,R,C]
            invalid_mass = (joint * (1.0 - reachable_mask).unsqueeze(0)).sum(dim=(-2, -1)).mean()
        else:
            invalid_mass = torch.tensor(0.0, device=device, dtype=dtype)

        context_loss = F.mse_loss(ctx, ctx_init)

        loss = (
            cfg.w_kl * kl_loss
            + cfg.w_entropy * entropy_loss
            + cfg.w_invalid * invalid_mass
            + cfg.w_context * context_loss
        )

        loss.backward()
        opt.step()
        final_loss = loss.detach()

    refined = dict(belief)
    refined["row_probs"] = F.softmax(row_logits.detach(), dim=-1)
    refined["col_probs"] = F.softmax(col_logits.detach(), dim=-1)
    refined["heading_probs"] = F.softmax(hdg_logits.detach(), dim=-1)
    refined["context"] = ctx.detach()

    refined["vfe_loss"] = final_loss if final_loss is not None else torch.tensor(0.0, device=device)
    refined["vfe_refined"] = True

    return refined
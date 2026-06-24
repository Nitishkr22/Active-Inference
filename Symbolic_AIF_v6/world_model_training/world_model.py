"""
world_model_training/world_model.py — RSSM-style generative world model.

Architecture
============

Encoder (per-frame)
-------------------
RGB  [3, H, W]   → CNN → flat features
Depth[1, H, W]   → CNN → flat features
concat → MLP → μ_post [z_dim], logvar_post [z_dim]
z ~ N(μ_post, exp(logvar_post))               # posterior / recognition model q(z|o)

Recurrent core (GRU)
---------------------
(h_{t-1}, z_{t-1}, a_{t-1}) → GRU → h_t     # deterministic state (memory)

Prior
-----
h_t → MLP → μ_prior [z_dim], logvar_prior [z_dim]   # p(z_t | h_t)

Decoder (per-frame)
-------------------
(h_t, z_t) → deconv → predicted_RGB [3, H, W]
(h_t, z_t) → deconv → predicted_depth [1, H, W]

Occupancy head
--------------
(h_t, z_t) → MLP → local_occupancy [grid_size, grid_size]   # binary, sigmoid output

Training objective (VFE)
========================
L = E_q[
      -log p(o_RGB | h, z)        # RGB reconstruction (MSE)
      -log p(o_depth | h, z)      # depth reconstruction (MSE)
      -log p(occ | h, z)          # occupancy BCE
      + KL( q(z|o) || p(z|h) )   # prior regularisation
    ]
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig, EncoderConfig, RSSMConfig, DecoderConfig, OccupancyHeadConfig


# ── helpers ──────────────────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int, *, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.ELU(inplace=True),
    )


def _deconv_block(in_ch: int, out_ch: int, *, stride: int = 2) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=False),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.ELU(inplace=True),
    )


def kl_divergence(
    mu_post: torch.Tensor,
    logvar_post: torch.Tensor,
    mu_prior: torch.Tensor,
    logvar_prior: torch.Tensor,
    free_nats: float = 3.0,
) -> torch.Tensor:
    """Analytic KL between two diagonal Gaussians, with free-nats floor."""
    var_post  = logvar_post.exp()
    var_prior = logvar_prior.exp()
    kl = 0.5 * (
        logvar_prior - logvar_post
        + var_post / var_prior
        + (mu_post - mu_prior).pow(2) / var_prior
        - 1.0
    )
    kl = kl.sum(-1)                           # sum over z_dim
    return torch.clamp(kl - free_nats, min=0.0).mean()


# ── Encoder ──────────────────────────────────────────────────────────────────

class ObsEncoder(nn.Module):
    """
    Encode (RGB, depth) → posterior (μ, logvar) for z.

    CNN extracts features from each modality separately, then a shared MLP
    fuses them and outputs the recognition model parameters.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        chs = cfg.conv_channels

        # RGB CNN: 3 → 32 → 64 → 128 → 256
        self.rgb_cnn = nn.Sequential(
            _conv_block(cfg.rgb_in_channels, chs[0]),
            _conv_block(chs[0], chs[1]),
            _conv_block(chs[1], chs[2]),
            _conv_block(chs[2], chs[3]),
        )

        # Depth CNN: 1 → 32 → 64 → 128 → 256
        self.depth_cnn = nn.Sequential(
            _conv_block(cfg.depth_in_channels, chs[0]),
            _conv_block(chs[0], chs[1]),
            _conv_block(chs[1], chs[2]),
            _conv_block(chs[2], chs[3]),
        )

        # After 4× stride-2 convolutions: 64 → 4 spatial (each modality)
        flat_dim = chs[3] * 4 * 4 * 2   # ×2 for RGB + depth

        self.fusion_mlp = nn.Sequential(
            nn.Linear(flat_dim, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
        )

        self.mu_head      = nn.Linear(256, cfg.z_dim)
        self.logvar_head  = nn.Linear(256, cfg.z_dim)

    def forward(
        self,
        rgb:   torch.Tensor,    # [B, 3, H, W]  normalised 0-1
        depth: torch.Tensor,    # [B, 1, H, W]  normalised 0-1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgb_feat   = self.rgb_cnn(rgb).flatten(1)
        depth_feat = self.depth_cnn(depth).flatten(1)
        fused = self.fusion_mlp(torch.cat([rgb_feat, depth_feat], dim=-1))
        return self.mu_head(fused), self.logvar_head(fused)


# ── RSSM recurrent core ───────────────────────────────────────────────────────

class RSSMCore(nn.Module):
    """
    GRU-based deterministic state transition + diagonal-Gaussian prior.

    State: (h_t, z_t)
      h_t — deterministic GRU hidden (carries memory)
      z_t — stochastic latent (sampled from posterior or prior)

    Prior: p(z_t | h_t) = N(μ_prior(h_t), σ_prior(h_t))
    """

    def __init__(self, cfg: RSSMConfig):
        super().__init__()
        self.h_dim = cfg.h_dim
        self.z_dim = cfg.z_dim

        # Action embedding (one-hot → dense)
        self.action_embed = nn.Sequential(
            nn.Linear(cfg.action_dim, 64),
            nn.ELU(inplace=True),
        )

        # Input to GRU: z_{t-1} + action_embed
        gru_in_dim = cfg.z_dim + 64
        self.gru = nn.GRUCell(gru_in_dim, cfg.h_dim)

        # Prior head: h_t → (μ_prior, logvar_prior)
        self.prior_mlp = nn.Sequential(
            nn.Linear(cfg.h_dim, cfg.prior_hidden),
            nn.ELU(inplace=True),
        )
        self.prior_mu     = nn.Linear(cfg.prior_hidden, cfg.z_dim)
        self.prior_logvar = nn.Linear(cfg.prior_hidden, cfg.z_dim)

    def initial_state(self, batch_size: int, device: torch.device
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.h_dim, device=device)
        z = torch.zeros(batch_size, self.z_dim, device=device)
        return h, z

    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.prior_mlp(h)
        return self.prior_mu(feat), self.prior_logvar(feat)

    def step(
        self,
        h_prev:  torch.Tensor,   # [B, h_dim]
        z_prev:  torch.Tensor,   # [B, z_dim]
        action:  torch.Tensor,   # [B, action_dim] one-hot
    ) -> torch.Tensor:
        """Compute h_t from (h_{t-1}, z_{t-1}, a_{t-1})."""
        a_emb = self.action_embed(action)
        gru_in = torch.cat([z_prev, a_emb], dim=-1)
        h_new  = self.gru(gru_in, h_prev)
        return h_new

    @staticmethod
    def reparameterise(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)


# ── Decoder ───────────────────────────────────────────────────────────────────

class ObsDecoder(nn.Module):
    """
    Decode (h_t, z_t) → predicted RGB [3, H, W] and depth [1, H, W].
    """

    def __init__(self, cfg: DecoderConfig):
        super().__init__()
        in_dim = cfg.h_dim + cfg.z_dim
        chs    = (256, 128, 64, 32)

        self.proj = nn.Linear(in_dim, chs[0] * 4 * 4)

        self.rgb_deconv = nn.Sequential(
            _deconv_block(chs[0], chs[1]),
            _deconv_block(chs[1], chs[2]),
            _deconv_block(chs[2], chs[3]),
            nn.ConvTranspose2d(chs[3], 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        self.depth_deconv = nn.Sequential(
            _deconv_block(chs[0], chs[1]),
            _deconv_block(chs[1], chs[2]),
            _deconv_block(chs[2], chs[3]),
            nn.ConvTranspose2d(chs[3], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h: torch.Tensor,   # [B, h_dim]
        z: torch.Tensor,   # [B, z_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.proj(torch.cat([h, z], dim=-1))
        feat = feat.view(feat.size(0), 256, 4, 4)
        return self.rgb_deconv(feat), self.depth_deconv(feat)


# ── Occupancy head ────────────────────────────────────────────────────────────

class OccupancyHead(nn.Module):
    """
    (h_t, z_t) → local binary occupancy grid [grid_size, grid_size].

    Output is sigmoid-activated — use BCEWithLogitsLoss for training.
    """

    def __init__(self, cfg: OccupancyHeadConfig):
        super().__init__()
        in_dim = cfg.h_dim + cfg.z_dim
        gs     = cfg.grid_size
        self.grid_size = gs

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 1024),
            nn.ELU(inplace=True),
            nn.Linear(1024, gs * gs),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, grid_size, grid_size] (NOT sigmoid-activated)."""
        logits = self.mlp(torch.cat([h, z], dim=-1))
        return logits.view(logits.size(0), self.grid_size, self.grid_size)


# ── Full world model ──────────────────────────────────────────────────────────

class WorldModelRSSM(nn.Module):
    """
    Generative world model for indoor robot navigation.

    Modules
    -------
    encoder      : ObsEncoder       — recognition model q(z|o)
    rssm         : RSSMCore         — deterministic state + prior p(z|h)
    decoder      : ObsDecoder       — p(o_rgb, o_depth | h, z)
    occ_head     : OccupancyHead    — p(occ | h, z)

    Forward pass (training, teacher-forced)
    ---------------------------------------
    Given a batch of sequences [B, T, ...]:
      1. For each t, encode (rgb_t, depth_t) → posterior z_t (reparameterised)
      2. Step GRU: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
      3. Compute prior from h_t
      4. Decode (h_t, z_t) → rgb_pred, depth_pred, occ_logits

    Returns a dict of losses (already reduced) and predictions for logging.
    """

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.encoder  = ObsEncoder(cfg.encoder)
        self.rssm     = RSSMCore(cfg.rssm)
        self.decoder  = ObsDecoder(cfg.decoder)
        self.occ_head = OccupancyHead(cfg.occupancy)
        self.cfg      = cfg

    def forward(
        self,
        rgb:    torch.Tensor,   # [B, T, 3, H, W]
        depth:  torch.Tensor,   # [B, T, 1, H, W]
        action: torch.Tensor,   # [B, T, action_dim] one-hot
        occ_gt: torch.Tensor,   # [B, T, G, G] binary occupancy ground truth
    ) -> Dict[str, torch.Tensor]:
        B, T = rgb.shape[:2]
        device = rgb.device

        h, z = self.rssm.initial_state(B, device)

        loss_rgb   = rgb.new_zeros(())
        loss_depth = rgb.new_zeros(())
        loss_occ   = rgb.new_zeros(())
        loss_kl    = rgb.new_zeros(())

        rgb_img   = rgb.view(B * T, *rgb.shape[2:])
        depth_img = depth.view(B * T, *depth.shape[2:])

        # Encode the full sequence in one batched pass (faster than per-step)
        mu_post_all, logvar_post_all = self.encoder(rgb_img, depth_img)
        mu_post_all    = mu_post_all.view(B, T, -1)
        logvar_post_all = logvar_post_all.view(B, T, -1)

        for t in range(T):
            a_t = action[:, t]                         # [B, action_dim]

            # 1. GRU step: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
            h = self.rssm.step(h, z, a_t)

            # 2. Prior p(z_t | h_t)
            mu_prior, logvar_prior = self.rssm.prior(h)

            # 3. Posterior q(z_t | o_t)
            mu_post    = mu_post_all[:, t]
            logvar_post = logvar_post_all[:, t]

            # 4. Sample z_t from posterior (reparameterisation)
            z = self.rssm.reparameterise(mu_post, logvar_post)

            # 5. Decode
            rgb_pred, depth_pred = self.decoder(h, z)
            occ_logits           = self.occ_head(h, z)

            # 6. Losses
            loss_rgb   = loss_rgb   + F.mse_loss(rgb_pred,   rgb[:, t])
            loss_depth = loss_depth + F.mse_loss(depth_pred, depth[:, t])
            loss_occ   = loss_occ   + F.binary_cross_entropy_with_logits(
                occ_logits, occ_gt[:, t]
            )
            loss_kl = loss_kl + kl_divergence(
                mu_post, logvar_post, mu_prior, logvar_prior,
                free_nats=self.cfg.kl_free_nats,
            )

        c = self.cfg
        loss_total = (
            c.w_rgb   * loss_rgb
            + c.w_depth * loss_depth
            + c.w_occ   * loss_occ
            + c.w_kl    * loss_kl
        ) / T

        return {
            "loss":       loss_total,
            "loss_rgb":   loss_rgb   / T,
            "loss_depth": loss_depth / T,
            "loss_occ":   loss_occ   / T,
            "loss_kl":    loss_kl    / T,
            # Last-step predictions for logging / visualisation
            "rgb_pred":   rgb_pred.detach(),
            "depth_pred": depth_pred.detach(),
            "occ_pred":   occ_logits.sigmoid().detach(),
        }

    @torch.no_grad()
    def imagine(
        self,
        h0: torch.Tensor,
        z0: torch.Tensor,
        actions: torch.Tensor,   # [B, H, action_dim]
    ) -> Dict[str, torch.Tensor]:
        """
        Rollout the world model from (h0, z0) using the PRIOR (no observations).
        Used at test time for model-based planning or imagined trajectories.

        Returns predicted rgb, depth, occupancy for each step.
        """
        B, H = actions.shape[:2]
        h, z = h0, z0
        preds = {"rgb": [], "depth": [], "occ": []}
        for t in range(H):
            h = self.rssm.step(h, z, actions[:, t])
            mu_prior, logvar_prior = self.rssm.prior(h)
            z = self.rssm.reparameterise(mu_prior, logvar_prior)
            rgb_p, depth_p = self.decoder(h, z)
            occ_p          = self.occ_head(h, z).sigmoid()
            preds["rgb"].append(rgb_p)
            preds["depth"].append(depth_p)
            preds["occ"].append(occ_p)
        return {k: torch.stack(v, dim=1) for k, v in preds.items()}

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

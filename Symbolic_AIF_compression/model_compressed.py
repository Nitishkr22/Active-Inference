"""
model_compressed.py — Parameterized AIF world model for compression studies.

Drop-in replacement for WorldModelV5, with all hidden dimensions configurable
so we can sweep compression levels CL0 (baseline 1.68M) → CL5 (~6K) without
changing the training or evaluation code.

Key additions:
  - CompressedModelConfig: every dimension is a free parameter
  - use_depthwise: swap standard ResConvBlocks with depthwise-separable blocks
  - Channels auto-scale with encoder_feat_dim (base = feat//4)
  - inference_param_count / model_size_mb properties exclude the decoder
    (decoder is training scaffolding only — never loaded at deployment)
  - All transition/factor dims scale with gru_hidden_dim
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class CompressedModelConfig:
    # --- Observation ---
    obs_channels:        int = 1
    obs_height:          int = 64
    obs_width:           int = 64

    # --- Environment factors ---
    num_actions:         int = 4
    num_row_classes:     int = 9
    num_col_classes:     int = 9
    num_heading_classes: int = 4

    # --- Compression knobs (change these for ablation) ---
    gru_hidden_dim:      int  = 256   # GRU hidden size (scales O(H²))
    encoder_feat_dim:    int  = 128   # CNN output features; channels = feat//4
    context_dim:         int  = 32    # Continuous scene-style latent
    use_depthwise:       bool = False  # Depthwise-separable encoder

    # --- Derived (auto-computed if left at -1) ---
    factor_hidden_dim:      int = -1  # shared trunk width; auto = max(H//2, 8)
    transition_hidden_dim:  int = -1  # transition MLP width; auto = max(H//2, 16)
    collision_hidden_dim:   int = -1  # collision head width; auto = max(H//2, 16)
    action_emb_dim:         int = 16

    # --- Decoder (training only, never deployed) ---
    # Set to -1 for auto: decoder_base = max(feat, 32), decoder_hidden = max(2*feat, 64)
    decoder_base_channels:  int = -1
    decoder_hidden_dim:      int = -1

    # --- V4+ features ---
    use_predictive_prior: bool = True
    context_free_bits:    float = 0.05

    def __post_init__(self):
        H = self.gru_hidden_dim
        if self.factor_hidden_dim     < 0: self.factor_hidden_dim     = max(H // 2, 8)
        if self.transition_hidden_dim < 0: self.transition_hidden_dim = max(H // 2, 16)
        if self.collision_hidden_dim  < 0: self.collision_hidden_dim  = max(H // 2, 16)
        F_ = self.encoder_feat_dim
        if self.decoder_base_channels < 0: self.decoder_base_channels = max(F_, 32)
        if self.decoder_hidden_dim    < 0: self.decoder_hidden_dim    = max(F_ * 2, 64)

    @property
    def encoder_base_channels(self) -> int:
        """Number of base channels in the CNN encoder (scales all layer widths)."""
        return max(self.encoder_feat_dim // 4, 1)


# ============================================================
# Standard conv blocks
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


# ============================================================
# Depthwise-separable conv blocks
# ============================================================

class DepthwiseSepConvBlock(nn.Module):
    """Depthwise-separable replacement for ConvBlock (~8-9× fewer params per layer)."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDepthwiseSepBlock(nn.Module):
    """Depthwise-separable residual block — same interface as ResidualConvBlock."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        # Two DW-PW pairs with residual connection
        self.dw1  = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn1  = nn.BatchNorm2d(channels)
        self.pw1  = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2  = nn.BatchNorm2d(channels)
        self.dw2  = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn3  = nn.BatchNorm2d(channels)
        self.pw2  = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn4  = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn2(self.pw1(self.relu(self.bn1(self.dw1(x))))))
        x = self.bn4(self.pw2(self.relu(self.bn3(self.dw2(x)))))
        return self.relu(x + residual)


# ============================================================
# Encoder
# ============================================================

class ObservationEncoderCompressed(nn.Module):
    def __init__(self, cfg: CompressedModelConfig) -> None:
        super().__init__()
        b   = cfg.encoder_base_channels
        F_  = cfg.encoder_feat_dim
        dw  = cfg.use_depthwise

        CB  = DepthwiseSepConvBlock  if dw else ConvBlock
        RB  = ResidualDepthwiseSepBlock if dw else ResidualConvBlock

        self.net = nn.Sequential(
            CB(cfg.obs_channels, b,     stride=2),
            CB(b,               b * 2,  stride=2),
            RB(b * 2),
            CB(b * 2,           b * 4,  stride=2),
            RB(b * 4),
            CB(b * 4,           b * 4,  stride=2),
            RB(b * 4),
        )
        h_out     = cfg.obs_height // 16
        w_out     = cfg.obs_width  // 16
        flat_dim  = (b * 4) * h_out * w_out
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, F_),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


# ============================================================
# Factor heads  (shared trunk, proportional to gru_hidden_dim)
# ============================================================

class FactorHeadsCompressed(nn.Module):
    def __init__(self, cfg: CompressedModelConfig) -> None:
        super().__init__()
        H  = cfg.gru_hidden_dim
        Fh = cfg.factor_hidden_dim
        C  = cfg.context_dim

        self.shared_trunk     = nn.Sequential(nn.Linear(H, Fh), nn.ReLU(inplace=True))
        self.row_head         = nn.Linear(Fh, cfg.num_row_classes)
        self.col_head         = nn.Linear(Fh, cfg.num_col_classes)
        self.heading_head     = nn.Linear(Fh, cfg.num_heading_classes)
        self.context_mu_head  = nn.Linear(Fh, C)
        self.context_logvar_head = nn.Linear(Fh, C)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat   = self.shared_trunk(h)
        mu     = self.context_mu_head(feat)
        logvar = self.context_logvar_head(feat).clamp(-4.0, 4.0)

        if self.training:
            context = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            context = mu

        return {
            "row_logits":      self.row_head(feat),
            "col_logits":      self.col_head(feat),
            "heading_logits":  self.heading_head(feat),
            "context":         context,
            "context_mu":      mu,
            "context_logvar":  logvar,
        }


# ============================================================
# Transition model  (proportional to gru_hidden_dim / context_dim)
# ============================================================

class FactorizedTransitionModelCompressed(nn.Module):
    def __init__(self, cfg: CompressedModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        state_dim = (
            cfg.num_row_classes
            + cfg.num_col_classes
            + cfg.num_heading_classes
            + cfg.context_dim
            + cfg.action_emb_dim
        )
        Th = cfg.transition_hidden_dim
        Ch = cfg.collision_hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(state_dim, Th), nn.ReLU(inplace=True),
            nn.Linear(Th, Th),        nn.ReLU(inplace=True),
        )
        self.row_head     = nn.Linear(Th, cfg.num_row_classes)
        self.col_head     = nn.Linear(Th, cfg.num_col_classes)
        self.heading_head = nn.Linear(Th, cfg.num_heading_classes)
        self.context_head = nn.Linear(Th, cfg.context_dim)

        self.collision_head = nn.Sequential(
            nn.Linear(state_dim, Ch), nn.ReLU(inplace=True),
            nn.Linear(Ch, 2),
        )

    def forward_step(
        self,
        q_row:      torch.Tensor,
        q_col:      torch.Tensor,
        q_heading:  torch.Tensor,
        context:    torch.Tensor,
        action_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        state_in = torch.cat([q_row, q_col, q_heading, context, action_emb], dim=-1)
        shared   = self.shared(state_in)

        move_row_logits     = self.row_head(shared)
        move_col_logits     = self.col_head(shared)
        move_heading_logits = self.heading_head(shared)
        move_context        = self.context_head(shared)

        collision_logits = self.collision_head(state_in)
        collision_prob   = torch.softmax(collision_logits, dim=-1)[:, 1:2]

        move_row_probs     = torch.softmax(move_row_logits,     dim=-1)
        move_col_probs     = torch.softmax(move_col_logits,     dim=-1)
        move_heading_probs = torch.softmax(move_heading_logits, dim=-1)

        next_row_probs     = (1 - collision_prob) * move_row_probs     + collision_prob * q_row
        next_col_probs     = (1 - collision_prob) * move_col_probs     + collision_prob * q_col
        next_heading_probs = (1 - collision_prob) * move_heading_probs + collision_prob * q_heading
        next_context       = (1 - collision_prob) * move_context       + collision_prob * context

        eps = 1e-8
        return {
            "move_row_logits":      move_row_logits,
            "move_col_logits":      move_col_logits,
            "move_heading_logits":  move_heading_logits,
            "move_context":         move_context,
            "collision_logits":     collision_logits,
            "next_row_logits":      torch.log(next_row_probs.clamp_min(eps)),
            "next_col_logits":      torch.log(next_col_probs.clamp_min(eps)),
            "next_heading_logits":  torch.log(next_heading_probs.clamp_min(eps)),
            "next_row_probs":       next_row_probs,
            "next_col_probs":       next_col_probs,
            "next_heading_probs":   next_heading_probs,
            "next_context":         next_context,
        }


# ============================================================
# Decoder  (training-only reconstruction, never deployed)
# ============================================================

class ObservationDecoderCompressed(nn.Module):
    def __init__(self, cfg: CompressedModelConfig) -> None:
        super().__init__()
        in_dim  = cfg.num_row_classes + cfg.num_col_classes + cfg.num_heading_classes + cfg.context_dim
        base    = cfg.decoder_base_channels
        hidden  = cfg.decoder_hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),       nn.ReLU(inplace=True),
            nn.Linear(hidden, base * 4 * 4), nn.ReLU(inplace=True),
        )
        self.block_4 = nn.Sequential(ResidualConvBlock(base), ResidualConvBlock(base))
        self.up_8    = nn.Sequential(
            nn.ConvTranspose2d(base, base // 2, 4, 2, 1),
            nn.BatchNorm2d(base // 2), nn.ReLU(inplace=True), ResidualConvBlock(base // 2),
        )
        self.up_16   = nn.Sequential(
            nn.ConvTranspose2d(base // 2, base // 4, 4, 2, 1),
            nn.BatchNorm2d(base // 4), nn.ReLU(inplace=True), ResidualConvBlock(base // 4),
        )
        self.up_32   = nn.Sequential(
            nn.ConvTranspose2d(base // 4, base // 8, 4, 2, 1),
            nn.BatchNorm2d(base // 8), nn.ReLU(inplace=True), ResidualConvBlock(base // 8),
        )
        self.up_64   = nn.Sequential(
            nn.ConvTranspose2d(base // 8, base // 16, 4, 2, 1),
            nn.BatchNorm2d(base // 16), nn.ReLU(inplace=True), ResidualConvBlock(base // 16),
        )
        self.out_conv = nn.Conv2d(base // 16, cfg.obs_channels, 3, padding=1)

    def forward(
        self,
        q_row: torch.Tensor, q_col: torch.Tensor,
        q_heading: torch.Tensor, context: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([q_row, q_col, q_heading, context], dim=-1)
        B = x.shape[0]
        x = self.fc(x).view(B, -1, 4, 4)
        x = self.block_4(x)
        x = self.up_8(x)
        x = self.up_16(x)
        x = self.up_32(x)
        x = self.up_64(x)
        return torch.sigmoid(self.out_conv(x))


# ============================================================
# Main model
# ============================================================

class WorldModelCompressed(nn.Module):
    """
    Parameterized AIF world model for compression ablations.

    Interface is identical to WorldModelV5 (forward_filter, forward_step_online,
    rollout_from_filtered_state) so evaluate_compressed.py and the EFE planner
    work without changes.

    The only new surface area:
      - inference_param_count: parameters used at deployment (excludes decoder)
      - model_size_mb: FP32 inference memory footprint
    """

    def __init__(self, cfg: CompressedModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.action_embedding    = nn.Embedding(cfg.num_actions, cfg.action_emb_dim)
        self.encoder             = ObservationEncoderCompressed(cfg)
        self.filter_input_proj   = nn.Sequential(
            nn.Linear(cfg.encoder_feat_dim + cfg.action_emb_dim, cfg.gru_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.filter_gru          = nn.GRU(
            input_size=cfg.gru_hidden_dim,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.factor_heads        = FactorHeadsCompressed(cfg)
        self.decoder             = ObservationDecoderCompressed(cfg)
        self.transition_model    = FactorizedTransitionModelCompressed(cfg)

    # -------------------------------------------------------
    # Size / count helpers
    # -------------------------------------------------------

    @property
    def inference_param_count(self) -> int:
        """Parameters used at deployment (decoder excluded)."""
        decoder_p = sum(p.numel() for p in self.decoder.parameters())
        return sum(p.numel() for p in self.parameters()) - decoder_p

    @property
    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def model_size_mb(self) -> float:
        """FP32 inference model size in MB."""
        return self.inference_param_count * 4 / (1024 ** 2)

    # -------------------------------------------------------
    # Helpers
    # -------------------------------------------------------

    def encode_observation_sequence(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = obs_seq.shape
        return self.encoder(obs_seq.view(B * T, C, H, W)).view(B, T, -1)

    def build_filter_inputs(self, feat_seq: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, T, _ = feat_seq.shape
        device = feat_seq.device
        zero_action = torch.zeros(B, 1, dtype=torch.long, device=device)
        prev_actions = torch.cat([zero_action, actions], dim=1)[:, :T]
        prev_a_emb   = self.action_embedding(prev_actions)
        return self.filter_input_proj(torch.cat([feat_seq, prev_a_emb], dim=-1))

    def compute_context_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        if self.cfg.context_free_bits > 0.0:
            kl_per_dim = kl.mean(dim=[0, 1]).clamp_min(self.cfg.context_free_bits)
            return kl_per_dim.sum()
        return kl.sum(dim=-1).mean()

    # -------------------------------------------------------
    # Filtering pass  (same logic as WorldModelV5)
    # -------------------------------------------------------

    def forward_filter(
        self,
        obs_seq:   torch.Tensor,       # [B, T, C, H, W]
        actions:   torch.Tensor,       # [B, T-1]
        h0:        Optional[torch.Tensor] = None,
        skip_recon: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = obs_seq.shape

        feat_seq      = self.encode_observation_sequence(obs_seq)
        filter_inputs = self.build_filter_inputs(feat_seq, actions)
        h_seq, h_last = self.filter_gru(filter_inputs, h0)

        h_flat       = h_seq.reshape(B * T, -1)
        factors_flat = self.factor_heads(h_flat)

        enc_row_logits_seq     = factors_flat["row_logits"].view(B, T, -1)
        enc_col_logits_seq     = factors_flat["col_logits"].view(B, T, -1)
        enc_heading_logits_seq = factors_flat["heading_logits"].view(B, T, -1)
        context_seq            = factors_flat["context"].view(B, T, -1)
        context_mu_seq         = factors_flat["context_mu"].view(B, T, -1)
        context_logvar_seq     = factors_flat["context_logvar"].view(B, T, -1)

        eps = 1e-8

        if self.cfg.use_predictive_prior:
            post_row_list   = [enc_row_logits_seq[:, 0:1, :]]
            post_col_list   = [enc_col_logits_seq[:, 0:1, :]]
            post_hdg_list   = [enc_heading_logits_seq[:, 0:1, :]]
            tr_row_list, tr_col_list, tr_hdg_list, coll_list = [], [], [], []

            prev_row_probs = torch.softmax(enc_row_logits_seq[:, 0, :], dim=-1)
            prev_col_probs = torch.softmax(enc_col_logits_seq[:, 0, :], dim=-1)
            prev_hdg_probs = torch.softmax(enc_heading_logits_seq[:, 0, :], dim=-1)
            prev_context   = context_seq[:, 0, :]

            for t in range(1, T):
                a_emb = self.action_embedding(actions[:, t - 1])
                tr    = self.transition_model.forward_step(
                    prev_row_probs, prev_col_probs, prev_hdg_probs, prev_context, a_emb
                )
                coll_list.append(tr["collision_logits"])
                tr_row_list.append(tr["next_row_probs"])
                tr_col_list.append(tr["next_col_probs"])
                tr_hdg_list.append(tr["next_heading_probs"])

                log_prior_row = torch.log(tr["next_row_probs"].clamp_min(eps))
                log_prior_col = torch.log(tr["next_col_probs"].clamp_min(eps))
                log_prior_hdg = torch.log(tr["next_heading_probs"].clamp_min(eps))
                log_lik_row   = F.log_softmax(enc_row_logits_seq[:, t, :],     dim=-1)
                log_lik_col   = F.log_softmax(enc_col_logits_seq[:, t, :],     dim=-1)
                log_lik_hdg   = F.log_softmax(enc_heading_logits_seq[:, t, :], dim=-1)

                post_row = F.softmax(log_prior_row + log_lik_row, dim=-1)
                post_col = F.softmax(log_prior_col + log_lik_col, dim=-1)
                post_hdg = F.softmax(log_prior_hdg + log_lik_hdg, dim=-1)

                post_row_list.append(torch.log(post_row.clamp_min(eps)).unsqueeze(1))
                post_col_list.append(torch.log(post_col.clamp_min(eps)).unsqueeze(1))
                post_hdg_list.append(torch.log(post_hdg.clamp_min(eps)).unsqueeze(1))

                prev_row_probs = post_row
                prev_col_probs = post_col
                prev_hdg_probs = post_hdg
                prev_context   = context_seq[:, t, :]

            row_logits_seq     = torch.cat(post_row_list,  dim=1)
            col_logits_seq     = torch.cat(post_col_list,  dim=1)
            heading_logits_seq = torch.cat(post_hdg_list,  dim=1)
            collision_logits_seq               = torch.stack(coll_list,   dim=1)
            transition_prior_row_probs         = torch.stack(tr_row_list, dim=1)
            transition_prior_col_probs         = torch.stack(tr_col_list, dim=1)
            transition_prior_heading_probs     = torch.stack(tr_hdg_list, dim=1)

        else:
            row_logits_seq     = enc_row_logits_seq
            col_logits_seq     = enc_col_logits_seq
            heading_logits_seq = enc_heading_logits_seq
            coll_list = []
            for t in range(T - 1):
                a_emb = self.action_embedding(actions[:, t])
                tr    = self.transition_model.forward_step(
                    torch.softmax(row_logits_seq[:, t, :],     dim=-1),
                    torch.softmax(col_logits_seq[:, t, :],     dim=-1),
                    torch.softmax(heading_logits_seq[:, t, :], dim=-1),
                    context_seq[:, t, :], a_emb,
                )
                coll_list.append(tr["collision_logits"])
            collision_logits_seq           = torch.stack(coll_list, dim=1)
            transition_prior_row_probs     = torch.zeros(B, T-1, self.cfg.num_row_classes,     device=obs_seq.device)
            transition_prior_col_probs     = torch.zeros(B, T-1, self.cfg.num_col_classes,     device=obs_seq.device)
            transition_prior_heading_probs = torch.zeros(B, T-1, self.cfg.num_heading_classes, device=obs_seq.device)

        row_probs_seq     = torch.softmax(row_logits_seq,     dim=-1)
        col_probs_seq     = torch.softmax(col_logits_seq,     dim=-1)
        heading_probs_seq = torch.softmax(heading_logits_seq, dim=-1)

        if skip_recon or not self.training:
            recon_seq = None
        else:
            q_row_flat     = row_probs_seq.reshape(B * T, -1)
            q_col_flat     = col_probs_seq.reshape(B * T, -1)
            q_hdg_flat     = heading_probs_seq.reshape(B * T, -1)
            ctx_flat       = context_seq.reshape(B * T, -1)
            recon_seq      = self.decoder(q_row_flat, q_col_flat, q_hdg_flat, ctx_flat).view(B, T, C, H, W)

        return {
            "feat_seq":                       feat_seq,
            "h_seq":                          h_seq,
            "h_last":                         h_last,
            "row_logits_seq":                 row_logits_seq,
            "col_logits_seq":                 col_logits_seq,
            "heading_logits_seq":             heading_logits_seq,
            "row_probs_seq":                  row_probs_seq,
            "col_probs_seq":                  col_probs_seq,
            "heading_probs_seq":              heading_probs_seq,
            "context_seq":                    context_seq,
            "recon_seq":                      recon_seq,
            "collision_logits_seq":           collision_logits_seq,
            "encoder_row_logits_seq":         enc_row_logits_seq,
            "encoder_col_logits_seq":         enc_col_logits_seq,
            "encoder_heading_logits_seq":     enc_heading_logits_seq,
            "transition_prior_row_probs":     transition_prior_row_probs,
            "transition_prior_col_probs":     transition_prior_col_probs,
            "transition_prior_heading_probs": transition_prior_heading_probs,
            "context_mu_seq":                 context_mu_seq,
            "context_logvar_seq":             context_logvar_seq,
        }

    # -------------------------------------------------------
    # Online single-step inference  (O(1) per step)
    # -------------------------------------------------------

    def forward_step_online(
        self,
        obs_t:       torch.Tensor,
        action_prev: Optional[torch.Tensor],
        h_prev:      Optional[torch.Tensor]            = None,
        prev_belief: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        device = obs_t.device

        feat = self.encoder(obs_t)

        if action_prev is None:
            action_prev = torch.zeros(1, dtype=torch.long, device=device)
        a_emb = self.action_embedding(action_prev)

        x = self.filter_input_proj(torch.cat([feat, a_emb], dim=-1))
        h_seq, h_new = self.filter_gru(x.unsqueeze(1), h_prev)
        h_t = h_seq[:, 0, :]

        factors = self.factor_heads(h_t)

        if prev_belief is not None and self.cfg.use_predictive_prior:
            tr = self.transition_model.forward_step(
                prev_belief["row_probs"],
                prev_belief["col_probs"],
                prev_belief["heading_probs"],
                prev_belief["context"],
                a_emb,
            )
            eps = 1e-8
            row_probs     = F.softmax(torch.log(tr["next_row_probs"].clamp_min(eps))     + F.log_softmax(factors["row_logits"],     dim=-1), dim=-1)
            col_probs     = F.softmax(torch.log(tr["next_col_probs"].clamp_min(eps))     + F.log_softmax(factors["col_logits"],     dim=-1), dim=-1)
            heading_probs = F.softmax(torch.log(tr["next_heading_probs"].clamp_min(eps)) + F.log_softmax(factors["heading_logits"], dim=-1), dim=-1)
        else:
            row_probs     = F.softmax(factors["row_logits"],     dim=-1)
            col_probs     = F.softmax(factors["col_logits"],     dim=-1)
            heading_probs = F.softmax(factors["heading_logits"], dim=-1)

        belief = {
            "row_probs":      row_probs,
            "col_probs":      col_probs,
            "heading_probs":  heading_probs,
            "context":        factors["context"],
            "context_mu":     factors["context_mu"],
            "context_logvar": factors["context_logvar"],
        }
        return belief, h_new

    # -------------------------------------------------------
    # Rollout  (same as V5)
    # -------------------------------------------------------

    def rollout_from_filtered_state(
        self,
        row_probs_start:     torch.Tensor,
        col_probs_start:     torch.Tensor,
        heading_probs_start: torch.Tensor,
        context_start:       torch.Tensor,
        action_seq:          torch.Tensor,
        skip_recon:          bool = False,
    ) -> Dict[str, torch.Tensor]:
        B, K = action_seq.shape
        q_row, q_col, q_hdg, ctx = (
            row_probs_start, col_probs_start, heading_probs_start, context_start
        )
        row_logits_roll, col_logits_roll, hdg_logits_roll = [], [], []
        row_probs_roll,  col_probs_roll,  hdg_probs_roll  = [], [], []
        ctx_roll, coll_roll, recon_roll = [], [], []

        for k in range(K):
            a_emb   = self.action_embedding(action_seq[:, k])
            step    = self.transition_model.forward_step(q_row, q_col, q_hdg, ctx, a_emb)
            q_row   = step["next_row_probs"]
            q_col   = step["next_col_probs"]
            q_hdg   = step["next_heading_probs"]
            ctx     = step["next_context"]

            if not skip_recon:
                recon_roll.append(self.decoder(q_row, q_col, q_hdg, ctx))

            row_logits_roll.append(step["next_row_logits"])
            col_logits_roll.append(step["next_col_logits"])
            hdg_logits_roll.append(step["next_heading_logits"])
            row_probs_roll.append(q_row)
            col_probs_roll.append(q_col)
            hdg_probs_roll.append(q_hdg)
            ctx_roll.append(ctx)
            coll_roll.append(step["collision_logits"])

        return {
            "row_logits_roll":       torch.stack(row_logits_roll, dim=1),
            "col_logits_roll":       torch.stack(col_logits_roll, dim=1),
            "heading_logits_roll":   torch.stack(hdg_logits_roll, dim=1),
            "row_probs_roll":        torch.stack(row_probs_roll,  dim=1),
            "col_probs_roll":        torch.stack(col_probs_roll,  dim=1),
            "heading_probs_roll":    torch.stack(hdg_probs_roll,  dim=1),
            "context_roll":          torch.stack(ctx_roll,        dim=1),
            "collision_logits_roll": torch.stack(coll_roll,       dim=1),
            "recon_roll":            torch.stack(recon_roll, dim=1) if recon_roll else None,
        }


# ============================================================
# Quick shape / size test
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = {
        "CL0 (baseline)":     CompressedModelConfig(gru_hidden_dim=256, encoder_feat_dim=128, context_dim=32),
        "CL1 (~4× smaller)":  CompressedModelConfig(gru_hidden_dim=128, encoder_feat_dim=64,  context_dim=16),
        "CL2 (~14× smaller)": CompressedModelConfig(gru_hidden_dim=64,  encoder_feat_dim=32,  context_dim=16),
        "CL3 (~15× smaller)": CompressedModelConfig(gru_hidden_dim=64,  encoder_feat_dim=32,  context_dim=8),
        "CL4 (~52× smaller)": CompressedModelConfig(gru_hidden_dim=32,  encoder_feat_dim=16,  context_dim=8),
        "CL5 (~250× smaller)":CompressedModelConfig(gru_hidden_dim=16,  encoder_feat_dim=8,   context_dim=4),
        "CL1-DW (depthwise)": CompressedModelConfig(gru_hidden_dim=128, encoder_feat_dim=64,  context_dim=16, use_depthwise=True),
        "CL2-DW (depthwise)": CompressedModelConfig(gru_hidden_dim=64,  encoder_feat_dim=32,  context_dim=16, use_depthwise=True),
    }

    print(f"\n{'Config':<22} | {'Total':>8} | {'Inf':>8} | {'Inf MB':>7} | {'Compression':>11}")
    print("-" * 70)
    baseline_inf = None
    for name, cfg in configs.items():
        m = WorldModelCompressed(cfg).to(device)
        if baseline_inf is None:
            baseline_inf = m.inference_param_count
        ratio = baseline_inf / m.inference_param_count
        print(
            f"{name:<22} | {m.total_param_count:>8,} | {m.inference_param_count:>8,} | "
            f"{m.model_size_mb:>6.2f}MB | {ratio:>8.1f}×"
        )

    # Shape smoke-test with CL2
    print("\n--- Shape test (CL2) ---")
    cfg = CompressedModelConfig(gru_hidden_dim=64, encoder_feat_dim=32, context_dim=16)
    m   = WorldModelCompressed(cfg).to(device)
    B, T = 4, 16
    obs  = torch.randn(B, T, 1, 64, 64, device=device)
    acts = torch.randint(0, 4, (B, T-1), device=device)
    m.train()
    out  = m.forward_filter(obs, acts)
    print(f"  row_probs_seq: {tuple(out['row_probs_seq'].shape)}  recon_seq: {tuple(out['recon_seq'].shape)}")
    m.eval()
    with torch.no_grad():
        out2 = m.forward_filter(obs, acts)
    print(f"  eval recon_seq: {out2['recon_seq']}  (None expected)")
    # Online
    h, bel = None, None
    for s in range(3):
        obs_t   = torch.randn(1, 1, 64, 64, device=device)
        act_t   = None if s == 0 else torch.tensor([1], dtype=torch.long, device=device)
        with torch.no_grad():
            bel, h = m.forward_step_online(obs_t, act_t, h, bel)
    print(f"  online row_probs: {tuple(bel['row_probs'].shape)}  h: {tuple(h.shape)}")
    print("\nAll checks passed.")

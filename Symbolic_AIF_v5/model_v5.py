from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class ModelV5Config:
    # Observation shape
    obs_channels: int = 1
    obs_height: int = 64
    obs_width: int = 64

    # Environment state factors
    num_actions: int = 4
    num_row_classes: int = 9
    num_col_classes: int = 9
    num_heading_classes: int = 4

    # Encoder / filter
    action_emb_dim: int = 16
    encoder_feat_dim: int = 128
    gru_hidden_dim: int = 256

    # Factorized state representation
    # context_dim reduced 64→32: empirical KL analysis showed only ~20-30 dims
    # were active (>free_bits) in V4c, so 64 was over-parameterised.
    context_dim: int = 32
    factor_hidden_dim: int = 128    # shared trunk output width
    transition_hidden_dim: int = 128
    collision_hidden_dim: int = 128

    # Decoder (used for training reconstruction loss only)
    decoder_hidden_dim: int = 256
    decoder_base_channels: int = 128

    # V4+ retained features
    use_predictive_prior: bool = True
    context_free_bits: float = 0.05


# ============================================================
# Small blocks
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
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


# ============================================================
# Encoder  (identical to V4)
# ============================================================

class ObservationEncoderV5(nn.Module):
    def __init__(self, cfg: ModelV5Config) -> None:
        super().__init__()
        c = cfg.decoder_base_channels // 4   # 32
        self.net = nn.Sequential(
            ConvBlock(cfg.obs_channels, c, stride=2),
            ConvBlock(c, c * 2, stride=2),
            ResidualConvBlock(c * 2),
            ConvBlock(c * 2, c * 4, stride=2),
            ResidualConvBlock(c * 4),
            ConvBlock(c * 4, c * 4, stride=2),
            ResidualConvBlock(c * 4),
        )
        reduced_h = cfg.obs_height // 16
        reduced_w = cfg.obs_width // 16
        flat_dim = (c * 4) * reduced_h * reduced_w
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, cfg.encoder_feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


# ============================================================
# Factor heads  (V5: shared trunk — one projection for all heads)
# ============================================================

class FactorHeadsV5(nn.Module):
    """
    V5 change: all heads share a single Linear(H→Fh)+ReLU trunk before
    branching into thin output layers.  In V4 each of the 5 heads had its
    own independent 256→128 projection, duplicating ~40% of the module's
    parameters.  The shared trunk halves this while keeping the same
    representational width per branch.
    """

    def __init__(self, cfg: ModelV5Config) -> None:
        super().__init__()
        H = cfg.gru_hidden_dim
        Fh = cfg.factor_hidden_dim

        # One shared projection instead of five independent ones
        self.shared_trunk = nn.Sequential(
            nn.Linear(H, Fh),
            nn.ReLU(inplace=True),
        )

        # Thin output branches
        self.row_head = nn.Linear(Fh, cfg.num_row_classes)
        self.col_head = nn.Linear(Fh, cfg.num_col_classes)
        self.heading_head = nn.Linear(Fh, cfg.num_heading_classes)
        self.context_mu_head = nn.Linear(Fh, cfg.context_dim)
        self.context_logvar_head = nn.Linear(Fh, cfg.context_dim)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.shared_trunk(h)

        mu = self.context_mu_head(feat)
        logvar = self.context_logvar_head(feat).clamp(-4.0, 4.0)

        if self.training:
            std = torch.exp(0.5 * logvar)
            context = mu + torch.randn_like(std) * std
        else:
            context = mu

        return {
            "row_logits": self.row_head(feat),
            "col_logits": self.col_head(feat),
            "heading_logits": self.heading_head(feat),
            "context": context,
            "context_mu": mu,
            "context_logvar": logvar,
        }


# ============================================================
# Transition model  (identical to V4)
# ============================================================

class FactorizedTransitionModelV5(nn.Module):
    def __init__(self, cfg: ModelV5Config) -> None:
        super().__init__()
        self.cfg = cfg

        state_dim = (
            cfg.num_row_classes
            + cfg.num_col_classes
            + cfg.num_heading_classes
            + cfg.context_dim
            + cfg.action_emb_dim
        )
        H = cfg.transition_hidden_dim

        self.shared = nn.Sequential(
            nn.Linear(state_dim, H),
            nn.ReLU(inplace=True),
            nn.Linear(H, H),
            nn.ReLU(inplace=True),
        )
        self.row_head = nn.Linear(H, cfg.num_row_classes)
        self.col_head = nn.Linear(H, cfg.num_col_classes)
        self.heading_head = nn.Linear(H, cfg.num_heading_classes)
        self.context_head = nn.Linear(H, cfg.context_dim)

        self.collision_head = nn.Sequential(
            nn.Linear(state_dim, cfg.collision_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.collision_hidden_dim, 2),
        )

    def forward_step(
        self,
        q_row: torch.Tensor,
        q_col: torch.Tensor,
        q_heading: torch.Tensor,
        context: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        state_in = torch.cat([q_row, q_col, q_heading, context, action_emb], dim=-1)

        shared = self.shared(state_in)

        move_row_logits = self.row_head(shared)
        move_col_logits = self.col_head(shared)
        move_heading_logits = self.heading_head(shared)
        move_context = self.context_head(shared)

        collision_logits = self.collision_head(state_in)
        collision_prob = torch.softmax(collision_logits, dim=-1)[:, 1:2]

        move_row_probs = torch.softmax(move_row_logits, dim=-1)
        move_col_probs = torch.softmax(move_col_logits, dim=-1)
        move_heading_probs = torch.softmax(move_heading_logits, dim=-1)

        next_row_probs = (1.0 - collision_prob) * move_row_probs + collision_prob * q_row
        next_col_probs = (1.0 - collision_prob) * move_col_probs + collision_prob * q_col
        next_heading_probs = (1.0 - collision_prob) * move_heading_probs + collision_prob * q_heading
        next_context = (1.0 - collision_prob) * move_context + collision_prob * context

        eps = 1e-8
        return {
            "move_row_logits": move_row_logits,
            "move_col_logits": move_col_logits,
            "move_heading_logits": move_heading_logits,
            "move_context": move_context,
            "collision_logits": collision_logits,
            "next_row_logits": torch.log(next_row_probs.clamp_min(eps)),
            "next_col_logits": torch.log(next_col_probs.clamp_min(eps)),
            "next_heading_logits": torch.log(next_heading_probs.clamp_min(eps)),
            "next_row_probs": next_row_probs,
            "next_col_probs": next_col_probs,
            "next_heading_probs": next_heading_probs,
            "next_context": next_context,
        }


# ============================================================
# Decoder  (identical to V4 — training-only reconstruction)
# ============================================================

class ObservationDecoderV5(nn.Module):
    def __init__(self, cfg: ModelV5Config) -> None:
        super().__init__()
        in_dim = (
            cfg.num_row_classes
            + cfg.num_col_classes
            + cfg.num_heading_classes
            + cfg.context_dim
        )
        base = cfg.decoder_base_channels

        self.fc = nn.Sequential(
            nn.Linear(in_dim, cfg.decoder_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.decoder_hidden_dim, base * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.block_4 = nn.Sequential(ResidualConvBlock(base), ResidualConvBlock(base))
        self.up_8 = nn.Sequential(
            nn.ConvTranspose2d(base, base // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base // 2), nn.ReLU(inplace=True), ResidualConvBlock(base // 2),
        )
        self.up_16 = nn.Sequential(
            nn.ConvTranspose2d(base // 2, base // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base // 4), nn.ReLU(inplace=True), ResidualConvBlock(base // 4),
        )
        self.up_32 = nn.Sequential(
            nn.ConvTranspose2d(base // 4, base // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base // 8), nn.ReLU(inplace=True), ResidualConvBlock(base // 8),
        )
        self.up_64 = nn.Sequential(
            nn.ConvTranspose2d(base // 8, base // 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base // 16), nn.ReLU(inplace=True), ResidualConvBlock(base // 16),
        )
        self.out_conv = nn.Conv2d(base // 16, cfg.obs_channels, kernel_size=3, padding=1)

    def forward(
        self,
        q_row: torch.Tensor,
        q_col: torch.Tensor,
        q_heading: torch.Tensor,
        context: torch.Tensor,
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

class WorldModelV5(nn.Module):
    """
    V5 — real-time-capable AIF world model.

    Key changes from V4c:
    1. Transformer removed: GRU hidden state feeds directly into factor heads.
       This eliminates the O(T²) attention cost and, critically, makes caching
       possible — the GRU state h_t can be carried forward one step at a time.

    2. Shared factor-head trunk: one Linear(H→Fh)+ReLU shared by all output
       branches, reducing the number of parameters vs V4's five independent trunks.

    3. context_dim 64→32: only ~20-30 dims were carrying signal in V4c (KL > free_bits).

    4. Decoder skipped at inference (skip_recon=True): the image reconstruction
       is training scaffolding only; the planner never uses recon_seq.

    5. forward_step_online(): O(1) per step for real-time deployment — process
       one new frame, carry h_t forward, run Bayesian update. No history replay.

    6. Collision logits collected inside the Bayesian update loop (no second
       pass through the transition model).
    """

    def __init__(self, cfg: ModelV5Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.action_embedding = nn.Embedding(cfg.num_actions, cfg.action_emb_dim)
        self.encoder = ObservationEncoderV5(cfg)

        self.filter_input_proj = nn.Sequential(
            nn.Linear(cfg.encoder_feat_dim + cfg.action_emb_dim, cfg.gru_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.filter_gru = nn.GRU(
            input_size=cfg.gru_hidden_dim,
            hidden_size=cfg.gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        # ← No temporal_attention in V5

        self.factor_heads = FactorHeadsV5(cfg)
        self.decoder = ObservationDecoderV5(cfg)
        self.transition_model = FactorizedTransitionModelV5(cfg)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def encode_observation_sequence(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = obs_seq.shape
        feat = self.encoder(obs_seq.view(B * T, C, H, W))
        return feat.view(B, T, -1)

    def build_filter_inputs(self, feat_seq: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, T, _ = feat_seq.shape
        device = feat_seq.device
        zero_action = torch.zeros(B, 1, dtype=torch.long, device=device)
        prev_actions = torch.cat([zero_action, actions], dim=1)[:, :T]
        prev_a_emb = self.action_embedding(prev_actions)
        return self.filter_input_proj(torch.cat([feat_seq, prev_a_emb], dim=-1))

    def compute_context_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
        if self.cfg.context_free_bits > 0.0:
            kl_per_dim = kl.mean(dim=[0, 1]).clamp_min(self.cfg.context_free_bits)
            return kl_per_dim.sum()
        return kl.sum(dim=-1).mean()

    # --------------------------------------------------------
    # Filtering pass  (V5: no attention, merged collision loop)
    # --------------------------------------------------------

    def forward_filter(
        self,
        obs_seq: torch.Tensor,     # [B, T, C, H, W]
        actions: torch.Tensor,     # [B, T-1]
        h0: Optional[torch.Tensor] = None,
        skip_recon: bool = False,  # True during inference — decoder not needed
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = obs_seq.shape
        if actions.shape[1] != T - 1:
            raise ValueError(
                f"Expected actions [B,T-1], got {tuple(actions.shape)} for T={T}"
            )

        # ---- Encode all observations ----
        feat_seq = self.encode_observation_sequence(obs_seq)         # [B,T,F]
        filter_inputs = self.build_filter_inputs(feat_seq, actions)  # [B,T,H]

        # V5: GRU output is used directly — no Transformer on top
        h_seq, h_last = self.filter_gru(filter_inputs, h0)          # [B,T,H]

        # ---- Raw encoder factor beliefs ----
        h_flat = h_seq.reshape(B * T, -1)
        factors_flat = self.factor_heads(h_flat)

        enc_row_logits_seq    = factors_flat["row_logits"].view(B, T, -1)
        enc_col_logits_seq    = factors_flat["col_logits"].view(B, T, -1)
        enc_heading_logits_seq = factors_flat["heading_logits"].view(B, T, -1)
        context_seq           = factors_flat["context"].view(B, T, -1)
        context_mu_seq        = factors_flat["context_mu"].view(B, T, -1)
        context_logvar_seq    = factors_flat["context_logvar"].view(B, T, -1)

        # ---- Bayesian update + collision collection in one pass ----
        eps = 1e-8

        if self.cfg.use_predictive_prior:
            post_row_logits_list    = [enc_row_logits_seq[:, 0:1, :]]
            post_col_logits_list    = [enc_col_logits_seq[:, 0:1, :]]
            post_heading_logits_list = [enc_heading_logits_seq[:, 0:1, :]]

            trans_prior_row_list    = []
            trans_prior_col_list    = []
            trans_prior_heading_list = []
            collision_logits_list   = []  # collected here — no second pass needed

            prev_row_probs    = torch.softmax(enc_row_logits_seq[:, 0, :], dim=-1)
            prev_col_probs    = torch.softmax(enc_col_logits_seq[:, 0, :], dim=-1)
            prev_heading_probs = torch.softmax(enc_heading_logits_seq[:, 0, :], dim=-1)
            prev_context      = context_seq[:, 0, :]

            for t in range(1, T):
                a_emb_t = self.action_embedding(actions[:, t - 1])

                tr_out = self.transition_model.forward_step(
                    q_row=prev_row_probs,
                    q_col=prev_col_probs,
                    q_heading=prev_heading_probs,
                    context=prev_context,
                    action_emb=a_emb_t,
                )

                # Collect collision and transition prior (same call, no repeat)
                collision_logits_list.append(tr_out["collision_logits"])
                trans_prior_row_list.append(tr_out["next_row_probs"])
                trans_prior_col_list.append(tr_out["next_col_probs"])
                trans_prior_heading_list.append(tr_out["next_heading_probs"])

                # log-space Bayesian update
                log_prior_row    = torch.log(tr_out["next_row_probs"].clamp_min(eps))
                log_prior_col    = torch.log(tr_out["next_col_probs"].clamp_min(eps))
                log_prior_heading = torch.log(tr_out["next_heading_probs"].clamp_min(eps))

                log_lik_row    = F.log_softmax(enc_row_logits_seq[:, t, :], dim=-1)
                log_lik_col    = F.log_softmax(enc_col_logits_seq[:, t, :], dim=-1)
                log_lik_heading = F.log_softmax(enc_heading_logits_seq[:, t, :], dim=-1)

                post_row_probs    = F.softmax(log_prior_row + log_lik_row, dim=-1)
                post_col_probs    = F.softmax(log_prior_col + log_lik_col, dim=-1)
                post_heading_probs = F.softmax(log_prior_heading + log_lik_heading, dim=-1)

                post_row_logits_list.append(torch.log(post_row_probs.clamp_min(eps)).unsqueeze(1))
                post_col_logits_list.append(torch.log(post_col_probs.clamp_min(eps)).unsqueeze(1))
                post_heading_logits_list.append(torch.log(post_heading_probs.clamp_min(eps)).unsqueeze(1))

                prev_row_probs    = post_row_probs
                prev_col_probs    = post_col_probs
                prev_heading_probs = post_heading_probs
                prev_context      = context_seq[:, t, :]

            row_logits_seq    = torch.cat(post_row_logits_list, dim=1)
            col_logits_seq    = torch.cat(post_col_logits_list, dim=1)
            heading_logits_seq = torch.cat(post_heading_logits_list, dim=1)

            collision_logits_seq           = torch.stack(collision_logits_list, dim=1)  # [B,T-1,2]
            transition_prior_row_probs    = torch.stack(trans_prior_row_list, dim=1)
            transition_prior_col_probs    = torch.stack(trans_prior_col_list, dim=1)
            transition_prior_heading_probs = torch.stack(trans_prior_heading_list, dim=1)

        else:
            # V3-fallback: raw encoder logits, separate collision pass
            row_logits_seq    = enc_row_logits_seq
            col_logits_seq    = enc_col_logits_seq
            heading_logits_seq = enc_heading_logits_seq

            collision_list = []
            for t in range(T - 1):
                a_emb_t = self.action_embedding(actions[:, t])
                tr = self.transition_model.forward_step(
                    q_row=torch.softmax(row_logits_seq[:, t, :], dim=-1),
                    q_col=torch.softmax(col_logits_seq[:, t, :], dim=-1),
                    q_heading=torch.softmax(heading_logits_seq[:, t, :], dim=-1),
                    context=context_seq[:, t, :],
                    action_emb=a_emb_t,
                )
                collision_list.append(tr["collision_logits"])
            collision_logits_seq = torch.stack(collision_list, dim=1)

            transition_prior_row_probs    = torch.zeros(B, T - 1, self.cfg.num_row_classes, device=obs_seq.device)
            transition_prior_col_probs    = torch.zeros(B, T - 1, self.cfg.num_col_classes, device=obs_seq.device)
            transition_prior_heading_probs = torch.zeros(B, T - 1, self.cfg.num_heading_classes, device=obs_seq.device)

        row_probs_seq    = torch.softmax(row_logits_seq, dim=-1)
        col_probs_seq    = torch.softmax(col_logits_seq, dim=-1)
        heading_probs_seq = torch.softmax(heading_logits_seq, dim=-1)

        # ---- Reconstruction (training only — skip for inference) ----
        if skip_recon or not self.training:
            recon_seq = None
        else:
            q_row_flat     = row_probs_seq.reshape(B * T, -1)
            q_col_flat     = col_probs_seq.reshape(B * T, -1)
            q_heading_flat = heading_probs_seq.reshape(B * T, -1)
            context_flat   = context_seq.reshape(B * T, -1)
            recon_seq      = self.decoder(q_row_flat, q_col_flat, q_heading_flat, context_flat).view(B, T, C, H, W)

        return {
            "feat_seq": feat_seq,
            "h_seq": h_seq,
            "h_last": h_last,
            "row_logits_seq": row_logits_seq,
            "col_logits_seq": col_logits_seq,
            "heading_logits_seq": heading_logits_seq,
            "row_probs_seq": row_probs_seq,
            "col_probs_seq": col_probs_seq,
            "heading_probs_seq": heading_probs_seq,
            "context_seq": context_seq,
            "recon_seq": recon_seq,
            "collision_logits_seq": collision_logits_seq,
            "encoder_row_logits_seq": enc_row_logits_seq,
            "encoder_col_logits_seq": enc_col_logits_seq,
            "encoder_heading_logits_seq": enc_heading_logits_seq,
            "transition_prior_row_probs": transition_prior_row_probs,
            "transition_prior_col_probs": transition_prior_col_probs,
            "transition_prior_heading_probs": transition_prior_heading_probs,
            "context_mu_seq": context_mu_seq,
            "context_logvar_seq": context_logvar_seq,
        }

    # --------------------------------------------------------
    # Online single-step inference  (NEW in V5)
    # --------------------------------------------------------

    def forward_step_online(
        self,
        obs_t: torch.Tensor,                       # [1, C, H, W]
        action_prev: Optional[torch.Tensor],        # [1] long, or None for first step
        h_prev: Optional[torch.Tensor] = None,     # (1, 1, gru_hidden_dim) — GRU cache
        prev_belief: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Process ONE new observation and return an updated belief + new GRU state.

        This is O(1) per step: no full-history replay, no attention over T frames.
        The caller stores h_t and passes it back each step — that is the entire
        temporal state.

        Args:
            obs_t:       Single frame [1, C, H, W]
            action_prev: Action taken to arrive at this frame; None on first step
            h_prev:      Cached GRU hidden state (1, 1, H); None on first step → zeros
            prev_belief: Posterior from last step (dict with row_probs, col_probs,
                         heading_probs, context); None on first step → no Bayesian update

        Returns:
            belief: dict with row_probs, col_probs, heading_probs, context,
                    context_mu, context_logvar  — all [1, *]
            h_new:  Updated GRU state (1, 1, H) — pass back next call
        """
        device = obs_t.device

        feat = self.encoder(obs_t)   # [1, F]

        if action_prev is None:
            action_prev = torch.zeros(1, dtype=torch.long, device=device)
        a_emb = self.action_embedding(action_prev)  # [1, A]

        x = self.filter_input_proj(torch.cat([feat, a_emb], dim=-1))  # [1, H]

        # Single GRU step — O(1), no sequence processing
        h_seq, h_new = self.filter_gru(x.unsqueeze(1), h_prev)  # [1,1,H], [1,1,H]
        h_t = h_seq[:, 0, :]   # [1, H]

        factors = self.factor_heads(h_t)

        # Bayesian update if we have a previous belief and a transition prior
        if prev_belief is not None and self.cfg.use_predictive_prior:
            tr_out = self.transition_model.forward_step(
                q_row=prev_belief["row_probs"],
                q_col=prev_belief["col_probs"],
                q_heading=prev_belief["heading_probs"],
                context=prev_belief["context"],
                action_emb=a_emb,
            )
            eps = 1e-8
            log_prior_row    = torch.log(tr_out["next_row_probs"].clamp_min(eps))
            log_prior_col    = torch.log(tr_out["next_col_probs"].clamp_min(eps))
            log_prior_heading = torch.log(tr_out["next_heading_probs"].clamp_min(eps))

            log_lik_row    = F.log_softmax(factors["row_logits"], dim=-1)
            log_lik_col    = F.log_softmax(factors["col_logits"], dim=-1)
            log_lik_heading = F.log_softmax(factors["heading_logits"], dim=-1)

            row_probs    = F.softmax(log_prior_row + log_lik_row, dim=-1)
            col_probs    = F.softmax(log_prior_col + log_lik_col, dim=-1)
            heading_probs = F.softmax(log_prior_heading + log_lik_heading, dim=-1)
        else:
            row_probs    = F.softmax(factors["row_logits"], dim=-1)
            col_probs    = F.softmax(factors["col_logits"], dim=-1)
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

    # --------------------------------------------------------
    # Rollout pass  (identical to V4)
    # --------------------------------------------------------

    def rollout_from_filtered_state(
        self,
        row_probs_start: torch.Tensor,
        col_probs_start: torch.Tensor,
        heading_probs_start: torch.Tensor,
        context_start: torch.Tensor,
        action_seq: torch.Tensor,
        skip_recon: bool = False,   # True during planning — decoder output unused
    ) -> Dict[str, torch.Tensor]:
        B, K = action_seq.shape

        q_row, q_col, q_heading, context = (
            row_probs_start, col_probs_start, heading_probs_start, context_start,
        )

        row_logits_roll, col_logits_roll, heading_logits_roll = [], [], []
        row_probs_roll, col_probs_roll, heading_probs_roll    = [], [], []
        context_roll, collision_logits_roll, recon_roll       = [], [], []

        for k in range(K):
            a_emb_t = self.action_embedding(action_seq[:, k])
            step_out = self.transition_model.forward_step(
                q_row=q_row, q_col=q_col, q_heading=q_heading,
                context=context, action_emb=a_emb_t,
            )
            q_row     = step_out["next_row_probs"]
            q_col     = step_out["next_col_probs"]
            q_heading = step_out["next_heading_probs"]
            context   = step_out["next_context"]

            if not skip_recon:
                recon_roll.append(self.decoder(q_row, q_col, q_heading, context))

            row_logits_roll.append(step_out["next_row_logits"])
            col_logits_roll.append(step_out["next_col_logits"])
            heading_logits_roll.append(step_out["next_heading_logits"])
            row_probs_roll.append(q_row)
            col_probs_roll.append(q_col)
            heading_probs_roll.append(q_heading)
            context_roll.append(context)
            collision_logits_roll.append(step_out["collision_logits"])

        out = {
            "row_logits_roll":       torch.stack(row_logits_roll, dim=1),
            "col_logits_roll":       torch.stack(col_logits_roll, dim=1),
            "heading_logits_roll":   torch.stack(heading_logits_roll, dim=1),
            "row_probs_roll":        torch.stack(row_probs_roll, dim=1),
            "col_probs_roll":        torch.stack(col_probs_roll, dim=1),
            "heading_probs_roll":    torch.stack(heading_probs_roll, dim=1),
            "context_roll":          torch.stack(context_roll, dim=1),
            "collision_logits_roll": torch.stack(collision_logits_roll, dim=1),
            "recon_roll":            torch.stack(recon_roll, dim=1) if recon_roll else None,
        }
        return out


# ============================================================
# Quick shape test
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = ModelV5Config()
    model = WorldModelV5(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    B, T, K = 4, 16, 5
    obs  = torch.randn(B, T, 1, 64, 64, device=device)
    acts = torch.randint(0, 4, (B, T - 1), device=device)

    # --- Training mode (decoder runs) ---
    model.train()
    filt = model.forward_filter(obs, acts)
    assert filt["recon_seq"] is not None, "Decoder should run in train mode"
    print("Train forward_filter recon_seq:", tuple(filt["recon_seq"].shape))

    # --- Eval mode (decoder skipped) ---
    model.eval()
    with torch.no_grad():
        filt2 = model.forward_filter(obs, acts)
    assert filt2["recon_seq"] is None, "Decoder should be skipped in eval mode"
    print("Eval forward_filter recon_seq: None (correct)")
    print("row_probs_seq:", tuple(filt2["row_probs_seq"].shape))
    print("context_seq  :", tuple(filt2["context_seq"].shape))

    # --- Online single-step inference ---
    print("\nOnline step test:")
    h_state = None
    prev_belief = None
    for step in range(4):
        obs_t = torch.randn(1, 1, 64, 64, device=device)
        action_prev = None if step == 0 else torch.tensor([1], dtype=torch.long, device=device)
        with torch.no_grad():
            belief, h_state = model.forward_step_online(obs_t, action_prev, h_state, prev_belief)
        prev_belief = belief
        print(f"  step {step}: row_probs={tuple(belief['row_probs'].shape)}, h_state={tuple(h_state.shape)}")

    # --- Rollout ---
    with torch.no_grad():
        roll = model.rollout_from_filtered_state(
            row_probs_start=filt2["row_probs_seq"][:, -1, :],
            col_probs_start=filt2["col_probs_seq"][:, -1, :],
            heading_probs_start=filt2["heading_probs_seq"][:, -1, :],
            context_start=filt2["context_seq"][:, -1, :],
            action_seq=acts[:, :K],
        )
    print("\nRollout row_probs_roll:", tuple(roll["row_probs_roll"].shape))

    kl = model.compute_context_kl(filt["context_mu_seq"], filt["context_logvar_seq"])
    print(f"\nContext KL (train): {kl.item():.4f}")
    print("\nAll shape checks passed.")

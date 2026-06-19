from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class ModelV4Config:
    # Observation shape
    obs_channels: int = 1
    obs_height: int = 64
    obs_width: int = 64

    # Environment state factors
    num_actions: int = 4
    num_row_classes: int = 9
    num_col_classes: int = 9
    num_heading_classes: int = 4

    # Encoder / filter backbone
    action_emb_dim: int = 16
    encoder_feat_dim: int = 128
    gru_hidden_dim: int = 256
    attn_num_heads: int = 4
    attn_num_layers: int = 2
    attn_ff_dim: int = 512
    attn_dropout: float = 0.1
    max_seq_len: int = 128

    # Factorized state representation
    context_dim: int = 64
    factor_hidden_dim: int = 128
    transition_hidden_dim: int = 128
    collision_hidden_dim: int = 128

    # Decoder
    decoder_hidden_dim: int = 256
    decoder_base_channels: int = 128

    # V4 additions — uncertainty and predictive prior
    use_predictive_prior: bool = True   # Bayesian update during filtering
    kl_anneal_start: float = 0.0        # starting KL weight (annealed during training)
    kl_anneal_end: float = 0.001        # final context KL weight
    vfe_kl_weight: float = 0.1          # weight for predictive-prior KL loss

    # V4b: free bits per context latent dimension.
    # Prevents posterior collapse after the context KL is properly regularised.
    # Each latent dim must contribute at least this many nats of KL on average.
    # 0.0 disables (V4 behaviour); 0.05 is a sensible default for V4b.
    context_free_bits: float = 0.05


# ============================================================
# Small blocks  (identical to V3)
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
        x = self.relu(x + residual)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1], :]


# ============================================================
# Encoder and temporal filter  (identical to V3)
# ============================================================

class ObservationEncoderV4(nn.Module):
    def __init__(self, cfg: ModelV4Config) -> None:
        super().__init__()
        c = cfg.decoder_base_channels // 4
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


class TemporalAttentionModule(nn.Module):
    def __init__(self, cfg: ModelV4Config) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(cfg.gru_hidden_dim, cfg.max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.gru_hidden_dim,
            nhead=cfg.attn_num_heads,
            dim_feedforward=cfg.attn_ff_dim,
            dropout=cfg.attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.attn_num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        return self.encoder(x)


# ============================================================
# Factor heads  (V4: stochastic context via VAE)
# ============================================================

class FactorHeadsV4(nn.Module):
    """
    From filtered hidden state, infer explicit factor beliefs:
      - row logits
      - col logits
      - heading logits
      - stochastic context vector (VAE: mu + logvar, reparameterisation trick)

    The stochastic context allows the model to represent genuine uncertainty
    about the continuous part of the state.  During eval, the mean is used.
    """
    def __init__(self, cfg: ModelV4Config) -> None:
        super().__init__()
        H = cfg.gru_hidden_dim
        Fh = cfg.factor_hidden_dim

        self.row_head = nn.Sequential(
            nn.Linear(H, Fh),
            nn.ReLU(inplace=True),
            nn.Linear(Fh, cfg.num_row_classes),
        )
        self.col_head = nn.Sequential(
            nn.Linear(H, Fh),
            nn.ReLU(inplace=True),
            nn.Linear(Fh, cfg.num_col_classes),
        )
        self.heading_head = nn.Sequential(
            nn.Linear(H, Fh),
            nn.ReLU(inplace=True),
            nn.Linear(Fh, cfg.num_heading_classes),
        )
        # Stochastic context: separate mu and logvar heads
        self.context_mu_head = nn.Sequential(
            nn.Linear(H, Fh),
            nn.ReLU(inplace=True),
            nn.Linear(Fh, cfg.context_dim),
        )
        self.context_logvar_head = nn.Sequential(
            nn.Linear(H, Fh),
            nn.ReLU(inplace=True),
            nn.Linear(Fh, cfg.context_dim),
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu = self.context_mu_head(h)
        logvar = self.context_logvar_head(h).clamp(-4.0, 4.0)

        if self.training:
            # Reparameterisation trick: z = mu + eps * sigma
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            context = mu + eps * std
        else:
            # Use mean at eval time for deterministic, reproducible inference
            context = mu

        return {
            "row_logits": self.row_head(h),
            "col_logits": self.col_head(h),
            "heading_logits": self.heading_head(h),
            "context": context,
            "context_mu": mu,
            "context_logvar": logvar,
        }


# ============================================================
# Transition model  (identical to V3)
# ============================================================

class FactorizedTransitionModelV4(nn.Module):
    """
    Separate transition heads for row / col / heading / context / collision.

    State input:  q_row, q_col, q_heading, context, action
    Outputs:
      - move candidate next row/col/heading/context
      - collision probability
      - collision-aware soft blend of move and stay candidates
    """
    def __init__(self, cfg: ModelV4Config) -> None:
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

    def _build_state_input(
        self,
        q_row: torch.Tensor,
        q_col: torch.Tensor,
        q_heading: torch.Tensor,
        context: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([q_row, q_col, q_heading, context, action_emb], dim=-1)

    def forward_step(
        self,
        q_row: torch.Tensor,       # [B, R]
        q_col: torch.Tensor,       # [B, C]
        q_heading: torch.Tensor,   # [B, Hd]
        context: torch.Tensor,     # [B, U]
        action_emb: torch.Tensor,  # [B, A]
    ) -> Dict[str, torch.Tensor]:
        state_in = self._build_state_input(q_row, q_col, q_heading, context, action_emb)

        shared = self.shared(state_in)

        move_row_logits = self.row_head(shared)
        move_col_logits = self.col_head(shared)
        move_heading_logits = self.heading_head(shared)
        move_context = self.context_head(shared)

        collision_logits = self.collision_head(state_in)
        collision_prob = torch.softmax(collision_logits, dim=-1)[:, 1:2]  # [B,1]

        stay_row_probs = q_row
        stay_col_probs = q_col
        stay_heading_probs = q_heading
        stay_context = context

        move_row_probs = torch.softmax(move_row_logits, dim=-1)
        move_col_probs = torch.softmax(move_col_logits, dim=-1)
        move_heading_probs = torch.softmax(move_heading_logits, dim=-1)

        next_row_probs = (1.0 - collision_prob) * move_row_probs + collision_prob * stay_row_probs
        next_col_probs = (1.0 - collision_prob) * move_col_probs + collision_prob * stay_col_probs
        next_heading_probs = (1.0 - collision_prob) * move_heading_probs + collision_prob * stay_heading_probs
        next_context = (1.0 - collision_prob) * move_context + collision_prob * stay_context

        eps = 1e-8
        next_row_logits = torch.log(next_row_probs.clamp_min(eps))
        next_col_logits = torch.log(next_col_probs.clamp_min(eps))
        next_heading_logits = torch.log(next_heading_probs.clamp_min(eps))

        return {
            "move_row_logits": move_row_logits,
            "move_col_logits": move_col_logits,
            "move_heading_logits": move_heading_logits,
            "move_context": move_context,
            "collision_logits": collision_logits,
            "next_row_logits": next_row_logits,
            "next_col_logits": next_col_logits,
            "next_heading_logits": next_heading_logits,
            "next_row_probs": next_row_probs,
            "next_col_probs": next_col_probs,
            "next_heading_probs": next_heading_probs,
            "next_context": next_context,
        }


# ============================================================
# Decoder  (identical to V3)
# ============================================================

class ObservationDecoderV4(nn.Module):
    """Decode structured state (q_row, q_col, q_heading, context) → image."""

    def __init__(self, cfg: ModelV4Config) -> None:
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

        self.block_4 = nn.Sequential(
            ResidualConvBlock(base),
            ResidualConvBlock(base),
        )
        self.up_8 = nn.Sequential(
            nn.ConvTranspose2d(base, base // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base // 2),
            nn.ReLU(inplace=True),
            ResidualConvBlock(base // 2),
        )
        self.up_16 = nn.Sequential(
            nn.ConvTranspose2d(base // 2, base // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base // 4),
            nn.ReLU(inplace=True),
            ResidualConvBlock(base // 4),
        )
        self.up_32 = nn.Sequential(
            nn.ConvTranspose2d(base // 4, base // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base // 8),
            nn.ReLU(inplace=True),
            ResidualConvBlock(base // 8),
        )
        self.up_64 = nn.Sequential(
            nn.ConvTranspose2d(base // 8, base // 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base // 16),
            nn.ReLU(inplace=True),
            ResidualConvBlock(base // 16),
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
        x = self.out_conv(x)
        return torch.sigmoid(x)


# ============================================================
# Main model
# ============================================================

class WorldModelV4(nn.Module):
    """
    V4 design — builds on V3 with two key uncertainty improvements:

    1. Stochastic context (VAE): the continuous context vector is sampled
       from N(mu, sigma) during training.  KL(N(mu,sigma)||N(0,1)) is added
       to the loss.  This lets the model express genuine uncertainty about
       the unstructured aspects of the scene.

    2. Predictive prior (Bayesian update): during the filtering pass, for
       each t >= 1 the posterior belief is computed as:

           log Q(s_t | o_{1:t}) ∝ log P(s_t | s_{t-1}, a_{t-1})   [transition prior]
                                  + log P(o_t | s_t)               [encoder likelihood]

       This is the core of predictive processing / active inference: every
       observation is interpreted against what the transition model expected.
       When use_predictive_prior=False the model degrades to V3 behaviour.
    """

    def __init__(self, cfg: ModelV4Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.action_embedding = nn.Embedding(cfg.num_actions, cfg.action_emb_dim)
        self.encoder = ObservationEncoderV4(cfg)

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
        self.temporal_attention = TemporalAttentionModule(cfg)

        self.factor_heads = FactorHeadsV4(cfg)
        self.decoder = ObservationDecoderV4(cfg)
        self.transition_model = FactorizedTransitionModelV4(cfg)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def encode_observation_sequence(self, obs_seq: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = obs_seq.shape
        x = obs_seq.view(B * T, C, H, W)
        feat = self.encoder(x)
        return feat.view(B, T, -1)

    def build_filter_inputs(self, feat_seq: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, T, _ = feat_seq.shape
        device = feat_seq.device
        zero_action = torch.zeros(B, 1, dtype=torch.long, device=device)
        prev_actions = torch.cat([zero_action, actions], dim=1)[:, :T]
        prev_a_emb = self.action_embedding(prev_actions)
        x = torch.cat([feat_seq, prev_a_emb], dim=-1)
        return self.filter_input_proj(x)

    def logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def compute_context_kl(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL( N(mu, sigma) || N(0, I) ) with optional free-bits floor.

        Free bits: average KL over batch+time per latent dimension, then clamp
        each dimension from below at cfg.context_free_bits nats.  This prevents
        posterior collapse (any dimension silently shrinking to KL≈0) without
        penalising dimensions that are already well-regularised.
        Returns a scalar (sum over dimensions).
        """
        # [B, T, D]
        kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())

        if self.cfg.context_free_bits > 0.0:
            # Average over batch and time → [D], then clamp per dim
            kl_per_dim = kl.mean(dim=[0, 1])
            kl_per_dim = kl_per_dim.clamp_min(self.cfg.context_free_bits)
            return kl_per_dim.sum()

        return kl.sum(dim=-1).mean()

    # --------------------------------------------------------
    # Filtering pass over real observations
    # --------------------------------------------------------

    def forward_filter(
        self,
        obs_seq: torch.Tensor,    # [B, T, C, H, W]
        actions: torch.Tensor,    # [B, T-1]
        h0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = obs_seq.shape
        if actions.shape[1] != T - 1:
            raise ValueError(
                f"Expected actions [B,T-1], got {tuple(actions.shape)} for T={T}"
            )

        # ---- Encode all observations ----
        feat_seq = self.encode_observation_sequence(obs_seq)         # [B,T,F]
        filter_inputs = self.build_filter_inputs(feat_seq, actions)  # [B,T,H]

        h_seq_gru, h_last = self.filter_gru(filter_inputs, h0)       # [B,T,H]
        h_seq = self.temporal_attention(h_seq_gru)                   # [B,T,H]

        # ---- Raw encoder factor beliefs ----
        h_flat = h_seq.reshape(B * T, -1)
        factors_flat = self.factor_heads(h_flat)

        enc_row_logits_seq = factors_flat["row_logits"].view(B, T, -1)
        enc_col_logits_seq = factors_flat["col_logits"].view(B, T, -1)
        enc_heading_logits_seq = factors_flat["heading_logits"].view(B, T, -1)
        context_seq = factors_flat["context"].view(B, T, -1)
        context_mu_seq = factors_flat["context_mu"].view(B, T, -1)
        context_logvar_seq = factors_flat["context_logvar"].view(B, T, -1)

        # ---- Bayesian update: combine transition prior with encoder likelihood ----
        if self.cfg.use_predictive_prior:
            # Posterior beliefs accumulate timestep by timestep
            post_row_logits_list = [enc_row_logits_seq[:, 0:1, :]]      # t=0: no prior
            post_col_logits_list = [enc_col_logits_seq[:, 0:1, :]]
            post_heading_logits_list = [enc_heading_logits_seq[:, 0:1, :]]

            trans_prior_row_list = []
            trans_prior_col_list = []
            trans_prior_heading_list = []

            # Posterior probs for previous step (needed by transition model)
            eps = 1e-8
            prev_row_probs = self.logits_to_probs(enc_row_logits_seq[:, 0, :])
            prev_col_probs = self.logits_to_probs(enc_col_logits_seq[:, 0, :])
            prev_heading_probs = self.logits_to_probs(enc_heading_logits_seq[:, 0, :])
            prev_context = context_seq[:, 0, :]

            for t in range(1, T):
                a_emb_t = self.action_embedding(actions[:, t - 1])  # action taken at t-1

                tr_out = self.transition_model.forward_step(
                    q_row=prev_row_probs,
                    q_col=prev_col_probs,
                    q_heading=prev_heading_probs,
                    context=prev_context,
                    action_emb=a_emb_t,
                )

                prior_row_probs = tr_out["next_row_probs"]    # [B, R]
                prior_col_probs = tr_out["next_col_probs"]    # [B, C]
                prior_heading_probs = tr_out["next_heading_probs"]  # [B, Hd]

                trans_prior_row_list.append(prior_row_probs)
                trans_prior_col_list.append(prior_col_probs)
                trans_prior_heading_list.append(prior_heading_probs)

                # log-space Bayesian update: log_posterior = log_prior + log_likelihood
                log_prior_row = torch.log(prior_row_probs.clamp_min(eps))
                log_prior_col = torch.log(prior_col_probs.clamp_min(eps))
                log_prior_heading = torch.log(prior_heading_probs.clamp_min(eps))

                log_lik_row = F.log_softmax(enc_row_logits_seq[:, t, :], dim=-1)
                log_lik_col = F.log_softmax(enc_col_logits_seq[:, t, :], dim=-1)
                log_lik_heading = F.log_softmax(enc_heading_logits_seq[:, t, :], dim=-1)

                post_row_probs = F.softmax(log_prior_row + log_lik_row, dim=-1)
                post_col_probs = F.softmax(log_prior_col + log_lik_col, dim=-1)
                post_heading_probs = F.softmax(log_prior_heading + log_lik_heading, dim=-1)

                # Store as log-probs (logits) for supervision
                post_row_logits_list.append(torch.log(post_row_probs.clamp_min(eps)).unsqueeze(1))
                post_col_logits_list.append(torch.log(post_col_probs.clamp_min(eps)).unsqueeze(1))
                post_heading_logits_list.append(torch.log(post_heading_probs.clamp_min(eps)).unsqueeze(1))

                # These posterior probs feed the next transition prior
                prev_row_probs = post_row_probs
                prev_col_probs = post_col_probs
                prev_heading_probs = post_heading_probs
                prev_context = context_seq[:, t, :]

            row_logits_seq = torch.cat(post_row_logits_list, dim=1)           # [B,T,R]
            col_logits_seq = torch.cat(post_col_logits_list, dim=1)           # [B,T,C]
            heading_logits_seq = torch.cat(post_heading_logits_list, dim=1)   # [B,T,Hd]

            transition_prior_row_probs = torch.stack(trans_prior_row_list, dim=1)       # [B,T-1,R]
            transition_prior_col_probs = torch.stack(trans_prior_col_list, dim=1)       # [B,T-1,C]
            transition_prior_heading_probs = torch.stack(trans_prior_heading_list, dim=1)  # [B,T-1,Hd]
        else:
            # V3 fallback: use raw encoder logits directly
            row_logits_seq = enc_row_logits_seq
            col_logits_seq = enc_col_logits_seq
            heading_logits_seq = enc_heading_logits_seq

            # Provide empty tensors so downstream code doesn't need special casing
            transition_prior_row_probs = torch.zeros(B, T - 1, self.cfg.num_row_classes,
                                                     device=obs_seq.device)
            transition_prior_col_probs = torch.zeros(B, T - 1, self.cfg.num_col_classes,
                                                     device=obs_seq.device)
            transition_prior_heading_probs = torch.zeros(B, T - 1, self.cfg.num_heading_classes,
                                                         device=obs_seq.device)

        row_probs_seq = self.logits_to_probs(row_logits_seq)
        col_probs_seq = self.logits_to_probs(col_logits_seq)
        heading_probs_seq = self.logits_to_probs(heading_logits_seq)

        # ---- Reconstruction from posterior beliefs ----
        q_row_flat = row_probs_seq.reshape(B * T, -1)
        q_col_flat = col_probs_seq.reshape(B * T, -1)
        q_heading_flat = heading_probs_seq.reshape(B * T, -1)
        context_flat = context_seq.reshape(B * T, -1)

        recon_flat = self.decoder(q_row_flat, q_col_flat, q_heading_flat, context_flat)
        recon_seq = recon_flat.view(B, T, C, H, W)

        # ---- Collision prediction on real filtered states ----
        collision_logits_list = []
        for t in range(T - 1):
            a_emb_t = self.action_embedding(actions[:, t])
            tr = self.transition_model.forward_step(
                q_row=row_probs_seq[:, t, :],
                q_col=col_probs_seq[:, t, :],
                q_heading=heading_probs_seq[:, t, :],
                context=context_seq[:, t, :],
                action_emb=a_emb_t,
            )
            collision_logits_list.append(tr["collision_logits"])
        collision_logits_seq = torch.stack(collision_logits_list, dim=1)  # [B,T-1,2]

        return {
            # Core beliefs (posterior if use_predictive_prior, else raw encoder)
            "feat_seq": feat_seq,
            "h_seq_gru": h_seq_gru,
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
            # Raw encoder logits (pre-Bayesian-update) — useful for debugging
            "encoder_row_logits_seq": enc_row_logits_seq,
            "encoder_col_logits_seq": enc_col_logits_seq,
            "encoder_heading_logits_seq": enc_heading_logits_seq,
            # Transition priors for VFE KL loss [B, T-1, ...]
            "transition_prior_row_probs": transition_prior_row_probs,
            "transition_prior_col_probs": transition_prior_col_probs,
            "transition_prior_heading_probs": transition_prior_heading_probs,
            # Context VAE parameters for KL loss [B, T, context_dim]
            "context_mu_seq": context_mu_seq,
            "context_logvar_seq": context_logvar_seq,
        }

    # --------------------------------------------------------
    # Rollout pass (identical to V3)
    # --------------------------------------------------------

    def rollout_from_filtered_state(
        self,
        row_probs_start: torch.Tensor,      # [B, R]
        col_probs_start: torch.Tensor,      # [B, C]
        heading_probs_start: torch.Tensor,  # [B, Hd]
        context_start: torch.Tensor,        # [B, U]
        action_seq: torch.Tensor,           # [B, K]
    ) -> Dict[str, torch.Tensor]:
        B, K = action_seq.shape

        q_row = row_probs_start
        q_col = col_probs_start
        q_heading = heading_probs_start
        context = context_start

        row_logits_roll = []
        col_logits_roll = []
        heading_logits_roll = []
        row_probs_roll = []
        col_probs_roll = []
        heading_probs_roll = []
        context_roll = []
        collision_logits_roll = []
        recon_roll = []

        for k in range(K):
            a_t = action_seq[:, k]
            a_emb_t = self.action_embedding(a_t)

            step_out = self.transition_model.forward_step(
                q_row=q_row,
                q_col=q_col,
                q_heading=q_heading,
                context=context,
                action_emb=a_emb_t,
            )

            q_row = step_out["next_row_probs"]
            q_col = step_out["next_col_probs"]
            q_heading = step_out["next_heading_probs"]
            context = step_out["next_context"]

            recon_t = self.decoder(q_row, q_col, q_heading, context)

            row_logits_roll.append(step_out["next_row_logits"])
            col_logits_roll.append(step_out["next_col_logits"])
            heading_logits_roll.append(step_out["next_heading_logits"])
            row_probs_roll.append(q_row)
            col_probs_roll.append(q_col)
            heading_probs_roll.append(q_heading)
            context_roll.append(context)
            collision_logits_roll.append(step_out["collision_logits"])
            recon_roll.append(recon_t)

        return {
            "row_logits_roll": torch.stack(row_logits_roll, dim=1),
            "col_logits_roll": torch.stack(col_logits_roll, dim=1),
            "heading_logits_roll": torch.stack(heading_logits_roll, dim=1),
            "row_probs_roll": torch.stack(row_probs_roll, dim=1),
            "col_probs_roll": torch.stack(col_probs_roll, dim=1),
            "heading_probs_roll": torch.stack(heading_probs_roll, dim=1),
            "context_roll": torch.stack(context_roll, dim=1),
            "collision_logits_roll": torch.stack(collision_logits_roll, dim=1),
            "recon_roll": torch.stack(recon_roll, dim=1),
        }


# ============================================================
# Quick shape test
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = ModelV4Config(
        obs_channels=1,
        obs_height=64,
        obs_width=64,
        num_actions=4,
        num_row_classes=9,
        num_col_classes=9,
        num_heading_classes=4,
        action_emb_dim=16,
        encoder_feat_dim=128,
        gru_hidden_dim=256,
        attn_num_heads=4,
        attn_num_layers=2,
        attn_ff_dim=512,
        attn_dropout=0.1,
        max_seq_len=128,
        context_dim=64,
        factor_hidden_dim=128,
        transition_hidden_dim=128,
        collision_hidden_dim=128,
        decoder_hidden_dim=256,
        decoder_base_channels=128,
        use_predictive_prior=True,
    )

    model = WorldModelV4(cfg).to(device)
    model.eval()

    B, T, K = 8, 32, 5
    obs = torch.randn(B, T, 1, 64, 64, device=device)
    actions = torch.randint(0, 4, (B, T - 1), device=device)

    with torch.no_grad():
        filt = model.forward_filter(obs, actions)

        print("\nFilter outputs:")
        print("  feat_seq                       :", tuple(filt["feat_seq"].shape))
        print("  h_seq_gru                      :", tuple(filt["h_seq_gru"].shape))
        print("  h_seq                          :", tuple(filt["h_seq"].shape))
        print("  h_last                         :", tuple(filt["h_last"].shape))
        print("  row_logits_seq (posterior)     :", tuple(filt["row_logits_seq"].shape))
        print("  col_logits_seq (posterior)     :", tuple(filt["col_logits_seq"].shape))
        print("  heading_logits_seq (posterior) :", tuple(filt["heading_logits_seq"].shape))
        print("  row_probs_seq                  :", tuple(filt["row_probs_seq"].shape))
        print("  col_probs_seq                  :", tuple(filt["col_probs_seq"].shape))
        print("  heading_probs_seq              :", tuple(filt["heading_probs_seq"].shape))
        print("  context_seq                    :", tuple(filt["context_seq"].shape))
        print("  recon_seq                      :", tuple(filt["recon_seq"].shape))
        print("  collision_logits_seq           :", tuple(filt["collision_logits_seq"].shape))
        print("  encoder_row_logits_seq         :", tuple(filt["encoder_row_logits_seq"].shape))
        print("  encoder_col_logits_seq         :", tuple(filt["encoder_col_logits_seq"].shape))
        print("  encoder_heading_logits_seq     :", tuple(filt["encoder_heading_logits_seq"].shape))
        print("  transition_prior_row_probs     :", tuple(filt["transition_prior_row_probs"].shape))
        print("  transition_prior_col_probs     :", tuple(filt["transition_prior_col_probs"].shape))
        print("  transition_prior_heading_probs :", tuple(filt["transition_prior_heading_probs"].shape))
        print("  context_mu_seq                 :", tuple(filt["context_mu_seq"].shape))
        print("  context_logvar_seq             :", tuple(filt["context_logvar_seq"].shape))

        # Context KL
        kl = model.compute_context_kl(filt["context_mu_seq"], filt["context_logvar_seq"])
        print(f"\n  context KL (should be finite): {kl.item():.4f}")

        t0 = 10
        roll = model.rollout_from_filtered_state(
            row_probs_start=filt["row_probs_seq"][:, t0, :],
            col_probs_start=filt["col_probs_seq"][:, t0, :],
            heading_probs_start=filt["heading_probs_seq"][:, t0, :],
            context_start=filt["context_seq"][:, t0, :],
            action_seq=actions[:, t0:t0 + K],
        )

        print("\nRollout outputs:")
        print("  row_logits_roll      :", tuple(roll["row_logits_roll"].shape))
        print("  col_logits_roll      :", tuple(roll["col_logits_roll"].shape))
        print("  heading_logits_roll  :", tuple(roll["heading_logits_roll"].shape))
        print("  row_probs_roll       :", tuple(roll["row_probs_roll"].shape))
        print("  col_probs_roll       :", tuple(roll["col_probs_roll"].shape))
        print("  heading_probs_roll   :", tuple(roll["heading_probs_roll"].shape))
        print("  context_roll         :", tuple(roll["context_roll"].shape))
        print("  collision_logits_roll:", tuple(roll["collision_logits_roll"].shape))
        print("  recon_roll           :", tuple(roll["recon_roll"].shape))

    # Also test use_predictive_prior=False (V3 fallback)
    cfg_v3compat = ModelV4Config(use_predictive_prior=False)
    model2 = WorldModelV4(cfg_v3compat).to(device)
    model2.eval()
    with torch.no_grad():
        filt2 = model2.forward_filter(obs, actions)
        print("\nV3-compat (use_predictive_prior=False) row_logits_seq shape:",
              tuple(filt2["row_logits_seq"].shape))
    print("\nAll shape checks passed.")

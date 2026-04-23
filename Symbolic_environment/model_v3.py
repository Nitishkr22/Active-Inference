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
class ModelV3Config:
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
# Encoder and temporal filter
# ============================================================

class ObservationEncoderV3(nn.Module):
    def __init__(self, cfg: ModelV3Config) -> None:
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
    def __init__(self, cfg: ModelV3Config) -> None:
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
# Factor heads
# ============================================================

class FactorHeadsV3(nn.Module):
    """
    From filtered hidden state, infer explicit factor beliefs:
      - row logits
      - col logits
      - heading logits
      - continuous context vector
    """
    def __init__(self, cfg: ModelV3Config) -> None:
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
        self.context_head = nn.Sequential(
            nn.Linear(H, Fh),
            nn.ReLU(inplace=True),
            nn.Linear(Fh, cfg.context_dim),
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "row_logits": self.row_head(h),
            "col_logits": self.col_head(h),
            "heading_logits": self.heading_head(h),
            "context": self.context_head(h),
        }


# ============================================================
# Transition model over factorized state
# ============================================================

class FactorizedTransitionModelV3(nn.Module):
    """
    Separate transition heads for row / col / heading / context / collision.

    State input is explicit and structured:
      q_row, q_col, q_heading, context, action

    The model predicts:
      - move candidate next row/col/heading/context
      - collision probability
      - blends move candidate with stay candidate using predicted collision

    This is intentionally more structured than V1/V2.
    """
    def __init__(self, cfg: ModelV3Config) -> None:
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
        q_row: torch.Tensor,      # [B,R]
        q_col: torch.Tensor,      # [B,C]
        q_heading: torch.Tensor,  # [B,Hd]
        context: torch.Tensor,    # [B,U]
        action_emb: torch.Tensor, # [B,A]
    ) -> Dict[str, torch.Tensor]:
        state_in = self._build_state_input(q_row, q_col, q_heading, context, action_emb)

        shared = self.shared(state_in)

        move_row_logits = self.row_head(shared)
        move_col_logits = self.col_head(shared)
        move_heading_logits = self.heading_head(shared)
        move_context = self.context_head(shared)

        collision_logits = self.collision_head(state_in)  # [B,2]
        collision_prob = torch.softmax(collision_logits, dim=-1)[:, 1:2]  # [B,1]

        # stay candidate: preserve current factor beliefs and context
        stay_row_probs = q_row
        stay_col_probs = q_col
        stay_heading_probs = q_heading
        stay_context = context

        # move candidate as probabilities for row/col/heading
        move_row_probs = torch.softmax(move_row_logits, dim=-1)
        move_col_probs = torch.softmax(move_col_logits, dim=-1)
        move_heading_probs = torch.softmax(move_heading_logits, dim=-1)

        # collision-aware soft blending
        next_row_probs = (1.0 - collision_prob) * move_row_probs + collision_prob * stay_row_probs
        next_col_probs = (1.0 - collision_prob) * move_col_probs + collision_prob * stay_col_probs
        next_heading_probs = (1.0 - collision_prob) * move_heading_probs + collision_prob * stay_heading_probs
        next_context = (1.0 - collision_prob) * move_context + collision_prob * stay_context

        # convert next probabilities back to logits for supervision convenience
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
# Decoder from factorized state
# ============================================================

class ObservationDecoderV3(nn.Module):
    """
    Decode from structured state:
      q_row, q_col, q_heading, context
    """
    def __init__(self, cfg: ModelV3Config) -> None:
        super().__init__()
        in_dim = cfg.num_row_classes + cfg.num_col_classes + cfg.num_heading_classes + cfg.context_dim
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

class WorldModelV3(nn.Module):
    """
    V3 design:
      1. CNN observation encoder
      2. GRU temporal filter
      3. Temporal self-attention over GRU states
      4. Explicit factor heads producing:
           - row logits / probs
           - col logits / probs
           - heading logits / probs
           - continuous context vector
      5. Decoder from factorized state
      6. Factorized transition model with collision-aware blending

    Main motivation:
      - stop relying on a single monolithic continuous latent for exact state identity
      - make row / col / heading explicit and directly supervised
      - better align representation with active-inference style state factors
    """
    def __init__(self, cfg: ModelV3Config) -> None:
        super().__init__()
        self.cfg = cfg

        self.action_embedding = nn.Embedding(cfg.num_actions, cfg.action_emb_dim)
        self.encoder = ObservationEncoderV3(cfg)

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

        self.factor_heads = FactorHeadsV3(cfg)
        self.decoder = ObservationDecoderV3(cfg)
        self.transition_model = FactorizedTransitionModelV3(cfg)

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

    # --------------------------------------------------------
    # Filtering pass over real observations
    # --------------------------------------------------------

    def forward_filter(
        self,
        obs_seq: torch.Tensor,   # [B,T,C,H,W]
        actions: torch.Tensor,   # [B,T-1]
        h0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = obs_seq.shape
        if actions.shape[1] != T - 1:
            raise ValueError(f"Expected actions [B,T-1], got {tuple(actions.shape)} for T={T}")

        feat_seq = self.encode_observation_sequence(obs_seq)          # [B,T,F]
        filter_inputs = self.build_filter_inputs(feat_seq, actions)   # [B,T,H]

        h_seq_gru, h_last = self.filter_gru(filter_inputs, h0)        # [B,T,H], [1,B,H]
        h_seq = self.temporal_attention(h_seq_gru)                    # [B,T,H]

        # factor inference
        h_flat = h_seq.reshape(B * T, -1)
        factors_flat = self.factor_heads(h_flat)

        row_logits_seq = factors_flat["row_logits"].view(B, T, -1)
        col_logits_seq = factors_flat["col_logits"].view(B, T, -1)
        heading_logits_seq = factors_flat["heading_logits"].view(B, T, -1)
        context_seq = factors_flat["context"].view(B, T, -1)

        row_probs_seq = self.logits_to_probs(row_logits_seq)
        col_probs_seq = self.logits_to_probs(col_logits_seq)
        heading_probs_seq = self.logits_to_probs(heading_logits_seq)

        # reconstruction from structured state
        q_row_flat = row_probs_seq.reshape(B * T, -1)
        q_col_flat = col_probs_seq.reshape(B * T, -1)
        q_heading_flat = heading_probs_seq.reshape(B * T, -1)
        context_flat = context_seq.reshape(B * T, -1)

        recon_flat = self.decoder(q_row_flat, q_col_flat, q_heading_flat, context_flat)
        recon_seq = recon_flat.view(B, T, C, H, W)

        # collision prediction on filtered real states for real actions a_t
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
        }

    # --------------------------------------------------------
    # Rollout pass from filtered structured state
    # --------------------------------------------------------

    def rollout_from_filtered_state(
        self,
        row_probs_start: torch.Tensor,      # [B,R]
        col_probs_start: torch.Tensor,      # [B,C]
        heading_probs_start: torch.Tensor,  # [B,Hd]
        context_start: torch.Tensor,        # [B,U]
        action_seq: torch.Tensor,           # [B,K]
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

    cfg = ModelV3Config(
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
    )

    model = WorldModelV3(cfg).to(device)
    model.eval()

    B, T, K = 8, 32, 5
    obs = torch.randn(B, T, 1, 64, 64, device=device)
    actions = torch.randint(0, 4, (B, T - 1), device=device)

    with torch.no_grad():
        filt = model.forward_filter(obs, actions)

        print("Filter outputs:")
        print("  feat_seq             :", tuple(filt["feat_seq"].shape))
        print("  h_seq_gru            :", tuple(filt["h_seq_gru"].shape))
        print("  h_seq                :", tuple(filt["h_seq"].shape))
        print("  h_last               :", tuple(filt["h_last"].shape))
        print("  row_logits_seq       :", tuple(filt["row_logits_seq"].shape))
        print("  col_logits_seq       :", tuple(filt["col_logits_seq"].shape))
        print("  heading_logits_seq   :", tuple(filt["heading_logits_seq"].shape))
        print("  row_probs_seq        :", tuple(filt["row_probs_seq"].shape))
        print("  col_probs_seq        :", tuple(filt["col_probs_seq"].shape))
        print("  heading_probs_seq    :", tuple(filt["heading_probs_seq"].shape))
        print("  context_seq          :", tuple(filt["context_seq"].shape))
        print("  recon_seq            :", tuple(filt["recon_seq"].shape))
        print("  collision_logits_seq :", tuple(filt["collision_logits_seq"].shape))

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

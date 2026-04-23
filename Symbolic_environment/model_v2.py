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
class ModelV2Config:
    # Observation shape
    obs_channels: int = 1
    obs_height: int = 64
    obs_width: int = 64

    # Action/state sizes
    num_actions: int = 4
    num_row_classes: int = 9
    num_col_classes: int = 9
    num_heading_classes: int = 4

    # Encoder / sequence model sizes
    action_emb_dim: int = 16
    encoder_feat_dim: int = 128
    gru_hidden_dim: int = 256
    attn_num_heads: int = 4
    attn_num_layers: int = 2
    attn_ff_dim: int = 512
    attn_dropout: float = 0.1
    latent_dim: int = 128

    # Decoder sizes
    decoder_base_channels: int = 128

    # Collision head
    collision_head_hidden_dim: int = 128

    # Positional encoding for temporal attention
    max_seq_len: int = 128


# ============================================================
# Small building blocks
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
    """
    Standard sinusoidal positional encoding for temporal attention.
    """
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.shape[1]
        return x + self.pe[:, :T, :]


# ============================================================
# Encoder
# ============================================================

class ObservationEncoderV2(nn.Module):
    """
    Stronger encoder than V1.

    Input:  [B, C, H, W]
    Output: [B, encoder_feat_dim]
    """
    def __init__(self, cfg: ModelV2Config) -> None:
        super().__init__()
        c = cfg.decoder_base_channels // 4  # usually 32 if base is 128

        self.net = nn.Sequential(
            ConvBlock(cfg.obs_channels, c, stride=2),         # 64 -> 32
            ConvBlock(c, c * 2, stride=2),                    # 32 -> 16
            ResidualConvBlock(c * 2),
            ConvBlock(c * 2, c * 4, stride=2),                # 16 -> 8
            ResidualConvBlock(c * 4),
            ConvBlock(c * 4, c * 4, stride=2),                # 8 -> 4
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
        x = self.net(x)
        x = self.proj(x)
        return x


# ============================================================
# Temporal attention block
# ============================================================

class TemporalAttentionModule(nn.Module):
    """
    Applies self-attention over the sequence of GRU states.
    Input:  [B, T, H]
    Output: [B, T, H]
    """
    def __init__(self, cfg: ModelV2Config) -> None:
        super().__init__()
        self.pos_enc = PositionalEncoding(cfg.gru_hidden_dim, max_len=cfg.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.gru_hidden_dim,
            nhead=cfg.attn_num_heads,
            dim_feedforward=cfg.attn_ff_dim,
            dropout=cfg.attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.attn_num_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(x)
        x = self.encoder(x)
        return x


# ============================================================
# Transition model
# ============================================================

class RecurrentTransitionModelV2(nn.Module):
    """
    Rollout transition on top of latent + action + previous rollout hidden state.
    """
    def __init__(self, cfg: ModelV2Config) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_emb_dim, cfg.gru_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru_cell = nn.GRUCell(cfg.gru_hidden_dim, cfg.gru_hidden_dim)

    def forward_step(
        self,
        z_t: torch.Tensor,   # [B, Z]
        a_emb: torch.Tensor, # [B, A]
        h_t: torch.Tensor,   # [B, H]
    ) -> torch.Tensor:
        x = torch.cat([z_t, a_emb], dim=-1)
        x = self.input_proj(x)
        h_next = self.gru_cell(x, h_t)
        return h_next


# ============================================================
# Decoder
# ============================================================

class ObservationDecoderV2(nn.Module):
    """
    Stronger decoder than V1.
    Input:  [B, Z]
    Output: [B, 1, H, W]
    """
    def __init__(self, cfg: ModelV2Config) -> None:
        super().__init__()
        base = cfg.decoder_base_channels
        self.init_hw = (4, 4)
        self.fc = nn.Linear(cfg.latent_dim, base * self.init_hw[0] * self.init_hw[1])

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

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        x = self.fc(z)
        x = x.view(B, -1, self.init_hw[0], self.init_hw[1])
        x = self.block_4(x)
        x = self.up_8(x)
        x = self.up_16(x)
        x = self.up_32(x)
        x = self.up_64(x)
        x = self.out_conv(x)
        x = torch.sigmoid(x)
        return x


# ============================================================
# Pose / collision heads
# ============================================================

class PoseHeadsV2(nn.Module):
    def __init__(self, cfg: ModelV2Config) -> None:
        super().__init__()
        self.row_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.latent_dim, cfg.num_row_classes),
        )
        self.col_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.latent_dim, cfg.num_col_classes),
        )
        self.heading_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.latent_dim, cfg.num_heading_classes),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "row_logits": self.row_head(z),
            "col_logits": self.col_head(z),
            "heading_logits": self.heading_head(z),
        }


# ============================================================
# Main model
# ============================================================

class WorldModelV2(nn.Module):
    """
    V2 architecture:
      1. CNN encoder for each frame
      2. GRU filtering backbone over [obs_feat, prev_action_emb]
      3. Temporal self-attention over GRU states
      4. Latent projection from attended hidden sequence
      5. Decoder + pose heads + collision head
      6. Recurrent rollout transition from filtered latent state

    Main motivation over V1:
      - better temporal reasoning in ambiguous corridors
      - richer latent state
      - stronger reconstruction
      - keep collision-aware modeling
    """
    def __init__(self, cfg: ModelV2Config) -> None:
        super().__init__()
        self.cfg = cfg

        # embeddings / encoder
        self.action_embedding = nn.Embedding(cfg.num_actions, cfg.action_emb_dim)
        self.encoder = ObservationEncoderV2(cfg)

        # filtering GRU
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

        # temporal attention on GRU states
        self.temporal_attention = TemporalAttentionModule(cfg)

        # latent projection from attended states
        self.hidden_to_latent = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.gru_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.gru_hidden_dim, cfg.latent_dim),
        )

        # decoder / heads
        self.decoder = ObservationDecoderV2(cfg)
        self.pose_heads = PoseHeadsV2(cfg)

        # collision head: current state + action -> collision logits
        self.collision_head = nn.Sequential(
            nn.Linear(
                cfg.gru_hidden_dim + cfg.latent_dim + cfg.action_emb_dim,
                cfg.collision_head_hidden_dim,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.collision_head_hidden_dim, 2),
        )

        # rollout transition
        self.transition_model = RecurrentTransitionModelV2(cfg)

    # --------------------------------------------------------
    # Utility helpers
    # --------------------------------------------------------

    def encode_observation_sequence(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        obs_seq: [B, T, C, H, W]
        returns: [B, T, F]
        """
        B, T, C, H, W = obs_seq.shape
        x = obs_seq.view(B * T, C, H, W)
        feat = self.encoder(x)                 # [B*T, F]
        feat = feat.view(B, T, -1)            # [B, T, F]
        return feat

    def build_filter_inputs(
        self,
        feat_seq: torch.Tensor,      # [B, T, F]
        actions: torch.Tensor,       # [B, T-1]
    ) -> torch.Tensor:
        """
        Build GRU filtering inputs using current observation feat and previous action.
        For t=0, previous action is zero embedding.
        """
        B, T, Fdim = feat_seq.shape
        device = feat_seq.device

        zero_action = torch.zeros(B, 1, dtype=torch.long, device=device)
        prev_actions = torch.cat([zero_action, actions], dim=1)[:, :T]   # [B, T]
        prev_a_emb = self.action_embedding(prev_actions)                  # [B, T, A]

        x = torch.cat([feat_seq, prev_a_emb], dim=-1)                    # [B, T, F+A]
        x = self.filter_input_proj(x)                                     # [B, T, H]
        return x

    def predict_collision_logits(
        self,
        h: torch.Tensor,         # [B, H]
        z: torch.Tensor,         # [B, Z]
        actions: torch.Tensor,   # [B]
    ) -> torch.Tensor:
        a_emb = self.action_embedding(actions)  # [B, A]
        x = torch.cat([h, z, a_emb], dim=-1)
        logits = self.collision_head(x)
        return logits

    # --------------------------------------------------------
    # Filtering pass
    # --------------------------------------------------------

    def forward_filter(
        self,
        obs_seq: torch.Tensor,   # [B,T,C,H,W]
        actions: torch.Tensor,   # [B,T-1]
        h0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Filtering pass over real observations.

        Returns:
          feat_seq            [B,T,F]
          h_seq_gru           [B,T,H]
          h_seq               [B,T,H]   attended sequence states
          h_last              [1,B,H]   last GRU hidden
          z_seq               [B,T,Z]
          recon_seq           [B,T,C,H,W]
          row_logits_seq      [B,T,R]
          col_logits_seq      [B,T,Cc]
          heading_logits_seq  [B,T,Hd]
          collision_logits_seq[B,T-1,2]
        """
        B, T, C, H, W = obs_seq.shape
        if actions.shape[1] != T - 1:
            raise ValueError(
                f"Expected actions shape [B, T-1], got {actions.shape} for T={T}"
            )

        feat_seq = self.encode_observation_sequence(obs_seq)              # [B,T,F]
        filter_inputs = self.build_filter_inputs(feat_seq, actions)       # [B,T,H]

        h_seq_gru, h_last = self.filter_gru(filter_inputs, h0)            # [B,T,H], [1,B,H]
        h_seq_attn = self.temporal_attention(h_seq_gru)                   # [B,T,H]

        z_seq = self.hidden_to_latent(h_seq_attn)                         # [B,T,Z]

        # decode all filtered states
        z_flat = z_seq.reshape(B * T, -1)
        recon_flat = self.decoder(z_flat)                                 # [B*T,C,H,W]
        pose_flat = self.pose_heads(z_flat)

        recon_seq = recon_flat.view(B, T, C, H, W)
        row_logits_seq = pose_flat["row_logits"].view(B, T, -1)
        col_logits_seq = pose_flat["col_logits"].view(B, T, -1)
        heading_logits_seq = pose_flat["heading_logits"].view(B, T, -1)

        # collision prediction on real filtered states
        collision_logits_list = []
        h_for_collision = h_seq_attn[:, :-1, :]    # [B,T-1,H]
        z_for_collision = z_seq[:, :-1, :]         # [B,T-1,Z]
        for t in range(T - 1):
            logits_t = self.predict_collision_logits(
                h=h_for_collision[:, t, :],
                z=z_for_collision[:, t, :],
                actions=actions[:, t],
            )
            collision_logits_list.append(logits_t)
        collision_logits_seq = torch.stack(collision_logits_list, dim=1)  # [B,T-1,2]

        return {
            "feat_seq": feat_seq,
            "h_seq_gru": h_seq_gru,
            "h_seq": h_seq_attn,
            "h_last": h_last,
            "z_seq": z_seq,
            "recon_seq": recon_seq,
            "row_logits_seq": row_logits_seq,
            "col_logits_seq": col_logits_seq,
            "heading_logits_seq": heading_logits_seq,
            "collision_logits_seq": collision_logits_seq,
        }

    # --------------------------------------------------------
    # Rollout pass
    # --------------------------------------------------------

    def rollout_from_filtered_state(
        self,
        z_start: torch.Tensor,     # [B,Z]
        h_start: torch.Tensor,     # [B,H]
        action_seq: torch.Tensor,  # [B,K]
    ) -> Dict[str, torch.Tensor]:
        """
        Rollout future states from filtered start state.

        Returns:
          h_roll                 [B,K,H]
          z_roll                 [B,K,Z]
          recon_roll             [B,K,C,H,W]
          row_logits_roll        [B,K,R]
          col_logits_roll        [B,K,Cc]
          heading_logits_roll    [B,K,Hd]
          collision_logits_roll  [B,K,2]
        """
        B, K = action_seq.shape

        h_t = h_start
        z_t = z_start

        h_roll = []
        z_roll = []
        recon_roll = []
        row_roll = []
        col_roll = []
        heading_roll = []
        collision_logits_roll = []

        for k in range(K):
            a_t = action_seq[:, k]                              # [B]
            a_emb = self.action_embedding(a_t)                  # [B,A]

            collision_logits_t = self.predict_collision_logits(
                h=h_t,
                z=z_t,
                actions=a_t,
            )
            collision_logits_roll.append(collision_logits_t)

            h_t = self.transition_model.forward_step(z_t, a_emb, h_t)     # [B,H]
            z_t = self.hidden_to_latent(h_t)                              # [B,Z]

            recon_t = self.decoder(z_t)                                   # [B,C,H,W]
            pose_t = self.pose_heads(z_t)

            h_roll.append(h_t)
            z_roll.append(z_t)
            recon_roll.append(recon_t)
            row_roll.append(pose_t["row_logits"])
            col_roll.append(pose_t["col_logits"])
            heading_roll.append(pose_t["heading_logits"])

        h_roll = torch.stack(h_roll, dim=1)
        z_roll = torch.stack(z_roll, dim=1)
        recon_roll = torch.stack(recon_roll, dim=1)
        row_roll = torch.stack(row_roll, dim=1)
        col_roll = torch.stack(col_roll, dim=1)
        heading_roll = torch.stack(heading_roll, dim=1)
        collision_logits_roll = torch.stack(collision_logits_roll, dim=1)

        return {
            "h_roll": h_roll,
            "z_roll": z_roll,
            "recon_roll": recon_roll,
            "row_logits_roll": row_roll,
            "col_logits_roll": col_roll,
            "heading_logits_roll": heading_roll,
            "collision_logits_roll": collision_logits_roll,
        }


# ============================================================
# Quick shape test
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ModelV2Config(
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
        latent_dim=128,
        decoder_base_channels=128,
        collision_head_hidden_dim=128,
        max_seq_len=128,
    )

    model = WorldModelV2(cfg).to(device)
    model.eval()

    B, T = 8, 32
    K = 5
    obs = torch.randn(B, T, 1, 64, 64, device=device)
    actions = torch.randint(0, 4, (B, T - 1), device=device)

    with torch.no_grad():
        filt = model.forward_filter(obs, actions)

        print("Filter outputs:")
        print("  feat_seq             :", tuple(filt["feat_seq"].shape))
        print("  h_seq_gru            :", tuple(filt["h_seq_gru"].shape))
        print("  h_seq                :", tuple(filt["h_seq"].shape))
        print("  h_last               :", tuple(filt["h_last"].shape))
        print("  z_seq                :", tuple(filt["z_seq"].shape))
        print("  recon_seq            :", tuple(filt["recon_seq"].shape))
        print("  row_logits_seq       :", tuple(filt["row_logits_seq"].shape))
        print("  col_logits_seq       :", tuple(filt["col_logits_seq"].shape))
        print("  heading_logits_seq   :", tuple(filt["heading_logits_seq"].shape))
        print("  collision_logits_seq :", tuple(filt["collision_logits_seq"].shape))

        z_start = filt["z_seq"][:, 10, :]
        h_start = filt["h_seq"][:, 10, :]
        action_roll = actions[:, 10:10 + K]
        roll = model.rollout_from_filtered_state(z_start, h_start, action_roll)

        print("\nRollout outputs:")
        print("  h_roll               :", tuple(roll["h_roll"].shape))
        print("  z_roll               :", tuple(roll["z_roll"].shape))
        print("  recon_roll           :", tuple(roll["recon_roll"].shape))
        print("  row_logits_roll      :", tuple(roll["row_logits_roll"].shape))
        print("  col_logits_roll      :", tuple(roll["col_logits_roll"].shape))
        print("  heading_logits_roll  :", tuple(roll["heading_logits_roll"].shape))
        print("  collision_logits_roll:", tuple(roll["collision_logits_roll"].shape))

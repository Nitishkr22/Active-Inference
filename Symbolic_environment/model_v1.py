# model_v1.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Config
# ============================================================

@dataclass
class ModelV1Config:
    obs_channels: int = 1
    obs_height: int = 64
    obs_width: int = 64

    num_actions: int = 4
    action_emb_dim: int = 16

    encoder_feat_dim: int = 128
    gru_hidden_dim: int = 256
    latent_dim: int = 64

    num_row_classes: int = 9
    num_col_classes: int = 9
    num_heading_classes: int = 4

    decoder_base_channels: int = 128


# ============================================================
# CNN Encoder
# ============================================================

class CNNEncoder(nn.Module):
    """
    Image encoder:
      [B,1,64,64] -> [B, encoder_feat_dim]

    Design:
      conv -> conv -> conv -> conv -> flatten -> linear
    """

    def __init__(self, cfg: ModelV1Config):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(cfg.obs_channels, 32, kernel_size=4, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),                  # 32 -> 16
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),                 # 16 -> 8
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),                # 8 -> 4
            nn.ReLU(inplace=True),
        )

        conv_out_dim = 128 * 4 * 4

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, cfg.encoder_feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,64,64]
        returns: [B, encoder_feat_dim]
        """
        h = self.conv(x)
        feat = self.fc(h)
        return feat


# ============================================================
# Decoder
# ============================================================

class CNNDecoder(nn.Module):
    """
    Latent decoder:
      [B, latent_dim] -> [B,1,64,64]
    """

    def __init__(self, cfg: ModelV1Config):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.decoder_base_channels * 4 * 4),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(cfg.decoder_base_channels, 128, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),                         # 8 -> 16
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),                          # 16 -> 32
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, cfg.obs_channels, kernel_size=4, stride=2, padding=1),            # 32 -> 64
            nn.Sigmoid(),
        )

        self.base_channels = cfg.decoder_base_channels

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, latent_dim]
        returns: [B,1,64,64]
        """
        h = self.fc(z)                                   # [B, C*4*4]
        h = h.view(z.shape[0], self.base_channels, 4, 4)
        out = self.deconv(h)
        return out


# ============================================================
# Pose Heads
# ============================================================

class PoseHeads(nn.Module):
    """
    Predict row / col / heading logits from latent z.
    """

    def __init__(self, cfg: ModelV1Config):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(cfg.latent_dim, 128),
            nn.ReLU(inplace=True),
        )

        self.row_head = nn.Linear(128, cfg.num_row_classes)
        self.col_head = nn.Linear(128, cfg.num_col_classes)
        self.heading_head = nn.Linear(128, cfg.num_heading_classes)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        z: [B, latent_dim]
        returns:
          row_logits     : [B, num_row_classes]
          col_logits     : [B, num_col_classes]
          heading_logits : [B, num_heading_classes]
        """
        h = self.shared(z)
        return {
            "row_logits": self.row_head(h),
            "col_logits": self.col_head(h),
            "heading_logits": self.heading_head(h),
        }


# ============================================================
# Action embedding
# ============================================================

class ActionEmbedding(nn.Module):
    """
    Embed discrete actions to continuous vectors.
    """

    def __init__(self, cfg: ModelV1Config):
        super().__init__()
        self.emb = nn.Embedding(cfg.num_actions, cfg.action_emb_dim)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        a: [B] or [B,T]
        returns: [..., action_emb_dim]
        """
        return self.emb(a)


# ============================================================
# Recurrent state estimator
# ============================================================

class StateEstimatorGRU(nn.Module):
    """
    Filtering GRU:
      input_t = concat(encoder_feat_t, previous_action_embedding)
      hidden_t = GRU(input_t, hidden_{t-1})

    This produces the filtered hidden state from actual observations.
    """

    def __init__(self, cfg: ModelV1Config):
        super().__init__()
        input_dim = cfg.encoder_feat_dim + cfg.action_emb_dim
        self.gru = nn.GRU(input_dim, cfg.gru_hidden_dim, batch_first=True)

    def forward(
        self,
        seq_inputs: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq_inputs: [B,T,input_dim]
        h0: [1,B,H] or None

        returns:
          h_seq: [B,T,H]
          h_last: [1,B,H]
        """
        h_seq, h_last = self.gru(seq_inputs, h0)
        return h_seq, h_last


# ============================================================
# Transition model
# ============================================================

class RecurrentTransitionModel(nn.Module):
    """
    Recurrent latent transition model for imagination / rollout.

    At rollout time:
      input = concat(current_latent, action_embedding)
      next_hidden = GRUCell(input, current_hidden)

    This models future hidden-state evolution under actions.
    """

    def __init__(self, cfg: ModelV1Config):
        super().__init__()
        input_dim = cfg.latent_dim + cfg.action_emb_dim
        self.gru_cell = nn.GRUCell(input_dim, cfg.gru_hidden_dim)

    def forward_step(
        self,
        z_t: torch.Tensor,
        a_t_emb: torch.Tensor,
        h_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_t:    [B, latent_dim]
        a_t_emb:[B, action_emb_dim]
        h_t:    [B, gru_hidden_dim]

        returns:
          h_next: [B, gru_hidden_dim]
        """
        inp = torch.cat([z_t, a_t_emb], dim=-1)
        h_next = self.gru_cell(inp, h_t)
        return h_next


# ============================================================
# Full model
# ============================================================

class WorldModelV1(nn.Module):
    """
    Version 1 world model:
      - CNN encoder
      - action embedding
      - filtering GRU state estimator
      - hidden -> latent projection
      - decoder
      - pose heads
      - recurrent transition model

    Main outputs:
      - filtered hidden states h_t
      - filtered latents z_t
      - reconstructions from z_t
      - pose logits from z_t
      - rollout transitions using recurrent transition model
    """

    def __init__(self, cfg: ModelV1Config):
        super().__init__()
        self.cfg = cfg

        self.encoder = CNNEncoder(cfg)
        self.action_embedding = ActionEmbedding(cfg)
        self.state_estimator = StateEstimatorGRU(cfg)

        self.hidden_to_latent = nn.Sequential(
            nn.Linear(cfg.gru_hidden_dim, cfg.latent_dim),
            nn.ReLU(inplace=True),
        )

        self.decoder = CNNDecoder(cfg)
        self.pose_heads = PoseHeads(cfg)
        self.transition_model = RecurrentTransitionModel(cfg)

    # --------------------------------------------------------
    # Helper: encode observation sequence
    # --------------------------------------------------------
    def encode_observation_sequence(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        obs_seq: [B,T,1,H,W]
        returns:
          feat_seq: [B,T,encoder_feat_dim]
        """
        B, T, C, H, W = obs_seq.shape
        flat = obs_seq.view(B * T, C, H, W)
        feat_flat = self.encoder(flat)                                # [B*T,F]
        feat_seq = feat_flat.view(B, T, -1)
        return feat_seq

    # --------------------------------------------------------
    # Helper: build GRU inputs from obs features + previous actions
    # --------------------------------------------------------
    def build_filter_inputs(
        self,
        feat_seq: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        feat_seq: [B,T,F]
        actions:  [B,T-1]

        We use previous action at each time step:
          t=0 gets zero-action embedding
          t>=1 gets embedding of action[t-1]

        returns:
          gru_inputs: [B,T,F+A]
        """
        B, T, Fdim = feat_seq.shape
        device = feat_seq.device

        # zero previous action for t=0
        prev_actions = torch.zeros((B, T), dtype=torch.long, device=device)
        prev_actions[:, 1:] = actions

        a_emb = self.action_embedding(prev_actions)                   # [B,T,A]
        gru_inputs = torch.cat([feat_seq, a_emb], dim=-1)            # [B,T,F+A]
        return gru_inputs

    # --------------------------------------------------------
    # Filtering pass over actual observations
    # --------------------------------------------------------
    def forward_filter(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        observations: [B,T,1,H,W]
        actions:      [B,T-1]

        returns dictionary with:
          feat_seq        : [B,T,F]
          h_seq           : [B,T,H]
          h_last          : [1,B,H]
          z_seq           : [B,T,Z]
          recon_seq       : [B,T,1,H,W]
          row_logits_seq  : [B,T,R]
          col_logits_seq  : [B,T,C]
          heading_logits_seq : [B,T,Hd]
        """
        B, T, C, H, W = observations.shape

        feat_seq = self.encode_observation_sequence(observations)         # [B,T,F]
        gru_inputs = self.build_filter_inputs(feat_seq, actions)          # [B,T,F+A]

        h_seq, h_last = self.state_estimator(gru_inputs, h0=h0)           # [B,T,H], [1,B,H]

        z_seq = self.hidden_to_latent(h_seq)                              # [B,T,Z]

        # decode all time steps
        z_flat = z_seq.reshape(B * T, -1)
        recon_flat = self.decoder(z_flat)                                 # [B*T,1,H,W]
        recon_seq = recon_flat.view(B, T, C, H, W)

        pose_dict = self.pose_heads(z_flat)
        row_logits_seq = pose_dict["row_logits"].view(B, T, -1)
        col_logits_seq = pose_dict["col_logits"].view(B, T, -1)
        heading_logits_seq = pose_dict["heading_logits"].view(B, T, -1)

        return {
            "feat_seq": feat_seq,
            "h_seq": h_seq,
            "h_last": h_last,
            "z_seq": z_seq,
            "recon_seq": recon_seq,
            "row_logits_seq": row_logits_seq,
            "col_logits_seq": col_logits_seq,
            "heading_logits_seq": heading_logits_seq,
        }

    # --------------------------------------------------------
    # Rollout from a chosen start time
    # --------------------------------------------------------
    def rollout_from_filtered_state(
        self,
        z_start: torch.Tensor,
        h_start: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Rollout future hidden/latent states from filtered state.

        Inputs:
          z_start:    [B,Z]   filtered latent at start time
          h_start:    [B,H]   filtered hidden at start time
          action_seq: [B,K]   actions to rollout over K steps

        returns:
          h_roll: [B,K,H]
          z_roll: [B,K,Z]
          recon_roll: [B,K,1,H,W]
          row_logits_roll: [B,K,R]
          col_logits_roll: [B,K,C]
          heading_logits_roll: [B,K,Hd]
        """
        B, K = action_seq.shape
        device = action_seq.device

        h_t = h_start
        z_t = z_start

        h_roll = []
        z_roll = []
        row_roll = []
        col_roll = []
        heading_roll = []
        recon_roll = []

        for k in range(K):
            a_t = action_seq[:, k]                              # [B]
            a_emb = self.action_embedding(a_t)                  # [B,A]

            h_t = self.transition_model.forward_step(z_t, a_emb, h_t)   # [B,H]
            z_t = self.hidden_to_latent(h_t)                               # [B,Z]

            recon_t = self.decoder(z_t)                         # [B,1,H,W]
            pose_t = self.pose_heads(z_t)

            h_roll.append(h_t)
            z_roll.append(z_t)
            recon_roll.append(recon_t)
            row_roll.append(pose_t["row_logits"])
            col_roll.append(pose_t["col_logits"])
            heading_roll.append(pose_t["heading_logits"])

        h_roll = torch.stack(h_roll, dim=1)                     # [B,K,H]
        z_roll = torch.stack(z_roll, dim=1)                     # [B,K,Z]
        recon_roll = torch.stack(recon_roll, dim=1)             # [B,K,1,H,W]
        row_roll = torch.stack(row_roll, dim=1)                 # [B,K,R]
        col_roll = torch.stack(col_roll, dim=1)                 # [B,K,C]
        heading_roll = torch.stack(heading_roll, dim=1)         # [B,K,Hd]

        return {
            "h_roll": h_roll,
            "z_roll": z_roll,
            "recon_roll": recon_roll,
            "row_logits_roll": row_roll,
            "col_logits_roll": col_roll,
            "heading_logits_roll": heading_roll,
        }

    # --------------------------------------------------------
    # Full forward (filter only)
    # --------------------------------------------------------
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Default forward = filtering pass on actual observations.
        """
        return self.forward_filter(observations, actions, h0=h0)


# ============================================================
# Debug
# ============================================================

if __name__ == "__main__":
    cfg = ModelV1Config(
        obs_channels=1,
        obs_height=64,
        obs_width=64,
        num_actions=4,
        action_emb_dim=16,
        encoder_feat_dim=128,
        gru_hidden_dim=256,
        latent_dim=64,
        num_row_classes=9,
        num_col_classes=9,
        num_heading_classes=4,
        decoder_base_channels=128,
    )

    model = WorldModelV1(cfg)

    B = 8
    T = 32
    H = 64
    W = 64

    observations = torch.randn(B, T, 1, H, W)
    actions = torch.randint(0, cfg.num_actions, (B, T - 1))

    out = model.forward_filter(observations, actions)

    print("Filter outputs:")
    for k, v in out.items():
        print(f"  {k:20s}: {tuple(v.shape)}")

    # test rollout from time step 10 for horizon 5
    t0 = 10
    horizon = 5
    z_start = out["z_seq"][:, t0]                 # [B,Z]
    h_start = out["h_seq"][:, t0]                 # [B,H]
    action_seq = actions[:, t0:t0 + horizon]      # [B,K]

    roll = model.rollout_from_filtered_state(z_start, h_start, action_seq)

    print("\nRollout outputs:")
    for k, v in roll.items():
        print(f"  {k:20s}: {tuple(v.shape)}")
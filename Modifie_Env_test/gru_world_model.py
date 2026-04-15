## gru_world_model.py. ##

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# 8 -> 4
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 4 * 4, feature_dim)

    def forward(self, x):
        h = self.net(x)
        h = h.view(h.size(0), -1)
        e = torch.tanh(self.fc(h))
        return e


class Decoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 128 * 4 * 4)

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, h):
        x = self.fc(h)
        x = x.view(h.size(0), 128, 4, 4)
        return self.net(x)


class PoseHead(nn.Module):
    def __init__(self, hidden_dim=128, num_rows=10, num_cols=10, num_headings=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.row_head = nn.Linear(64, num_rows)
        self.col_head = nn.Linear(64, num_cols)
        self.heading_head = nn.Linear(64, num_headings)

    def forward(self, h):
        z = self.shared(h)
        return self.row_head(z), self.col_head(z), self.heading_head(z)


class GRUWorldModel(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=128, num_actions=4):
        super().__init__()
        self.encoder = CNNEncoder(feature_dim=feature_dim)
        self.action_embed = nn.Embedding(num_actions, 16)
        self.gru = nn.GRUCell(feature_dim + 16, hidden_dim)
        self.decoder = Decoder(hidden_dim=hidden_dim)
        self.pose_head = PoseHead(hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward_sequence(self, obs_seq, action_seq):
        """
        obs_seq: [B, T, 1, 64, 64]
        action_seq: [B, T-1]
        """
        B, T = obs_seq.shape[:2]
        device = obs_seq.device

        h = self.init_hidden(B, device)

        hidden_states = []
        reconstructions = []
        row_logits_all = []
        col_logits_all = []
        heading_logits_all = []

        for t in range(T):
            obs_t = obs_seq[:, t]               # [B,1,64,64]
            e_t = self.encoder(obs_t)           # [B,F]

            if t == 0:
                a_embed = torch.zeros(B, 16, device=device)
            else:
                a_prev = action_seq[:, t - 1]   # previous action
                a_embed = self.action_embed(a_prev)

            gru_input = torch.cat([e_t, a_embed], dim=-1)
            h = self.gru(gru_input, h)

            recon_t = self.decoder(h)
            row_logits, col_logits, heading_logits = self.pose_head(h)

            hidden_states.append(h)
            reconstructions.append(recon_t)
            row_logits_all.append(row_logits)
            col_logits_all.append(col_logits)
            heading_logits_all.append(heading_logits)

        hidden_states = torch.stack(hidden_states, dim=1)         # [B,T,H]
        reconstructions = torch.stack(reconstructions, dim=1)     # [B,T,1,64,64]
        row_logits_all = torch.stack(row_logits_all, dim=1)       # [B,T,10]
        col_logits_all = torch.stack(col_logits_all, dim=1)       # [B,T,10]
        heading_logits_all = torch.stack(heading_logits_all, dim=1) # [B,T,4]

        return {
            "hidden_states": hidden_states,
            "reconstructions": reconstructions,
            "row_logits": row_logits_all,
            "col_logits": col_logits_all,
            "heading_logits": heading_logits_all,
        }
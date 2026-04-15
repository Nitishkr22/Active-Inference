# train_bottleneck_latent_transition.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sequence_dataset import SequenceDataset
from gru_world_model import GRUWorldModel


# ==========================================================
# Bottleneck latent modules
# ==========================================================

class HiddenToLatentEncoder(nn.Module):
    """
    Encode frozen GRU hidden state h into smaller latent z.
    """
    def __init__(self, hidden_dim=128, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, h):
        """
        h : [B, H]
        z : [B, Z]
        """
        return self.net(h)


class LatentTransitionModel(nn.Module):
    """
    Transition in bottleneck latent space:
        z_next = f(z, action)
    """
    def __init__(self, latent_dim=32, num_actions=4, action_emb_dim=16):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, action_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, z, action):
        """
        z      : [B, Z]
        action : [B]
        """
        a_emb = self.action_embed(action)          # [B, action_emb_dim]
        x = torch.cat([z, a_emb], dim=-1)          # [B, Z + action_emb_dim]
        z_next =  self.net(x)                       # [B, Z]
        return z_next


class LatentPoseHead(nn.Module):
    """
    Decode pose directly from bottleneck latent z.
    """
    def __init__(self, latent_dim=32, num_rows=10, num_cols=10, num_headings=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
        )
        self.row_head = nn.Linear(128, num_rows)
        self.col_head = nn.Linear(128, num_cols)
        self.heading_head = nn.Linear(128, num_headings)

    def forward(self, z):
        """
        z : [B, Z]
        returns:
            row_logits     : [B, num_rows]
            col_logits     : [B, num_cols]
            heading_logits : [B, num_headings]
        """
        x = self.shared(z)
        row_logits = self.row_head(x)
        col_logits = self.col_head(x)
        heading_logits = self.heading_head(x)
        return row_logits, col_logits, heading_logits


# ==========================================================
# Loss and evaluation
# ==========================================================

def rollout_multistep_bottleneck_loss(
    h_seq,
    actions,
    pos,
    heading,
    encoder,
    transition_model,
    pose_head,
    latent_loss_fn,
    ce_loss_fn,
    rollout_len=3,
    lambda_latent=1.0,
    lambda_pose=0.3,
    lambda_row=1.0,
    lambda_col=1.0,
    lambda_heading=5.0,
):
    """
    Multi-step rollout loss in bottleneck latent space.

    h_seq    : [B, T, H]    frozen GRU hidden states
    actions  : [B, T-1]
    pos      : [B, T, 2]
    heading  : [B, T]
    """
    B, T, H = h_seq.shape

    total_loss = 0.0
    total_terms = 0

    max_start = T - rollout_len - 1
    if max_start < 0:
        return torch.tensor(0.0, device=h_seq.device)

    for start_t in range(max_start + 1):
        # Start from true hidden state, encoded into z
        z_pred = encoder(h_seq[:, start_t, :])   # [B, Z]

        for k in range(rollout_len):
            a_t = actions[:, start_t + k]                        # [B]

            # True target latent from next true hidden state
            z_target = encoder(h_seq[:, start_t + k + 1, :])     # [B, Z]

            # Roll one step in z-space
            z_pred = transition_model(z_pred, a_t)               # [B, Z]

            # 1) latent consistency in z-space
            loss_latent = latent_loss_fn(z_pred, z_target)

            # 2) pose consistency from z-space
            row_logits, col_logits, heading_logits = pose_head(z_pred)

            row_target = pos[:, start_t + k + 1, 0]       # [B]
            col_target = pos[:, start_t + k + 1, 1]       # [B]
            heading_target = heading[:, start_t + k + 1]  # [B]

            loss_pose = (
                lambda_row * ce_loss_fn(row_logits, row_target)
                + lambda_col * ce_loss_fn(col_logits, col_target)
                + lambda_heading * ce_loss_fn(heading_logits, heading_target)
            )

            loss = lambda_latent * loss_latent + lambda_pose * loss_pose

            total_loss = total_loss + loss
            total_terms += 1

    return total_loss / max(total_terms, 1)


def evaluate_bottleneck_rollout_accuracy(
    h_seq,
    actions,
    pos,
    heading,
    encoder,
    transition_model,
    pose_head,
    rollout_len=3,
):
    """
    Free-rollout pose accuracy in bottleneck latent space.

    Returns:
        row_correct, col_correct, heading_correct, total
    """
    _, T, _ = h_seq.shape

    row_correct = 0
    col_correct = 0
    heading_correct = 0
    total = 0

    max_start = T - rollout_len - 1
    if max_start < 0:
        return 0, 0, 0, 0

    with torch.no_grad():
        for start_t in range(max_start + 1):
            z_pred = encoder(h_seq[:, start_t, :])   # [B, Z]

            for k in range(rollout_len):
                a_t = actions[:, start_t + k]
                z_pred = transition_model(z_pred, a_t)

                row_logits, col_logits, heading_logits = pose_head(z_pred)

                row_pred = row_logits.argmax(dim=1)
                col_pred = col_logits.argmax(dim=1)
                heading_pred = heading_logits.argmax(dim=1)

                row_target = pos[:, start_t + k + 1, 0]
                col_target = pos[:, start_t + k + 1, 1]
                heading_target = heading[:, start_t + k + 1]

                row_correct += (row_pred == row_target).sum().item()
                col_correct += (col_pred == col_target).sum().item()
                heading_correct += (heading_pred == heading_target).sum().item()
                total += row_target.size(0)

    return row_correct, col_correct, heading_correct, total


# ==========================================================
# Main training
# ==========================================================

def train_bottleneck_latent_transition(
    dataset_path="../../dataset/tiny_nav_sequence_dataset.npz",
    gru_ckpt_path="checkpoints/best_gru_world_model.pt",
    batch_size=128,
    lr=3e-4,
    epochs=25,
    rollout_len=3,
    latent_dim=32,
    lambda_latent=1.0,
    lambda_pose=0.3,
    lambda_row=1.0,
    lambda_col=1.0,
    lambda_heading=5.0,
    save_dir="checkpoints",
    seed=42,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    dataset = SequenceDataset(dataset_path)

    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Total sequences: {n_total}")
    print(f"Train sequences: {n_train}")
    print(f"Val sequences  : {n_val}")

    # --------------------------------------------------
    # Load frozen GRU world model
    # --------------------------------------------------
    wm = GRUWorldModel().to(device)
    ckpt = torch.load(gru_ckpt_path, map_location=device)
    wm.load_state_dict(ckpt["model_state_dict"])
    wm.eval()

    for p in wm.parameters():
        p.requires_grad = False

    print(f"Loaded frozen GRU world model from: {gru_ckpt_path}")
    print(f"Frozen GRU hidden_dim: {wm.hidden_dim}")

    # --------------------------------------------------
    # Trainable bottleneck-latent modules
    # --------------------------------------------------
    encoder = HiddenToLatentEncoder(
        hidden_dim=wm.hidden_dim,
        latent_dim=latent_dim,
    ).to(device)

    transition_model = LatentTransitionModel(
        latent_dim=latent_dim,
        num_actions=4,
        action_emb_dim=16,
    ).to(device)

    pose_head = LatentPoseHead(
        latent_dim=latent_dim,
        num_rows=10,
        num_cols=10,
        num_headings=4,
    ).to(device)

    params = (
        list(encoder.parameters())
        + list(transition_model.parameters())
        + list(pose_head.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=lr)

    latent_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(1, epochs + 1):
        encoder.train()
        transition_model.train()
        pose_head.train()

        train_loss = 0.0

        for batch in train_loader:
            obs = batch["obs"].to(device)           # [B, T, 1, 64, 64]
            actions = batch["actions"].to(device)   # [B, T-1]
            pos = batch["pos"].to(device)           # [B, T, 2]
            heading = batch["heading"].to(device)   # [B, T]

            # Get frozen GRU hidden states
            with torch.no_grad():
                out = wm.forward_sequence(obs, actions)
                h_seq = out["hidden_states"]        # [B, T, H]

            loss = rollout_multistep_bottleneck_loss(
                h_seq=h_seq,
                actions=actions,
                pos=pos,
                heading=heading,
                encoder=encoder,
                transition_model=transition_model,
                pose_head=pose_head,
                latent_loss_fn=latent_loss_fn,
                ce_loss_fn=ce_loss_fn,
                rollout_len=rollout_len,
                lambda_latent=lambda_latent,
                lambda_pose=lambda_pose,
                lambda_row=lambda_row,
                lambda_col=lambda_col,
                lambda_heading=lambda_heading,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * obs.size(0)

        train_loss /= len(train_loader.dataset)

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        encoder.eval()
        transition_model.eval()
        pose_head.eval()

        val_loss = 0.0
        total_row_correct = 0
        total_col_correct = 0
        total_heading_correct = 0
        total_count = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                actions = batch["actions"].to(device)
                pos = batch["pos"].to(device)
                heading = batch["heading"].to(device)

                out = wm.forward_sequence(obs, actions)
                h_seq = out["hidden_states"]

                loss = rollout_multistep_bottleneck_loss(
                    h_seq=h_seq,
                    actions=actions,
                    pos=pos,
                    heading=heading,
                    encoder=encoder,
                    transition_model=transition_model,
                    pose_head=pose_head,
                    latent_loss_fn=latent_loss_fn,
                    ce_loss_fn=ce_loss_fn,
                    rollout_len=rollout_len,
                    lambda_latent=lambda_latent,
                    lambda_pose=lambda_pose,
                    lambda_row=lambda_row,
                    lambda_col=lambda_col,
                    lambda_heading=lambda_heading,
                )

                row_correct, col_correct, heading_correct, count = evaluate_bottleneck_rollout_accuracy(
                    h_seq=h_seq,
                    actions=actions,
                    pos=pos,
                    heading=heading,
                    encoder=encoder,
                    transition_model=transition_model,
                    pose_head=pose_head,
                    rollout_len=rollout_len,
                )

                val_loss += loss.item() * obs.size(0)
                total_row_correct += row_correct
                total_col_correct += col_correct
                total_heading_correct += heading_correct
                total_count += count

        val_loss /= len(val_loader.dataset)

        rollout_row_acc = total_row_correct / max(total_count, 1)
        rollout_col_acc = total_col_correct / max(total_count, 1)
        rollout_heading_acc = total_heading_correct / max(total_count, 1)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"rollout_row_acc={rollout_row_acc:.3f} | "
            f"rollout_col_acc={rollout_col_acc:.3f} | "
            f"rollout_heading_acc={rollout_heading_acc:.3f}"
        )

        if epoch >= 3 and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, "best_bottleneck_latent_transition.pt")

            torch.save(
                {
                    "encoder_state_dict": encoder.state_dict(),
                    "transition_model_state_dict": transition_model.state_dict(),
                    "pose_head_state_dict": pose_head.state_dict(),
                    "hidden_dim": wm.hidden_dim,
                    "latent_dim": latent_dim,
                    "rollout_len": rollout_len,
                    "lambda_latent": lambda_latent,
                    "lambda_pose": lambda_pose,
                    "lambda_row": lambda_row,
                    "lambda_col": lambda_col,
                    "lambda_heading": lambda_heading,
                    "model_type": "bottleneck_latent_transition",
                },
                save_path,
            )
    print(f"  Saved best checkpoint to: {save_path}")

    print("Bottleneck latent transition training finished.")
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    train_bottleneck_latent_transition(
        dataset_path="../../dataset/tiny_nav_sequence_dataset.npz",
        gru_ckpt_path="checkpoints/best_gru_world_model.pt",
        batch_size=128,
        lr=3e-4,
        epochs=25,
        rollout_len=3,
        latent_dim=64,          # try 32 first
        lambda_latent=0.5,
        lambda_pose=1.0,
        lambda_row=1.0,
        lambda_col=1.0,
        lambda_heading=5.0,
        save_dir="checkpoints",
        seed=42,
    )
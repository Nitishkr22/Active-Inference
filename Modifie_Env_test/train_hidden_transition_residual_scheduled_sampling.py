## train_hidden_transition_residual_scheduled_sampling.py ##
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sequence_dataset import SequenceDataset
from gru_world_model import GRUWorldModel


class ResidualHiddenTransitionModelMultiStepPose(nn.Module):
    """
    Residual transition model:
        h_next = h + delta(h, action)
    """
    def __init__(self, hidden_dim=128, num_actions=4, action_emb_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.action_emb_dim = action_emb_dim

        self.action_embed = nn.Embedding(num_actions, action_emb_dim)
        self.delta_net = nn.Sequential(
            nn.Linear(hidden_dim + action_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

    def forward(self, h, action):
        """
        h      : [B, H]
        action : [B]
        """
        a_emb = self.action_embed(action)          # [B, action_emb_dim]
        x = torch.cat([h, a_emb], dim=-1)          # [B, H + action_emb_dim]
        delta = self.delta_net(x)                  # [B, H]
        h_next = h + delta                         # [B, H]
        return h_next


def get_teacher_forcing_ratio(
    epoch,
    total_epochs,
    start_ratio=1.0,
    end_ratio=0.0,
    schedule="linear",
):
    """
    Returns probability of using the true hidden state as rollout input.

    start_ratio = probability at epoch 1
    end_ratio   = probability at final epoch
    """
    if total_epochs <= 1:
        return end_ratio

    progress = (epoch - 1) / (total_epochs - 1)

    if schedule == "linear":
        ratio = start_ratio + (end_ratio - start_ratio) * progress
    elif schedule == "cosine":
        # smooth decay from start_ratio to end_ratio
        import math
        alpha = 0.5 * (1.0 + math.cos(math.pi * progress))
        ratio = end_ratio + (start_ratio - end_ratio) * alpha
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    return float(max(min(ratio, 1.0), 0.0))


def rollout_multistep_pose_loss_scheduled_sampling(
    h_seq,
    actions,
    pos,
    heading,
    transition_model,
    wm,
    latent_loss_fn,
    ce_loss_fn,
    rollout_len=3,
    lambda_latent=1.0,
    lambda_pose=0.3,
    lambda_row=1.0,
    lambda_col=1.0,
    lambda_heading=5.0,
    teacher_forcing_ratio=1.0,
):
    """
    Multi-step rollout loss in hidden space + pose supervision
    with scheduled sampling.

    h_seq    : [B, T, H]
    actions  : [B, T-1]
    pos      : [B, T, 2]
    heading  : [B, T]

    teacher_forcing_ratio:
        probability of using true hidden state at each rollout step input.
    """
    B, T, H = h_seq.shape

    total_loss = 0.0
    total_terms = 0

    max_start = T - rollout_len - 1
    if max_start < 0:
        return torch.tensor(0.0, device=h_seq.device)

    for start_t in range(max_start + 1):
        # Initialize predicted hidden state from true hidden state at rollout start
        h_pred = h_seq[:, start_t, :]   # [B, H]

        for k in range(rollout_len):
            a_t = actions[:, start_t + k]                  # [B]
            h_true_in = h_seq[:, start_t + k, :]           # [B, H]
            h_target = h_seq[:, start_t + k + 1, :]       # [B, H]

            # Scheduled sampling:
            # choose whether to feed true hidden state or previous predicted hidden state
            if k == 0:
                # first step always starts from the true hidden state
                h_in = h_true_in
            else:
                use_teacher = random.random() < teacher_forcing_ratio
                h_in = h_true_in if use_teacher else h_pred

            # Predict next hidden state
            h_pred = transition_model(h_in, a_t)

            # 1) latent consistency
            loss_latent = latent_loss_fn(h_pred, h_target)

            # 2) pose consistency through frozen GRU pose head
            row_logits, col_logits, heading_logits = wm.pose_head(h_pred)

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


def evaluate_pose_rollout_accuracy(
    h_seq,
    actions,
    pos,
    heading,
    transition_model,
    wm,
    rollout_len=3,
):
    """
    Free-rollout pose accuracy over all valid rollout steps.

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
            h_pred = h_seq[:, start_t, :]

            for k in range(rollout_len):
                a_t = actions[:, start_t + k]
                h_pred = transition_model(h_pred, a_t)

                row_logits, col_logits, heading_logits = wm.pose_head(h_pred)

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


def train_hidden_transition_residual_scheduled_sampling(
    dataset_path="../../dataset/tiny_nav_sequence_dataset.npz",
    gru_ckpt_path="checkpoints/best_gru_world_model.pt",
    batch_size=128,
    lr=3e-4,
    epochs=25,
    rollout_len=3,
    lambda_latent=1.0,
    lambda_pose=0.3,
    lambda_row=1.0,
    lambda_col=1.0,
    lambda_heading=5.0,
    teacher_forcing_start=1.0,
    teacher_forcing_end=0.0,
    teacher_forcing_schedule="linear",
    save_dir="checkpoints",
    seed=42,
):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # Dataset
    # -----------------------------
    dataset = SequenceDataset(dataset_path)

    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Total sequences: {n_total}")
    print(f"Train sequences: {n_train}")
    print(f"Val sequences  : {n_val}")

    # -----------------------------
    # Load frozen GRU world model
    # -----------------------------
    wm = GRUWorldModel().to(device)
    ckpt = torch.load(gru_ckpt_path, map_location=device)
    wm.load_state_dict(ckpt["model_state_dict"])
    wm.eval()

    for p in wm.parameters():
        p.requires_grad = False

    print(f"Loaded frozen GRU world model from: {gru_ckpt_path}")
    print(f"Frozen GRU hidden_dim: {wm.hidden_dim}")

    # -----------------------------
    # Transition model to train
    # -----------------------------
    transition_model = ResidualHiddenTransitionModelMultiStepPose(
        hidden_dim=wm.hidden_dim,
        num_actions=4,
        action_emb_dim=16,
    ).to(device)

    optimizer = torch.optim.Adam(transition_model.parameters(), lr=lr)

    latent_loss_fn = nn.MSELoss()
    ce_loss_fn = nn.CrossEntropyLoss()

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in range(1, epochs + 1):
        transition_model.train()
        train_loss = 0.0

        teacher_forcing_ratio = get_teacher_forcing_ratio(
            epoch=epoch,
            total_epochs=epochs,
            start_ratio=teacher_forcing_start,
            end_ratio=teacher_forcing_end,
            schedule=teacher_forcing_schedule,
        )

        for batch in train_loader:
            obs = batch["obs"].to(device)           # [B, T, 1, 64, 64]
            actions = batch["actions"].to(device)   # [B, T-1]
            pos = batch["pos"].to(device)           # [B, T, 2]
            heading = batch["heading"].to(device)   # [B, T]

            # Get frozen GRU hidden states
            with torch.no_grad():
                out = wm.forward_sequence(obs, actions)
                h_seq = out["hidden_states"]        # [B, T, H]

            loss = rollout_multistep_pose_loss_scheduled_sampling(
                h_seq=h_seq,
                actions=actions,
                pos=pos,
                heading=heading,
                transition_model=transition_model,
                wm=wm,
                latent_loss_fn=latent_loss_fn,
                ce_loss_fn=ce_loss_fn,
                rollout_len=rollout_len,
                lambda_latent=lambda_latent,
                lambda_pose=lambda_pose,
                lambda_row=lambda_row,
                lambda_col=lambda_col,
                lambda_heading=lambda_heading,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * obs.size(0)

        train_loss /= len(train_loader.dataset)

        # -----------------------------
        # Validation
        # -----------------------------
        transition_model.eval()
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

                # Validation loss:
                # use pure free-rollout behavior to match test-time usage
                loss = rollout_multistep_pose_loss_scheduled_sampling(
                    h_seq=h_seq,
                    actions=actions,
                    pos=pos,
                    heading=heading,
                    transition_model=transition_model,
                    wm=wm,
                    latent_loss_fn=latent_loss_fn,
                    ce_loss_fn=ce_loss_fn,
                    rollout_len=rollout_len,
                    lambda_latent=lambda_latent,
                    lambda_pose=lambda_pose,
                    lambda_row=lambda_row,
                    lambda_col=lambda_col,
                    lambda_heading=lambda_heading,
                    teacher_forcing_ratio=0.0,
                )

                row_correct, col_correct, heading_correct, count = evaluate_pose_rollout_accuracy(
                    h_seq=h_seq,
                    actions=actions,
                    pos=pos,
                    heading=heading,
                    transition_model=transition_model,
                    wm=wm,
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
            f"tf_ratio={teacher_forcing_ratio:.3f} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"rollout_row_acc={rollout_row_acc:.3f} | "
            f"rollout_col_acc={rollout_col_acc:.3f} | "
            f"rollout_heading_acc={rollout_heading_acc:.3f}"
        )

        if epoch >= 3 and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(
                save_dir,
                "best_hidden_transition_residual_scheduled_sampling.pt"
            )

            torch.save(
                {
                    "model_state_dict": transition_model.state_dict(),
                    "hidden_dim": wm.hidden_dim,
                    "rollout_len": rollout_len,
                    "lambda_latent": lambda_latent,
                    "lambda_pose": lambda_pose,
                    "lambda_row": lambda_row,
                    "lambda_col": lambda_col,
                    "lambda_heading": lambda_heading,
                    "model_type": "residual_mlp_transition_scheduled_sampling",
                    "teacher_forcing_start": teacher_forcing_start,
                    "teacher_forcing_end": teacher_forcing_end,
                    "teacher_forcing_schedule": teacher_forcing_schedule,
                },
                save_path,
            )
    print(f"  Saved best checkpoint to: {save_path}")

    print("Residual scheduled-sampling transition training finished.")
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    train_hidden_transition_residual_scheduled_sampling(
        dataset_path="../../dataset/tiny_nav_sequence_dataset.npz",
        gru_ckpt_path="checkpoints/best_gru_world_model.pt",
        batch_size=128,
        lr=3e-4,
        epochs=25,
        rollout_len=3,
        lambda_latent=1.0,
        lambda_pose=0.3,
        lambda_row=1.0,
        lambda_col=1.0,
        lambda_heading=5.0,
        teacher_forcing_start=1.0,
        teacher_forcing_end=0.0,
        teacher_forcing_schedule="linear",   # "linear" or "cosine"
        save_dir="checkpoints",
        seed=42,
    )
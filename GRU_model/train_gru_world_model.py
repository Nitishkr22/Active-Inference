import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from sequence_dataset import SequenceDataset
from gru_world_model import GRUWorldModel


def train_gru_world_model(
    dataset_path="tiny_nav_sequence_dataset.npz",
    batch_size=32,
    lr=3e-4,
    epochs=30,
    save_dir="checkpoints",
    lambda_recon=1.0,
    lambda_pose=0.3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = SequenceDataset(dataset_path)

    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = GRUWorldModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    recon_loss_fn = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0

        for batch in train_loader:
            obs = batch["obs"].to(device)              # [B,T,1,64,64]
            actions = batch["actions"].to(device)      # [B,T-1]
            pos = batch["pos"].to(device)              # [B,T,2]
            heading = batch["heading"].to(device)      # [B,T]

            out = model.forward_sequence(obs, actions)

            recon = out["reconstructions"]             # [B,T,1,64,64]

            loss_recon = recon_loss_fn(recon, obs)  # calculate losses for each sequence element and average

            B, T = heading.shape
            row_logits = out["row_logits"].reshape(B * T, -1)
            col_logits = out["col_logits"].reshape(B * T, -1)
            heading_logits = out["heading_logits"].reshape(B * T, -1)

            row_targets = pos[:, :, 0].reshape(B * T)
            col_targets = pos[:, :, 1].reshape(B * T)
            heading_targets = heading.reshape(B * T)

            loss_pose = (
                ce_loss(row_logits, row_targets) +
                ce_loss(col_logits, col_targets) +
                ce_loss(heading_logits, heading_targets)
            )

            loss = lambda_recon * loss_recon + lambda_pose * loss_pose

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += loss.item() * obs.size(0)

        train_total /= len(train_loader.dataset)

        model.eval()
        val_total = 0.0
        val_recon = 0.0

        row_correct = 0
        col_correct = 0
        heading_correct = 0
        count = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                actions = batch["actions"].to(device)
                pos = batch["pos"].to(device)
                heading = batch["heading"].to(device)

                out = model.forward_sequence(obs, actions)
                recon = out["reconstructions"]

                loss_recon = recon_loss_fn(recon, obs)

                B, T = heading.shape
                row_logits = out["row_logits"].reshape(B * T, -1)
                col_logits = out["col_logits"].reshape(B * T, -1)
                heading_logits = out["heading_logits"].reshape(B * T, -1)

                row_targets = pos[:, :, 0].reshape(B * T)
                col_targets = pos[:, :, 1].reshape(B * T)
                heading_targets = heading.reshape(B * T)

                loss_pose = (
                    ce_loss(row_logits, row_targets) +
                    ce_loss(col_logits, col_targets) +
                    ce_loss(heading_logits, heading_targets)
                )

                loss = lambda_recon * loss_recon + lambda_pose * loss_pose

                val_total += loss.item() * obs.size(0)
                val_recon += loss_recon.item() * obs.size(0)

                row_pred = row_logits.argmax(dim=1)
                col_pred = col_logits.argmax(dim=1)
                heading_pred = heading_logits.argmax(dim=1)

                row_correct += (row_pred == row_targets).sum().item()
                col_correct += (col_pred == col_targets).sum().item()
                heading_correct += (heading_pred == heading_targets).sum().item()
                count += B * T

        val_total /= len(val_loader.dataset)
        val_recon /= len(val_loader.dataset)

        row_acc = row_correct / count
        col_acc = col_correct / count
        heading_acc = heading_correct / count

        print(
            f"Epoch {epoch:02d} | "
            f"train_total={train_total:.6f} | "
            f"val_total={val_total:.6f} | "
            f"val_recon={val_recon:.6f} | "
            f"row_acc={row_acc:.3f} | "
            f"col_acc={col_acc:.3f} | "
            f"heading_acc={heading_acc:.3f}"
        )

        if epoch >= 5 and val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(save_dir, "best_gru_world_model.pt"),
            )

    print("Training finished.")
    print("Best val total loss:", best_val_loss)


if __name__ == "__main__":
    train_gru_world_model()
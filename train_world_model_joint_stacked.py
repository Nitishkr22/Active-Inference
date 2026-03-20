import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.world_model_sequence_dataset import SequenceTransitionDataset
from models.models import WorldModel


def train_world_model_joint_stacked(
    dataset_path="tiny_nav_exhaustive_dataset.npz",
    latent_dim=32,
    batch_size=64,
    lr=1e-3,
    epochs=30,
    save_dir="checkpoints",
    lambda_recon=1.0,
    lambda_next_recon=1.0,
    lambda_latent=0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = SequenceTransitionDataset(dataset_path)

    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2-channel input now
    model = WorldModel(latent_dim=latent_dim, num_actions=4, in_channels=2).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recon_loss_fn = nn.MSELoss()
    latent_loss_fn = nn.MSELoss()

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0

        for batch in train_loader:
            obs_stack = batch["obs_stack"].to(device)   # [B, 2, 64, 64]
            next_obs = batch["next_obs"].to(device)     # [B, 1, 64, 64]
            actions = batch["action"].to(device)

            # Build next stacked observation:
            # current obs becomes previous, next_obs becomes current
            curr_obs = batch["obs"].to(device)          # [B, 1, 64, 64]
            next_stack = torch.cat([curr_obs, next_obs], dim=1)  # [B, 2, 64, 64]

            out = model(obs_stack, actions, next_obs=next_stack)

            # Reconstruct only the current frame from obs_stack latent
            # decoder outputs 1-channel image
            loss_recon = recon_loss_fn(out["obs_recon"], batch["obs"].to(device))
            loss_next_recon = recon_loss_fn(out["next_obs_pred"], next_obs)
            loss_latent = latent_loss_fn(out["z_next_pred"], out["z_next_true"].detach())

            loss = (
                lambda_recon * loss_recon
                + lambda_next_recon * loss_next_recon
                + lambda_latent * loss_latent
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total += loss.item() * obs_stack.size(0)

        train_total /= len(train_loader.dataset)

        model.eval()
        val_total = 0.0
        val_recon = 0.0
        val_next_recon = 0.0
        val_latent = 0.0

        with torch.no_grad():
            for batch in val_loader:
                obs_stack = batch["obs_stack"].to(device)
                next_obs = batch["next_obs"].to(device)
                actions = batch["action"].to(device)
                curr_obs = batch["obs"].to(device)
                next_stack = torch.cat([curr_obs, next_obs], dim=1)

                out = model(obs_stack, actions, next_obs=next_stack)

                loss_recon = recon_loss_fn(out["obs_recon"], curr_obs)
                loss_next_recon = recon_loss_fn(out["next_obs_pred"], next_obs)
                loss_latent = latent_loss_fn(out["z_next_pred"], out["z_next_true"])

                loss = (
                    lambda_recon * loss_recon
                    + lambda_next_recon * loss_next_recon
                    + lambda_latent * loss_latent
                )

                val_total += loss.item() * obs_stack.size(0)
                val_recon += loss_recon.item() * obs_stack.size(0)
                val_next_recon += loss_next_recon.item() * obs_stack.size(0)
                val_latent += loss_latent.item() * obs_stack.size(0)

        val_total /= len(val_loader.dataset)
        val_recon /= len(val_loader.dataset)
        val_next_recon /= len(val_loader.dataset)
        val_latent /= len(val_loader.dataset)

        print(
            f"Epoch {epoch:02d} | "
            f"train_total={train_total:.6f} | "
            f"val_total={val_total:.6f} | "
            f"val_recon={val_recon:.6f} | "
            f"val_next_recon={val_next_recon:.6f} | "
            f"val_latent={val_latent:.6f}"
        )

        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_dim": latent_dim,
                },
                os.path.join(save_dir, "best_world_model_stacked.pt"),
            )

    print("Stacked joint training finished.")
    print("Best val total loss:", best_val_loss)


if __name__ == "__main__":
    train_world_model_joint_stacked()
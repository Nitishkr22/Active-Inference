import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.world_model_dataset import TransitionDataset
from models.models import AutoEncoder


def train_autoencoder(
    dataset_path="tiny_nav_dataset.npz",
    latent_dim=32,
    batch_size=64,
    lr=1e-3,
    epochs=20,
    save_dir="checkpoints",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = TransitionDataset(dataset_path)

    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = AutoEncoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            obs = batch["obs"].to(device)  # [B, 1, 64, 64]

            recon, z = model(obs)
            loss = criterion(recon, obs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * obs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                recon, z = model(obs)
                loss = criterion(recon, obs)
                val_loss += loss.item() * obs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_dim": latent_dim,
                },
                os.path.join(save_dir, "best_autoencoder.pt"),
            )

    print("Training finished.")
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    train_autoencoder()
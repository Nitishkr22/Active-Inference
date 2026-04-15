import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.world_model_dataset import TransitionDataset
from models.models import Encoder, Decoder, DynamicsModel


def train_dynamics(
    dataset_path="tiny_nav_dataset.npz",
    ae_ckpt_path="checkpoints/best_autoencoder.pt",
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

    # Load pretrained encoder-decoder
    encoder = Encoder(latent_dim=latent_dim).to(device)  # creating objects of encoder and decoder classes defined in models.py
    decoder = Decoder(latent_dim=latent_dim).to(device)

    # loading trained weights from the autoencoder checkpoint into the encoder and decoder objects. We filter the state dict to only load the relevant parts for encoder and decoder.
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in ae_ckpt["model_state_dict"].items() if k.startswith("encoder.")})
    decoder.load_state_dict({k.replace("decoder.", ""): v for k, v in ae_ckpt["model_state_dict"].items() if k.startswith("decoder.")})

    # Freeze encoder/decoder for now
    encoder.eval()
    decoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    dynamics = DynamicsModel(latent_dim=latent_dim, num_actions=4).to(device)

    optimizer = torch.optim.Adam(dynamics.parameters(), lr=lr)
    latent_loss_fn = nn.MSELoss()
    recon_loss_fn = nn.MSELoss()

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        dynamics.train()
        train_latent_loss = 0.0
        train_recon_loss = 0.0

        for batch in train_loader:
            obs = batch["obs"].to(device)
            next_obs = batch["next_obs"].to(device)
            actions = batch["action"].to(device)

            with torch.no_grad():
                z = encoder(obs) # current latent state for all samples in the batch
                z_next_true = encoder(next_obs) # true next latent for all samples in the batch

            z_next_pred = dynamics(z, actions)

            latent_loss = latent_loss_fn(z_next_pred, z_next_true)

            # Optional image-space consistency through decoder
            next_obs_pred = decoder(z_next_pred) # predicted next obs for all samples in the batch
            recon_loss = recon_loss_fn(next_obs_pred, next_obs)

            loss = latent_loss + recon_loss # total loss

            # Gradient update steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_latent_loss += latent_loss.item() * obs.size(0)
            train_recon_loss += recon_loss.item() * obs.size(0)

        train_latent_loss /= len(train_loader.dataset)
        train_recon_loss /= len(train_loader.dataset)

        dynamics.eval()
        val_loss = 0.0
        val_latent_loss = 0.0
        val_recon_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["obs"].to(device)
                next_obs = batch["next_obs"].to(device)
                actions = batch["action"].to(device)

                z = encoder(obs)
                z_next_true = encoder(next_obs)
                z_next_pred = dynamics(z, actions)

                latent_loss = latent_loss_fn(z_next_pred, z_next_true)
                next_obs_pred = decoder(z_next_pred)
                recon_loss = recon_loss_fn(next_obs_pred, next_obs)

                loss = latent_loss + recon_loss

                val_loss += loss.item() * obs.size(0)
                val_latent_loss += latent_loss.item() * obs.size(0)
                val_recon_loss += recon_loss.item() * obs.size(0)

        val_loss /= len(val_loader.dataset)
        val_latent_loss /= len(val_loader.dataset)
        val_recon_loss /= len(val_loader.dataset)

        print(
            f"Epoch {epoch:02d} | "
            f"train_latent={train_latent_loss:.6f} | train_recon={train_recon_loss:.6f} | "
            f"val_latent={val_latent_loss:.6f} | val_recon={val_recon_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "dynamics_state_dict": dynamics.state_dict(),
                    "latent_dim": latent_dim,
                },
                os.path.join(save_dir, "best_dynamics.pt"),
            )

    print("Dynamics training finished.")
    print("Best val total loss:", best_val_loss)


if __name__ == "__main__":
    train_dynamics()
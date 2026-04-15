import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.world_model_dataset import TransitionDataset
from models.models import AutoEncoder


def inspect_model(
    dataset_path="tiny_nav_dataset.npz",
    ckpt_path="checkpoints/best_autoencoder.pt",
    latent_dim=32,
    num_examples=6,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TransitionDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=num_examples, shuffle=True)

    model = AutoEncoder(latent_dim=latent_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    batch = next(iter(loader))
    obs = batch["obs"].to(device)

    with torch.no_grad():
        recon, z = model(obs)

    obs = obs.cpu().numpy()
    recon = recon.cpu().numpy()

    fig, axes = plt.subplots(2, num_examples, figsize=(3 * num_examples, 6))

    for i in range(num_examples):
        axes[0, i].imshow(obs[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, i].set_title(f"Original {i}")
        axes[0, i].axis("off")

        axes[1, i].imshow(recon[i, 0], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, i].set_title(f"Recon {i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()

    print("Latent shape:", z.shape)
    print("Sample latent vector:", z[0].cpu().numpy())


if __name__ == "__main__":
    inspect_model()
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.world_model_dataset import TransitionDataset
from models.models import Encoder, Decoder, DynamicsModel


ACTION_NAMES = ["forward", "turn_left", "turn_right", "stay"]


def inspect_dynamics(
    dataset_path="tiny_nav_dataset.npz",
    ae_ckpt_path="checkpoints/best_autoencoder.pt",
    dyn_ckpt_path="checkpoints/best_dynamics.pt",
    latent_dim=32,
    num_examples=6,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TransitionDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=num_examples, shuffle=True)

    encoder = Encoder(latent_dim=latent_dim).to(device)
    decoder = Decoder(latent_dim=latent_dim).to(device)
    dynamics = DynamicsModel(latent_dim=latent_dim, num_actions=4).to(device)

    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    encoder.load_state_dict({k.replace("encoder.", ""): v for k, v in ae_ckpt["model_state_dict"].items() if k.startswith("encoder.")})
    decoder.load_state_dict({k.replace("decoder.", ""): v for k, v in ae_ckpt["model_state_dict"].items() if k.startswith("decoder.")})

    dyn_ckpt = torch.load(dyn_ckpt_path, map_location=device)
    dynamics.load_state_dict(dyn_ckpt["dynamics_state_dict"])

    encoder.eval()
    decoder.eval()
    dynamics.eval()

    batch = next(iter(loader))
    obs = batch["obs"].to(device)
    next_obs = batch["next_obs"].to(device)
    actions = batch["action"].to(device)

    with torch.no_grad():
        z = encoder(obs)
        z_next_pred = dynamics(z, actions)
        next_obs_pred = decoder(z_next_pred)

    obs = obs.cpu().numpy()
    next_obs = next_obs.cpu().numpy()
    next_obs_pred = next_obs_pred.cpu().numpy()
    actions = actions.cpu().numpy()

    fig, axes = plt.subplots(3, num_examples, figsize=(3 * num_examples, 8))

    for i in range(num_examples):
        axes[0, i].imshow(obs[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"Current\n{ACTION_NAMES[actions[i]]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(next_obs[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title("True Next")
        axes[1, i].axis("off")

        axes[2, i].imshow(next_obs_pred[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[2, i].set_title("Pred Next")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    inspect_dynamics()
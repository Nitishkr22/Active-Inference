import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.world_model_sequence_dataset import SequenceTransitionDataset
from models.models import WorldModel

ACTION_NAMES = ["forward", "turn_left", "turn_right", "stay"]


def inspect_world_model_stacked(
    dataset_path="tiny_nav_dataset.npz",
    ckpt_path="checkpoints/best_world_model_stacked.pt",
    latent_dim=32,
    num_examples=6,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SequenceTransitionDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=num_examples, shuffle=True)

    model = WorldModel(latent_dim=latent_dim, num_actions=4, in_channels=2).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    batch = next(iter(loader))

    obs_prev = batch["obs_prev"].to(device)
    obs = batch["obs"].to(device)
    obs_stack = batch["obs_stack"].to(device)
    next_obs = batch["next_obs"].to(device)
    actions = batch["action"].to(device)

    curr_obs = batch["obs"].to(device)
    next_stack = torch.cat([curr_obs, next_obs], dim=1)

    with torch.no_grad():
        out = model(obs_stack, actions, next_obs=next_stack)

    obs_prev = obs_prev.cpu().numpy()
    obs = obs.cpu().numpy()
    next_obs = next_obs.cpu().numpy()
    next_obs_pred = out["next_obs_pred"].cpu().numpy()
    actions = actions.cpu().numpy()

    fig, axes = plt.subplots(4, num_examples, figsize=(3 * num_examples, 10))

    for i in range(num_examples):
        axes[0, i].imshow(obs_prev[i, 0], cmap="gray")
        axes[0, i].set_title("Prev")
        axes[0, i].axis("off")

        axes[1, i].imshow(obs[i, 0], cmap="gray")
        axes[1, i].set_title(f"Current\n{ACTION_NAMES[actions[i]]}")
        axes[1, i].axis("off")

        axes[2, i].imshow(next_obs[i, 0], cmap="gray")
        axes[2, i].set_title("True Next")
        axes[2, i].axis("off")

        axes[3, i].imshow(next_obs_pred[i, 0], cmap="gray")
        axes[3, i].set_title("Pred Next")
        axes[3, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    inspect_world_model_stacked()
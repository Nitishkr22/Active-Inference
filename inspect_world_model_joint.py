import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.world_model_dataset import TransitionDataset
from models.models import WorldModel

ACTION_NAMES = ["forward", "turn_left", "turn_right", "stay"]


def inspect_world_model_joint(
    dataset_path="tiny_nav_dataset.npz",
    ckpt_path="checkpoints/best_world_model_joint.pt",
    latent_dim=32,
    num_examples=6,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TransitionDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=num_examples, shuffle=True)

    model = WorldModel(latent_dim=latent_dim, num_actions=4).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    batch = next(iter(loader))
    obs = batch["obs"].to(device)
    next_obs = batch["next_obs"].to(device)
    actions = batch["action"].to(device)

    with torch.no_grad():
        out = model(obs, actions, next_obs=next_obs)

    obs = obs.cpu().numpy()
    next_obs = next_obs.cpu().numpy()
    next_obs_pred = out["next_obs_pred"].cpu().numpy()
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
    inspect_world_model_joint()
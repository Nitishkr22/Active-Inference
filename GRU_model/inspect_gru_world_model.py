import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sequence_dataset import SequenceDataset
from gru_world_model import GRUWorldModel

ACTION_NAMES = ["forward", "turn_left", "turn_right", "stay"]


def inspect_gru_world_model(
    dataset_path="tiny_nav_sequence_dataset.npz",
    ckpt_path="checkpoints/best_gru_world_model.pt",
    num_examples=4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SequenceDataset(dataset_path)
    loader = DataLoader(dataset, batch_size=num_examples, shuffle=True)

    model = GRUWorldModel().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    batch = next(iter(loader))
    obs = batch["obs"].to(device)          # [B,T,1,64,64]
    actions = batch["actions"].to(device)  # [B,T-1]

    with torch.no_grad():
        out = model.forward_sequence(obs, actions)

    recon = out["reconstructions"].cpu().numpy()
    obs_np = obs.cpu().numpy()
    actions_np = actions.cpu().numpy()

    B, T = obs_np.shape[:2]

    fig, axes = plt.subplots(2 * T, B, figsize=(3 * B, 2.5 * 2 * T))

    for b in range(B):
        for t in range(T):
            axes[2 * t, b].imshow(obs_np[b, t, 0], cmap="gray")
            if t < T - 1:
                axes[2 * t, b].set_title(f"Obs t={t}\na={ACTION_NAMES[actions_np[b, t]]}")
            else:
                axes[2 * t, b].set_title(f"Obs t={t}")
            axes[2 * t, b].axis("off")

            axes[2 * t + 1, b].imshow(recon[b, t, 0], cmap="gray")
            axes[2 * t + 1, b].set_title(f"Recon t={t}")
            axes[2 * t + 1, b].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    inspect_gru_world_model()
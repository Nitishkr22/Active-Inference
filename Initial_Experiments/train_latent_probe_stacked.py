import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from models.world_model_sequence_dataset import SequenceTransitionDataset
from models.models import WorldModel


class PoseProbe(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=64, num_headings=4):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.row_head = nn.Linear(hidden_dim, 10)
        self.col_head = nn.Linear(hidden_dim, 10)
        self.heading_head = nn.Linear(hidden_dim, num_headings)

    def forward(self, z):
        h = self.shared(z)
        return self.row_head(h), self.col_head(h), self.heading_head(h)


def train_probe_stacked(
    dataset_path="tiny_nav_dataset.npz",
    wm_ckpt_path="checkpoints/best_world_model_stacked.pt",
    latent_dim=32,
    batch_size=64,
    lr=1e-3,
    epochs=15,
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

    wm = WorldModel(latent_dim=latent_dim, num_actions=4, in_channels=2).to(device)
    ckpt = torch.load(wm_ckpt_path, map_location=device)
    wm.load_state_dict(ckpt["model_state_dict"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    probe = PoseProbe(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        probe.train()
        for batch in train_loader:
            obs_stack = batch["obs_stack"].to(device)
            pos = batch["pos"].to(device)
            heading = batch["heading"].to(device)

            with torch.no_grad():
                z = wm.encoder(obs_stack)

            row_logits, col_logits, heading_logits = probe(z)

            loss = (
                ce(row_logits, pos[:, 0]) +
                ce(col_logits, pos[:, 1]) +
                ce(heading_logits, heading)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        total = 0
        correct_row = 0
        correct_col = 0
        correct_heading = 0

        with torch.no_grad():
            for batch in val_loader:
                obs_stack = batch["obs_stack"].to(device)
                pos = batch["pos"].to(device)
                heading = batch["heading"].to(device)

                z = wm.encoder(obs_stack)
                row_logits, col_logits, heading_logits = probe(z)

                row_pred = row_logits.argmax(dim=1)
                col_pred = col_logits.argmax(dim=1)
                heading_pred = heading_logits.argmax(dim=1)

                total += obs_stack.size(0)
                correct_row += (row_pred == pos[:, 0]).sum().item()
                correct_col += (col_pred == pos[:, 1]).sum().item()
                correct_heading += (heading_pred == heading).sum().item()

        print(
            f"Epoch {epoch:02d} | "
            f"row_acc={correct_row/total:.3f} | "
            f"col_acc={correct_col/total:.3f} | "
            f"heading_acc={correct_heading/total:.3f}"
        )


if __name__ == "__main__":
    train_probe_stacked()
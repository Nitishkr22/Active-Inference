import numpy as np
import torch
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)

        self.obs = data["obs"].astype(np.float32)          # [N, 64, 64]
        self.actions = data["actions"].astype(np.int64)    # [N]
        self.next_obs = data["next_obs"].astype(np.float32)
        self.done = data["done"].astype(np.float32)

        # Optional debug info
        self.pos = data["pos"].astype(np.int64)
        self.heading = data["heading"].astype(np.int64)
        self.next_pos = data["next_pos"].astype(np.int64)
        self.next_heading = data["next_heading"].astype(np.int64)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs[idx]).unsqueeze(0)       # [1, 64, 64]
        next_obs = torch.from_numpy(self.next_obs[idx]).unsqueeze(0)
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        done = torch.tensor(self.done[idx], dtype=torch.float32)

        sample = {
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "done": done,
            "pos": torch.tensor(self.pos[idx], dtype=torch.long),
            "heading": torch.tensor(self.heading[idx], dtype=torch.long),
            "next_pos": torch.tensor(self.next_pos[idx], dtype=torch.long),
            "next_heading": torch.tensor(self.next_heading[idx], dtype=torch.long),
        }
        return sample
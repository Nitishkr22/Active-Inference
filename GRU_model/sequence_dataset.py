import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)

        self.obs = data["obs"].astype(np.float32)          # [N, T, 64, 64]
        self.actions = data["actions"].astype(np.int64)    # [N, T-1]
        self.pos = data["pos"].astype(np.int64)            # [N, T, 2]
        self.heading = data["heading"].astype(np.int64)    # [N, T]
        self.done = data["done"].astype(np.float32)        # [N, T]

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs[idx]).unsqueeze(1)     # [T, 1, 64, 64]
        actions = torch.from_numpy(self.actions[idx])          # [T-1]
        pos = torch.from_numpy(self.pos[idx])                  # [T, 2]
        heading = torch.from_numpy(self.heading[idx])          # [T]
        done = torch.from_numpy(self.done[idx])                # [T]

        return {
            "obs": obs,
            "actions": actions,
            "pos": pos,
            "heading": heading,
            "done": done,
        }
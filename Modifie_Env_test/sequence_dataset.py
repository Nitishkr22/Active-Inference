## sequence_dataset.py ##

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

        # optional field
        self.goal_pos = data["goal_pos"].astype(np.int64) if "goal_pos" in data.files else None

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.obs[idx]).unsqueeze(1)   # [T, 1, 64, 64]
        actions = torch.from_numpy(self.actions[idx])        # [T-1]
        pos = torch.from_numpy(self.pos[idx])                # [T, 2]
        heading = torch.from_numpy(self.heading[idx])        # [T]
        done = torch.from_numpy(self.done[idx])              # [T]

        sample = {
            "obs": obs,
            "actions": actions,
            "pos": pos,
            "heading": heading,
            "done": done,
        }

        if self.goal_pos is not None:
            sample["goal_pos"] = torch.from_numpy(self.goal_pos[idx])   # [T, 2]

        return sample
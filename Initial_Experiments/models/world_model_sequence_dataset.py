# import numpy as np
# import torch
# from torch.utils.data import Dataset


# class SequenceTransitionDataset(Dataset):
#     """
#     Returns samples with:
#       obs_prev  : previous observation
#       obs       : current observation
#       next_obs  : next observation
#       action    : action taken at current step
#       done      : whether next state is terminal

#     Also includes debugging pose info.
#     """

#     def __init__(self, npz_path: str):
#         data = np.load(npz_path)

#         self.obs = data["obs"].astype(np.float32)              # [N, 64, 64]
#         self.actions = data["actions"].astype(np.int64)        # [N]
#         self.next_obs = data["next_obs"].astype(np.float32)    # [N, 64, 64]
#         self.done = data["done"].astype(np.float32)

#         self.pos = data["pos"].astype(np.int64)
#         self.heading = data["heading"].astype(np.int64)
#         self.next_pos = data["next_pos"].astype(np.int64)
#         self.next_heading = data["next_heading"].astype(np.int64)

#         # Build valid indices so we can get obs_{t-1}
#         # We avoid transitions where previous sample belongs to a different episode.
#         self.valid_indices = []
#         for i in range(1, len(self.obs)):
#             # if previous transition ended an episode, do not use it as obs_prev
#             if self.done[i - 1] > 0.5:
#                 continue
#             self.valid_indices.append(i)

#     def __len__(self):
#         return len(self.valid_indices)

#     def __getitem__(self, idx):
#         i = self.valid_indices[idx]

#         obs_prev = torch.from_numpy(self.obs[i - 1]).unsqueeze(0)   # [1, 64, 64]
#         obs = torch.from_numpy(self.obs[i]).unsqueeze(0)            # [1, 64, 64]
#         next_obs = torch.from_numpy(self.next_obs[i]).unsqueeze(0)  # [1, 64, 64]

#         # stack previous and current frames into 2 channels
#         obs_stack = torch.cat([obs_prev, obs], dim=0)               # [2, 64, 64]

#         action = torch.tensor(self.actions[i], dtype=torch.long)
#         done = torch.tensor(self.done[i], dtype=torch.float32)

#         sample = {
#             "obs_stack": obs_stack,
#             "obs_prev": obs_prev,
#             "obs": obs,
#             "next_obs": next_obs,
#             "action": action,
#             "done": done,
#             "pos": torch.tensor(self.pos[i], dtype=torch.long),
#             "heading": torch.tensor(self.heading[i], dtype=torch.long),
#             "next_pos": torch.tensor(self.next_pos[i], dtype=torch.long),
#             "next_heading": torch.tensor(self.next_heading[i], dtype=torch.long),
#         }
#         return sample

## after collecting exhaustive dataset use obs_prev directly ##

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceTransitionDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)

        self.obs_prev = data["obs_prev"].astype(np.float32)     # [N, 64, 64]
        self.obs = data["obs"].astype(np.float32)               # [N, 64, 64]
        self.actions = data["actions"].astype(np.int64)
        self.next_obs = data["next_obs"].astype(np.float32)
        self.done = data["done"].astype(np.float32)

        self.pos = data["pos"].astype(np.int64)
        self.heading = data["heading"].astype(np.int64)
        self.next_pos = data["next_pos"].astype(np.int64)
        self.next_heading = data["next_heading"].astype(np.int64)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs_prev = torch.from_numpy(self.obs_prev[idx]).unsqueeze(0)   # [1,64,64]
        obs = torch.from_numpy(self.obs[idx]).unsqueeze(0)
        next_obs = torch.from_numpy(self.next_obs[idx]).unsqueeze(0)

        obs_stack = torch.cat([obs_prev, obs], dim=0)                  # [2,64,64]

        return {
            "obs_prev": obs_prev,
            "obs": obs,
            "obs_stack": obs_stack,
            "next_obs": next_obs,
            "action": torch.tensor(self.actions[idx], dtype=torch.long),
            "done": torch.tensor(self.done[idx], dtype=torch.float32),
            "pos": torch.tensor(self.pos[idx], dtype=torch.long),
            "heading": torch.tensor(self.heading[idx], dtype=torch.long),
            "next_pos": torch.tensor(self.next_pos[idx], dtype=torch.long),
            "next_heading": torch.tensor(self.next_heading[idx], dtype=torch.long),
        }
# dataset_loader.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, random_split, DataLoader


# ============================================================
# Config
# ============================================================

@dataclass
class LoaderConfig:
    """
    Data loader configuration.

    Notes:
    - batch_size should be tuned based on GPU memory and model size.
    - num_workers should depend on CPU / storage / platform.
    - pin_memory is useful for CUDA training.
    - persistent_workers and prefetch_factor only matter when num_workers > 0.
    """
    dataset_path: str
    val_fraction: float = 0.1

    # Training-oriented defaults (not tiny debug defaults)
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    seed: int = 42

    # DataLoader performance options
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Keep last incomplete batch or not
    drop_last_train: bool = False
    drop_last_val: bool = False


# ============================================================
# Dataset
# ============================================================

class SequenceWorldModelDataset(Dataset):
    """
    PyTorch dataset for the symbolic world-model dataset.

    Expected arrays inside .npz:
      observations : [N, T, 1, H, W] float32
      actions      : [N, T-1] int64
      rows         : [N, T] int64
      cols         : [N, T] int64
      headings     : [N, T] int64
      collisions   : [N, T-1] uint8 or int
    """

    def __init__(self, dataset_path: str) -> None:
        super().__init__()

        data = np.load(dataset_path)

        required_keys = [
            "observations",
            "actions",
            "rows",
            "cols",
            "headings",
            "collisions",
        ]
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Missing required key '{key}' in dataset: {dataset_path}")

        self.observations = torch.from_numpy(data["observations"]).float()   # [N,T,1,H,W]
        self.actions = torch.from_numpy(data["actions"]).long()              # [N,T-1]
        self.rows = torch.from_numpy(data["rows"]).long()                    # [N,T]
        self.cols = torch.from_numpy(data["cols"]).long()                    # [N,T]
        self.headings = torch.from_numpy(data["headings"]).long()            # [N,T]
        self.collisions = torch.from_numpy(data["collisions"]).float()       # [N,T-1]

        # Optional metadata
        self.action_names = data["action_names"] if "action_names" in data else None
        self.heading_names = data["heading_names"] if "heading_names" in data else None

        # ----------------------------------------------------
        # Shape checks
        # ----------------------------------------------------
        N = self.observations.shape[0]

        if self.actions.shape[0] != N:
            raise ValueError("Mismatch: observations and actions have different number of episodes.")
        if self.rows.shape[0] != N or self.cols.shape[0] != N or self.headings.shape[0] != N:
            raise ValueError("Mismatch: observations and pose labels have different number of episodes.")
        if self.collisions.shape[0] != N:
            raise ValueError("Mismatch: observations and collisions have different number of episodes.")

        if self.observations.ndim != 5:
            raise ValueError(
                f"observations must have shape [N,T,1,H,W], got {tuple(self.observations.shape)}"
            )

        _, T, C, H, W = self.observations.shape
        if C != 1:
            raise ValueError(f"Expected grayscale channel dimension C=1, got C={C}")

        if self.actions.shape[1] != T - 1:
            raise ValueError(
                f"actions second dim should be T-1={T-1}, got {self.actions.shape[1]}"
            )
        if self.rows.shape[1] != T or self.cols.shape[1] != T or self.headings.shape[1] != T:
            raise ValueError("rows/cols/headings must have second dimension T")
        if self.collisions.shape[1] != T - 1:
            raise ValueError("collisions must have second dimension T-1")

        self.num_episodes = N
        self.seq_len = T
        self.channels = C
        self.height = H
        self.width = W

    def __len__(self) -> int:
        return self.num_episodes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns one full sequence.

        Shapes:
          observations : [T,1,H,W]
          actions      : [T-1]
          rows         : [T]
          cols         : [T]
          headings     : [T]
          collisions   : [T-1]
        """
        sample = {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rows": self.rows[idx],
            "cols": self.cols[idx],
            "headings": self.headings[idx],
            "collisions": self.collisions[idx],
        }
        return sample

    def get_num_row_classes(self) -> int:
        return int(self.rows.max().item()) + 1

    def get_num_col_classes(self) -> int:
        return int(self.cols.max().item()) + 1

    def get_num_heading_classes(self) -> int:
        return int(self.headings.max().item()) + 1

    def summary(self) -> Dict[str, Any]:
        return {
            "num_episodes": self.num_episodes,
            "seq_len": self.seq_len,
            "channels": self.channels,
            "height": self.height,
            "width": self.width,
            "num_row_classes": self.get_num_row_classes(),
            "num_col_classes": self.get_num_col_classes(),
            "num_heading_classes": self.get_num_heading_classes(),
        }


# ============================================================
# Loader helpers
# ============================================================

def create_train_val_datasets(
    cfg: LoaderConfig,
) -> Tuple[SequenceWorldModelDataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    Returns:
      full_dataset, train_dataset, val_dataset
    """
    full_dataset = SequenceWorldModelDataset(cfg.dataset_path)

    num_total = len(full_dataset)
    num_val = int(round(cfg.val_fraction * num_total))
    num_val = max(1, num_val)
    num_train = num_total - num_val

    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [num_train, num_val],
        generator=generator,
    )

    return full_dataset, train_dataset, val_dataset


def _build_loader_kwargs(cfg: LoaderConfig, is_train: bool) -> Dict[str, Any]:
    """
    Build DataLoader kwargs in a safe way.

    Important:
    - persistent_workers only works when num_workers > 0
    - prefetch_factor only applies when num_workers > 0
    """
    kwargs: Dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": cfg.shuffle_train if is_train else False,
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "drop_last": cfg.drop_last_train if is_train else cfg.drop_last_val,
    }

    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        kwargs["prefetch_factor"] = cfg.prefetch_factor

    return kwargs


def create_dataloaders(
    cfg: LoaderConfig,
) -> Tuple[SequenceWorldModelDataset, DataLoader, DataLoader]:
    """
    Returns:
      full_dataset, train_loader, val_loader
    """
    full_dataset, train_dataset, val_dataset = create_train_val_datasets(cfg)

    train_loader = DataLoader(
        train_dataset,
        **_build_loader_kwargs(cfg, is_train=True),
    )

    val_loader = DataLoader(
        val_dataset,
        **_build_loader_kwargs(cfg, is_train=False),
    )

    return full_dataset, train_loader, val_loader

def build_dataloaders(
    cfg: LoaderConfig,
) -> Tuple[SequenceWorldModelDataset, DataLoader, DataLoader]:
    """
    Returns:
      full_dataset, train_loader, val_loader
    """
    full_dataset, train_dataset, val_dataset = create_train_val_datasets(cfg)

    train_loader = DataLoader(
        train_dataset,
        **_build_loader_kwargs(cfg, is_train=True),
    )

    val_loader = DataLoader(
        val_dataset,
        **_build_loader_kwargs(cfg, is_train=False),
    )

    return full_dataset, train_loader, val_loader


# ============================================================
# Debug / sanity test
# ============================================================

def debug_print_batch(batch: Dict[str, torch.Tensor]) -> None:
    print("\nBatch keys and shapes:")
    for k, v in batch.items():
        print(f"  {k:12s}: shape={tuple(v.shape)}, dtype={v.dtype}")


if __name__ == "__main__":
    # Update this path before running directly
    dataset_path = "../../dataset/train_dataset_v7.npz"

    # These defaults are now more training-oriented.
    # Final values should still be tuned in the training script
    # depending on the actual machine (A100 vs 3080 etc).
    cfg = LoaderConfig(
        dataset_path=dataset_path,
        val_fraction=0.1,
        batch_size=64,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        shuffle_train=True,
        seed=42,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last_train=False,
        drop_last_val=False,
    )

    full_dataset, train_loader, val_loader = build_dataloaders(cfg)

    print("Dataset summary:")
    summary = full_dataset.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nLoader config:")
    print(f"  batch_size:         {cfg.batch_size}")
    print(f"  num_workers:        {cfg.num_workers}")
    print(f"  pin_memory:         {cfg.pin_memory}")
    print(f"  persistent_workers: {cfg.persistent_workers if cfg.num_workers > 0 else False}")
    print(f"  prefetch_factor:    {cfg.prefetch_factor if cfg.num_workers > 0 else 'N/A'}")

    train_batch = next(iter(train_loader))
    debug_print_batch(train_batch)

    val_batch = next(iter(val_loader))
    print("\nValidation batch loaded successfully.")
    debug_print_batch(val_batch)
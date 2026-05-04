# train_predictive_latent_v3.py

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model_v3 import WorldModelV3, ModelV3Config


# ============================================================
# Config
# ============================================================

@dataclass
class TrainConfig:
    dataset_path: str = "../../dataset/train_dataset_v7.npz"

    checkpoint_path: str = "./checkpoints_v32/best_model_v32.pt"
    model_config_json: str = "./checkpoints_v32/model_config.json"

    output_dir: str = "./checkpoints_v3_predictive_latent"
    output_name: str = "best_predictive_latent.pt"

    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5

    rollout_horizon: int = 8
    num_start_times_per_batch: int = 4

    # Loss weights
    w_context: float = 1.0
    w_pose: float = 1.0
    w_collision: float = 0.25
    w_entropy: float = 0.01

    val_fraction: float = 0.15
    seed: int = 42
    num_workers: int = 2

    grad_clip: float = 1.0


# ============================================================
# Dataset
# ============================================================

class SequenceDataset(Dataset):
    def __init__(self, npz_path: str, indices: np.ndarray) -> None:
        data = np.load(npz_path)

        self.observations = data["observations"][indices].astype(np.float32)
        self.actions = data["actions"][indices].astype(np.int64)
        self.rows = data["rows"][indices].astype(np.int64)
        self.cols = data["cols"][indices].astype(np.int64)
        self.headings = data["headings"][indices].astype(np.int64)

        self.collisions = None
        if "collisions" in data:
            self.collisions = data["collisions"][indices].astype(np.int64)

        # expected observations: [N,T,1,64,64]
        if self.observations.ndim == 4:
            self.observations = self.observations[:, :, None, :, :]

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "observations": torch.from_numpy(self.observations[idx]),
            "actions": torch.from_numpy(self.actions[idx]),
            "rows": torch.from_numpy(self.rows[idx]),
            "cols": torch.from_numpy(self.cols[idx]),
            "headings": torch.from_numpy(self.headings[idx]),
        }

        if self.collisions is not None:
            item["collisions"] = torch.from_numpy(self.collisions[idx])

        return item


def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    data = np.load(cfg.dataset_path)
    n = data["observations"].shape[0]

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(n)

    n_val = int(round(cfg.val_fraction * n))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_ds = SequenceDataset(cfg.dataset_path, train_idx)
    val_ds = SequenceDataset(cfg.dataset_path, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


# ============================================================
# Model loading
# ============================================================

def load_model_config_from_json(path: str) -> ModelV3Config:
    with open(path, "r") as f:
        return ModelV3Config(**json.load(f))


def build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_config_json: Optional[str] = None,
) -> WorldModelV3:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_config" in ckpt:
        model_cfg = ModelV3Config(**ckpt["model_config"])
    else:
        if model_config_json is None or not Path(model_config_json).exists():
            raise ValueError("Checkpoint has no model_config and model_config_json was not found.")
        print(f"Loaded model_config from JSON: {model_config_json}")
        model_cfg = load_model_config_from_json(model_config_json)

    model = WorldModelV3(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.train()

    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


# ============================================================
# Helpers
# ============================================================

def choose_start_times(
    T: int,
    rollout_horizon: int,
    num_start_times: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Need actions from t0 to t0+H-1 and targets at t0+1 ... t0+H.
    Therefore t0 <= T-H-1.
    """
    max_t0 = T - rollout_horizon - 1
    if max_t0 < 1:
        raise ValueError(f"Sequence too short T={T} for rollout_horizon={rollout_horizon}")

    if max_t0 == 1:
        return torch.ones(num_start_times, dtype=torch.long, device=device)

    return torch.randint(
        low=1,
        high=max_t0 + 1,
        size=(num_start_times,),
        device=device,
    )


def entropy_loss(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Small entropy penalty to avoid overly diffuse rollouts.
    probs: [B,H,K]
    """
    probs = probs.clamp_min(eps)
    ent = -(probs * probs.log()).sum(dim=-1)
    return ent.mean()


def get_context_rollout(roll: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    for key in ["context_roll", "context_rollout", "context_probs_roll", "context_seq_roll"]:
        if key in roll:
            return roll[key]
    return None


def compute_rollout_loss_for_t0(
    model: WorldModelV3,
    filter_out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    t0: int,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Teacher-forced latent at t0 -> free rollout H steps.
    Compare rollout to teacher-forced future context and true future pose.
    """
    actions = batch["actions"]
    rows = batch["rows"]
    cols = batch["cols"]
    headings = batch["headings"]

    B = actions.shape[0]
    H = cfg.rollout_horizon

    row0 = filter_out["row_probs_seq"][:, t0, :]
    col0 = filter_out["col_probs_seq"][:, t0, :]
    hdg0 = filter_out["heading_probs_seq"][:, t0, :]

    ctx0 = None
    if "context_seq" in filter_out:
        ctx0 = filter_out["context_seq"][:, t0, :]

    action_seq = actions[:, t0 : t0 + H]

    roll = model.rollout_from_filtered_state(
        row_probs_start=row0,
        col_probs_start=col0,
        heading_probs_start=hdg0,
        context_start=ctx0,
        action_seq=action_seq,
    )

    row_roll = roll["row_probs_roll"]
    col_roll = roll["col_probs_roll"]
    hdg_roll = roll["heading_probs_roll"]

    target_rows = rows[:, t0 + 1 : t0 + H + 1]
    target_cols = cols[:, t0 + 1 : t0 + H + 1]
    target_hdgs = headings[:, t0 + 1 : t0 + H + 1]

    pose_loss = (
        F.nll_loss(torch.log(row_roll.clamp_min(1e-8)).reshape(B * H, -1), target_rows.reshape(-1))
        + F.nll_loss(torch.log(col_roll.clamp_min(1e-8)).reshape(B * H, -1), target_cols.reshape(-1))
        + F.nll_loss(torch.log(hdg_roll.clamp_min(1e-8)).reshape(B * H, -1), target_hdgs.reshape(-1))
    )

    ctx_loss = torch.tensor(0.0, device=actions.device)
    ctx_roll = get_context_rollout(roll)

    if ctx0 is not None and ctx_roll is not None:
        target_ctx = filter_out["context_seq"][:, t0 + 1 : t0 + H + 1, :]
        ctx_loss = F.mse_loss(ctx_roll, target_ctx)

    collision_loss = torch.tensor(0.0, device=actions.device)
    if "collisions" in batch and "collision_logits_roll" in roll:
        target_coll = batch["collisions"][:, t0 : t0 + H]
        coll_logits = roll["collision_logits_roll"]
        collision_loss = F.cross_entropy(
            coll_logits.reshape(B * H, 2),
            target_coll.reshape(-1),
        )

    ent_loss = (
        entropy_loss(row_roll)
        + entropy_loss(col_roll)
        + entropy_loss(hdg_roll)
    )

    total = (
        cfg.w_context * ctx_loss
        + cfg.w_pose * pose_loss
        + cfg.w_collision * collision_loss
        + cfg.w_entropy * ent_loss
    )

    metrics = {
        "loss": float(total.detach().item()),
        "context_loss": float(ctx_loss.detach().item()),
        "pose_loss": float(pose_loss.detach().item()),
        "collision_loss": float(collision_loss.detach().item()),
        "entropy_loss": float(ent_loss.detach().item()),
    }

    return total, metrics


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    return {
        k: float(np.mean([m[k] for m in metrics_list]))
        for k in keys
    }


# ============================================================
# Train / validate
# ============================================================

def run_epoch(
    model: WorldModelV3,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    cfg: TrainConfig,
    device: torch.device,
    train: bool,
) -> Dict[str, float]:

    model.train(train)
    all_metrics: List[Dict[str, float]] = []

    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        obs = batch["observations"]
        actions = batch["actions"]

        B, T = obs.shape[:2]

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            filter_out = model.forward_filter(obs, actions)

            start_times = choose_start_times(
                T=T,
                rollout_horizon=cfg.rollout_horizon,
                num_start_times=cfg.num_start_times_per_batch,
                device=device,
            )

            total_loss = torch.tensor(0.0, device=device)
            metrics_accum: List[Dict[str, float]] = []

            for t0_t in start_times:
                t0 = int(t0_t.item())
                loss_t, metrics_t = compute_rollout_loss_for_t0(
                    model=model,
                    filter_out=filter_out,
                    batch=batch,
                    t0=t0,
                    cfg=cfg,
                )
                total_loss = total_loss + loss_t
                metrics_accum.append(metrics_t)

            total_loss = total_loss / len(start_times)

            if train:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                assert optimizer is not None
                optimizer.step()

        all_metrics.append(aggregate_metrics(metrics_accum))

    return aggregate_metrics(all_metrics)


def save_checkpoint(
    model: WorldModelV3,
    cfg: TrainConfig,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
        "model_config": model.cfg.__dict__ if hasattr(model.cfg, "__dict__") else None,
        "train_config": cfg.__dict__,
    }

    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}")


# ============================================================
# Main
# ============================================================

def main():
    cfg = TrainConfig()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("TRAIN PREDICTIVE LATENT V3")
    print("=" * 100)
    print(f"Using device: {device}")
    print(f"Dataset: {cfg.dataset_path}")
    print(f"Checkpoint: {cfg.checkpoint_path}")
    print(f"Output dir: {cfg.output_dir}")
    print(f"Rollout horizon: {cfg.rollout_horizon}")

    train_loader, val_loader = build_dataloaders(cfg)

    model = build_model_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        device=device,
        model_config_json=cfg.model_config_json,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val = float("inf")
    best_path = Path(cfg.output_dir) / cfg.output_name

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            train=True,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                cfg=cfg,
                device=device,
                train=False,
            )

        elapsed = time.time() - t0
        val_loss = val_metrics["loss"]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} | "
            f"val_ctx={val_metrics['context_loss']:.6f} | "
            f"val_pose={val_metrics['pose_loss']:.6f} | "
            f"val_coll={val_metrics['collision_loss']:.6f} | "
            f"time={elapsed:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                model=model,
                cfg=cfg,
                epoch=epoch,
                val_loss=val_loss,
                path=best_path,
            )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Best val loss: {best_val:.6f}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
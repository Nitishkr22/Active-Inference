# train_v1.py

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset_loader import LoaderConfig, create_dataloaders
from model_v1 import ModelV1Config, WorldModelV1


# ============================================================
# Training config
# ============================================================

@dataclass
class TrainConfig:
    # --------------------------------------------------------
    # Dataset / loaders
    # Defaults chosen for RTX 3080-style training
    # --------------------------------------------------------
    dataset_path: str = "../../dataset/train_dataset__v6.npz"
    val_fraction: float = 0.1

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 30
    grad_clip_norm: float = 1.0

    # --------------------------------------------------------
    # Rollout
    # --------------------------------------------------------
    rollout_start_t_min: int = 4
    rollout_horizon: int = 5

    # --------------------------------------------------------
    # Loss weights
    # Slightly more reconstruction-friendly than before
    # --------------------------------------------------------
    w_recon: float = 1.0
    w_pose: float = 0.8
    w_roll_latent: float = 1.0
    w_roll_pose: float = 1.0
    w_roll_recon: float = 1.0

    # --------------------------------------------------------
    # Logging / saving
    # --------------------------------------------------------
    print_every: int = 50
    save_dir: str = "./checkpoints_v1"

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------
    use_amp: bool = True
    use_channels_last: bool = False

    # For reproducibility/debugging.
    # deterministic=True can reduce machine-to-machine variation,
    # but may slow training.
    deterministic: bool = False

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------
    seed: int = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # Good default on Ampere GPUs like RTX 3080
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Helpers
# ============================================================

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def compute_pose_loss(
    row_logits: torch.Tensor,
    col_logits: torch.Tensor,
    heading_logits: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    headings: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Inputs:
      row_logits     : [B,T,R] or [B,K,R]
      col_logits     : [B,T,C] or [B,K,C]
      heading_logits : [B,T,Hd] or [B,K,Hd]
      rows           : [B,T] or [B,K]
      cols           : [B,T] or [B,K]
      headings       : [B,T] or [B,K]
    """
    B, T, R = row_logits.shape
    _, _, C = col_logits.shape
    _, _, Hd = heading_logits.shape

    row_loss = F.cross_entropy(row_logits.reshape(B * T, R), rows.reshape(B * T))
    col_loss = F.cross_entropy(col_logits.reshape(B * T, C), cols.reshape(B * T))
    heading_loss = F.cross_entropy(heading_logits.reshape(B * T, Hd), headings.reshape(B * T))

    total = row_loss + col_loss + heading_loss
    return total, {
        "row_loss": row_loss.detach(),
        "col_loss": col_loss.detach(),
        "heading_loss": heading_loss.detach(),
    }


def sample_rollout_start_and_targets(
    observations: torch.Tensor,
    z_seq: torch.Tensor,
    h_seq: torch.Tensor,
    actions: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    headings: torch.Tensor,
    rollout_start_t_min: int,
    rollout_horizon: int,
) -> Dict[str, torch.Tensor]:
    """
    Select one rollout start per batch element.

    Inputs:
      observations : [B,T,1,H,W]
      z_seq        : [B,T,Z]
      h_seq        : [B,T,H]
      actions      : [B,T-1]
      rows         : [B,T]
      cols         : [B,T]
      headings     : [B,T]

    Returns:
      z_start        : [B,Z]
      h_start        : [B,H]
      action_roll    : [B,K]
      z_target       : [B,K,Z]
      obs_target     : [B,K,1,H,W]
      row_target     : [B,K]
      col_target     : [B,K]
      heading_target : [B,K]
    """
    B, T, Z = z_seq.shape
    K = rollout_horizon

    t_min = rollout_start_t_min
    t_max = T - 1 - K
    if t_max < t_min:
        raise ValueError(
            f"Invalid rollout range: t_min={t_min}, t_max={t_max}, "
            f"T={T}, rollout_horizon={K}"
        )

    device = z_seq.device
    t0 = torch.randint(low=t_min, high=t_max + 1, size=(B,), device=device)

    z_start_list = []
    h_start_list = []
    action_roll_list = []
    z_target_list = []
    obs_target_list = []
    row_target_list = []
    col_target_list = []
    heading_target_list = []

    for b in range(B):
        t = int(t0[b].item())

        z_start_list.append(z_seq[b, t])                        # z_t
        h_start_list.append(h_seq[b, t])                        # h_t
        action_roll_list.append(actions[b, t:t + K])            # a_t ... a_{t+K-1}

        z_target_list.append(z_seq[b, t + 1:t + 1 + K])         # z_{t+1} ... z_{t+K}
        obs_target_list.append(observations[b, t + 1:t + 1 + K])
        row_target_list.append(rows[b, t + 1:t + 1 + K])
        col_target_list.append(cols[b, t + 1:t + 1 + K])
        heading_target_list.append(headings[b, t + 1:t + 1 + K])

    return {
        "z_start": torch.stack(z_start_list, dim=0),
        "h_start": torch.stack(h_start_list, dim=0),
        "action_roll": torch.stack(action_roll_list, dim=0),
        "z_target": torch.stack(z_target_list, dim=0),
        "obs_target": torch.stack(obs_target_list, dim=0),
        "row_target": torch.stack(row_target_list, dim=0),
        "col_target": torch.stack(col_target_list, dim=0),
        "heading_target": torch.stack(heading_target_list, dim=0),
    }


def compute_batch_losses(
    model: WorldModelV1,
    batch: Dict[str, torch.Tensor],
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    observations = batch["observations"]   # [B,T,1,H,W]
    actions = batch["actions"]             # [B,T-1]
    rows = batch["rows"]                   # [B,T]
    cols = batch["cols"]                   # [B,T]
    headings = batch["headings"]           # [B,T]

    # --------------------------------------------------------
    # Filtering pass on actual observations
    # --------------------------------------------------------
    filt = model.forward_filter(observations, actions)

    z_seq = filt["z_seq"]                                  # [B,T,Z]
    h_seq = filt["h_seq"]                                  # [B,T,H]
    recon_seq = filt["recon_seq"]                          # [B,T,1,H,W]
    row_logits_seq = filt["row_logits_seq"]                # [B,T,R]
    col_logits_seq = filt["col_logits_seq"]                # [B,T,C]
    heading_logits_seq = filt["heading_logits_seq"]        # [B,T,Hd]

    # --------------------------------------------------------
    # Reconstruction loss on filtered states
    # --------------------------------------------------------
    recon_loss = F.l1_loss(recon_seq, observations)

    # --------------------------------------------------------
    # Pose loss on filtered states
    # --------------------------------------------------------
    pose_loss, pose_parts = compute_pose_loss(
        row_logits_seq,
        col_logits_seq,
        heading_logits_seq,
        rows,
        cols,
        headings,
    )

    # --------------------------------------------------------
    # Rollout targets
    # --------------------------------------------------------
    rollout_targets = sample_rollout_start_and_targets(
        observations=observations,
        z_seq=z_seq,
        h_seq=h_seq,
        actions=actions,
        rows=rows,
        cols=cols,
        headings=headings,
        rollout_start_t_min=cfg.rollout_start_t_min,
        rollout_horizon=cfg.rollout_horizon,
    )

    roll = model.rollout_from_filtered_state(
        z_start=rollout_targets["z_start"],
        h_start=rollout_targets["h_start"],
        action_seq=rollout_targets["action_roll"],
    )

    z_roll = roll["z_roll"]                                # [B,K,Z]
    recon_roll = roll["recon_roll"]                        # [B,K,1,H,W]
    row_logits_roll = roll["row_logits_roll"]              # [B,K,R]
    col_logits_roll = roll["col_logits_roll"]              # [B,K,C]
    heading_logits_roll = roll["heading_logits_roll"]      # [B,K,Hd]

    z_target = rollout_targets["z_target"]                 # [B,K,Z]
    obs_target = rollout_targets["obs_target"]             # [B,K,1,H,W]
    row_target = rollout_targets["row_target"]             # [B,K]
    col_target = rollout_targets["col_target"]             # [B,K]
    heading_target = rollout_targets["heading_target"]     # [B,K]

    # --------------------------------------------------------
    # Rollout losses
    # --------------------------------------------------------
    roll_latent_loss = F.mse_loss(z_roll, z_target)

    roll_pose_loss, roll_pose_parts = compute_pose_loss(
        row_logits_roll,
        col_logits_roll,
        heading_logits_roll,
        row_target,
        col_target,
        heading_target,
    )

    # Important fix:
    # compare rollout reconstruction with TRUE future observations,
    # not decoded target latents.
    roll_recon_loss = F.l1_loss(recon_roll, obs_target)

    # --------------------------------------------------------
    # Total loss
    # --------------------------------------------------------
    total_loss = (
        cfg.w_recon * recon_loss +
        cfg.w_pose * pose_loss +
        cfg.w_roll_latent * roll_latent_loss +
        cfg.w_roll_pose * roll_pose_loss +
        cfg.w_roll_recon * roll_recon_loss
    )

    metrics = {
        "loss_total": float(total_loss.detach().item()),
        "loss_recon": float(recon_loss.detach().item()),
        "loss_pose": float(pose_loss.detach().item()),
        "loss_roll_latent": float(roll_latent_loss.detach().item()),
        "loss_roll_pose": float(roll_pose_loss.detach().item()),
        "loss_roll_recon": float(roll_recon_loss.detach().item()),
        "loss_row": float(pose_parts["row_loss"].item()),
        "loss_col": float(pose_parts["col_loss"].item()),
        "loss_heading": float(pose_parts["heading_loss"].item()),
        "loss_roll_row": float(roll_pose_parts["row_loss"].item()),
        "loss_roll_col": float(roll_pose_parts["col_loss"].item()),
        "loss_roll_heading": float(roll_pose_parts["heading_loss"].item()),
    }

    return total_loss, metrics


# ============================================================
# Epoch loops
# ============================================================

def run_epoch(
    model: WorldModelV1,
    loader: DataLoader,
    optimizer: Adam | None,
    device: torch.device,
    cfg: TrainConfig,
    train: bool,
    scaler: torch.cuda.amp.GradScaler | None,
) -> Dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_metrics: Dict[str, float] = {}
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == "cuda")):
                loss, metrics = compute_batch_losses(model, batch, cfg)

            if train:
                assert optimizer is not None
                if scaler is not None and cfg.use_amp and device.type == "cuda":
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                    optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v

        num_batches += 1

        if train and ((batch_idx + 1) % cfg.print_every == 0):
            print(
                f"  [batch {batch_idx + 1:4d}] "
                f"loss={metrics['loss_total']:.4f} | "
                f"recon={metrics['loss_recon']:.4f} | "
                f"pose={metrics['loss_pose']:.4f} | "
                f"roll_lat={metrics['loss_roll_latent']:.4f} | "
                f"roll_pose={metrics['loss_roll_pose']:.4f} | "
                f"roll_recon={metrics['loss_roll_recon']:.4f}"
            )

    avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
    return avg_metrics


# ============================================================
# Checkpointing
# ============================================================

def save_checkpoint(
    model: WorldModelV1,
    optimizer: Adam,
    epoch: int,
    cfg: TrainConfig,
    save_path: str,
    extra: Dict | None = None,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_config": cfg.__dict__,
    }

    if extra is not None:
        ckpt.update(extra)

    torch.save(ckpt, save_path)


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = TrainConfig()

    set_seed(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Loaders
    # --------------------------------------------------------
    loader_cfg = LoaderConfig(
        dataset_path=cfg.dataset_path,
        val_fraction=cfg.val_fraction,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        shuffle_train=cfg.shuffle_train,
        seed=cfg.seed,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
        drop_last_train=True,
        drop_last_val=False,
    )

    full_dataset, train_loader, val_loader = create_dataloaders(loader_cfg)
    summary = full_dataset.summary()

    print("Dataset summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model_cfg = ModelV1Config(
        obs_channels=summary["channels"],
        obs_height=summary["height"],
        obs_width=summary["width"],
        num_actions=4,
        action_emb_dim=16,
        encoder_feat_dim=128,
        gru_hidden_dim=256,
        latent_dim=64,
        num_row_classes=summary["num_row_classes"],
        num_col_classes=summary["num_col_classes"],
        num_heading_classes=summary["num_heading_classes"],
        decoder_base_channels=128,
    )

    model = WorldModelV1(model_cfg).to(device)

    if cfg.use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = None
    if cfg.use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    os.makedirs(cfg.save_dir, exist_ok=True)
    best_val_loss = math.inf

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\nEpoch {epoch:03d}/{cfg.num_epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            train=True,
            scaler=scaler,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            cfg=cfg,
            train=False,
            scaler=None,
        )

        print(
            f"Train | "
            f"loss={train_metrics['loss_total']:.4f} | "
            f"recon={train_metrics['loss_recon']:.4f} | "
            f"pose={train_metrics['loss_pose']:.4f} | "
            f"roll_lat={train_metrics['loss_roll_latent']:.4f} | "
            f"roll_pose={train_metrics['loss_roll_pose']:.4f} | "
            f"roll_recon={train_metrics['loss_roll_recon']:.4f}"
        )

        print(
            f"Val   | "
            f"loss={val_metrics['loss_total']:.4f} | "
            f"recon={val_metrics['loss_recon']:.4f} | "
            f"pose={val_metrics['loss_pose']:.4f} | "
            f"roll_lat={val_metrics['loss_roll_latent']:.4f} | "
            f"roll_pose={val_metrics['loss_roll_pose']:.4f} | "
            f"roll_recon={val_metrics['loss_roll_recon']:.4f}"
        )

        # Always save latest only
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            save_path=os.path.join(cfg.save_dir, "last_model.pt"),
            extra={"best_val_loss": best_val_loss},
        )

        # Save best only
        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                save_path=os.path.join(cfg.save_dir, "best_model.pt"),
                extra={"best_val_loss": best_val_loss},
            )
            print(f"  Saved new best model with val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
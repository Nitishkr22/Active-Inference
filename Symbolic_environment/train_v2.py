from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import json
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from dataset_loader import (
    LoaderConfig,
    build_dataloaders,
)
from model_v2 import ModelV2Config, WorldModelV2


# ============================================================
# Config
# ============================================================

@dataclass
class TrainV2Config:
    # Data
    dataset_path: str = "../../dataset/train_dataset_v7.npz"
    val_fraction: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 30
    grad_clip_norm: float = 1.0

    # Rollout training
    rollout_start_t_min: int = 4
    rollout_horizon: int = 5

    # Loss weights
    w_recon: float = 1.0
    w_pose: float = 1.2 # 0.8
    w_roll_latent: float = 1.0
    w_roll_pose: float = 1.5 # 1.0
    w_roll_recon: float = 1.0
    w_collision: float = 0.5
    w_roll_collision: float = 0.75
    w_roll_collision_stay: float = 0.5

    # Runtime
    print_every: int = 50
    save_dir: str = "./checkpoints_v2"
    save_every_epoch: bool = False
    use_amp: bool = True
    use_channels_last: bool = False
    deterministic: bool = False
    seed: int = 42


# ============================================================
# Reproducibility / device helpers
# ============================================================

def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Loss helpers
# ============================================================

def compute_pose_loss(
    row_logits: torch.Tensor,
    col_logits: torch.Tensor,
    heading_logits: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    headings: torch.Tensor,
) -> torch.Tensor:
    """
    Inputs are [B,T,*].
    """
    B, T, _ = row_logits.shape

    loss_row = F.cross_entropy(row_logits.reshape(B * T, -1), rows.reshape(B * T))
    loss_col = F.cross_entropy(col_logits.reshape(B * T, -1), cols.reshape(B * T))
    loss_heading = F.cross_entropy(
        heading_logits.reshape(B * T, -1),
        headings.reshape(B * T),
    )
    return loss_row + loss_col + loss_heading



def compute_recon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)



def compute_collision_loss(
    collision_logits: torch.Tensor,   # [B,T,2] or [B,K,2]
    collision_targets: torch.Tensor,  # [B,T] or [B,K]
) -> torch.Tensor:
    B, T, C = collision_logits.shape
    assert C == 2
    return F.cross_entropy(
        collision_logits.reshape(B * T, C),
        collision_targets.reshape(B * T).long(),
    )



def masked_latent_stay_loss(
    z_pred_next: torch.Tensor,      # [B,K,Z]
    z_should_stay: torch.Tensor,    # [B,K,Z]
    collision_targets: torch.Tensor # [B,K], 0/1
) -> torch.Tensor:
    mask = collision_targets.float()
    if mask.sum().item() < 1:
        return z_pred_next.new_tensor(0.0)

    per_step_mse = ((z_pred_next - z_should_stay) ** 2).mean(dim=-1)
    return (per_step_mse * mask).sum() / mask.sum()


# ============================================================
# Rollout target sampler
# ============================================================

def sample_rollout_start_and_targets(
    observations: torch.Tensor,
    z_seq: torch.Tensor,
    h_seq: torch.Tensor,
    actions: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    headings: torch.Tensor,
    collisions: torch.Tensor,
    rollout_start_t_min: int,
    rollout_horizon: int,
) -> Dict[str, torch.Tensor]:
    """
    For each batch item, choose a rollout start time t and build K-step targets.

    Start state at time t:
      z_start = z_seq[:, t]
      h_start = h_seq[:, t]

    Future targets correspond to states at times t+1 ... t+K.
    Actions correspond to a_t ... a_{t+K-1}.
    """
    B, T, _ = z_seq.shape
    K = rollout_horizon

    max_valid_t = T - K - 1
    if max_valid_t < rollout_start_t_min:
        raise ValueError(
            f"rollout_start_t_min={rollout_start_t_min} too large for T={T}, K={K}."
        )

    z_start_list: List[torch.Tensor] = []
    h_start_list: List[torch.Tensor] = []
    action_roll_list: List[torch.Tensor] = []
    z_target_list: List[torch.Tensor] = []
    obs_target_list: List[torch.Tensor] = []
    row_target_list: List[torch.Tensor] = []
    col_target_list: List[torch.Tensor] = []
    heading_target_list: List[torch.Tensor] = []
    collision_target_list: List[torch.Tensor] = []
    start_index_list: List[int] = []

    for b in range(B):
        t = random.randint(rollout_start_t_min, max_valid_t)
        start_index_list.append(t)

        z_start_list.append(z_seq[b, t])
        h_start_list.append(h_seq[b, t])
        action_roll_list.append(actions[b, t:t + K])

        z_target_list.append(z_seq[b, t + 1:t + 1 + K])
        obs_target_list.append(observations[b, t + 1:t + 1 + K])
        row_target_list.append(rows[b, t + 1:t + 1 + K])
        col_target_list.append(cols[b, t + 1:t + 1 + K])
        heading_target_list.append(headings[b, t + 1:t + 1 + K])
        collision_target_list.append(collisions[b, t:t + K])

    return {
        "z_start": torch.stack(z_start_list, dim=0),
        "h_start": torch.stack(h_start_list, dim=0),
        "action_roll": torch.stack(action_roll_list, dim=0),
        "z_target": torch.stack(z_target_list, dim=0),
        "obs_target": torch.stack(obs_target_list, dim=0),
        "row_target": torch.stack(row_target_list, dim=0),
        "col_target": torch.stack(col_target_list, dim=0),
        "heading_target": torch.stack(heading_target_list, dim=0),
        "collision_target": torch.stack(collision_target_list, dim=0),
        "start_index": torch.tensor(start_index_list, dtype=torch.long, device=z_seq.device),
    }


# ============================================================
# Batch loss computation
# ============================================================

def compute_batch_losses(
    model: WorldModelV2,
    batch: Dict[str, torch.Tensor],
    cfg: TrainV2Config,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    observations = batch["observations"]           # [B,T,1,H,W]
    actions = batch["actions"]                     # [B,T-1]
    rows = batch["rows"]                           # [B,T]
    cols = batch["cols"]                           # [B,T]
    headings = batch["headings"]                   # [B,T]
    collisions = batch["collisions"].long()        # [B,T-1]

    filt = model.forward_filter(observations, actions)

    z_seq = filt["z_seq"]
    h_seq = filt["h_seq"]
    recon_seq = filt["recon_seq"]
    row_logits_seq = filt["row_logits_seq"]
    col_logits_seq = filt["col_logits_seq"]
    heading_logits_seq = filt["heading_logits_seq"]
    collision_logits_seq = filt["collision_logits_seq"]

    # Filtering losses
    recon_loss = compute_recon_loss(recon_seq, observations)
    pose_loss = compute_pose_loss(
        row_logits=row_logits_seq,
        col_logits=col_logits_seq,
        heading_logits=heading_logits_seq,
        rows=rows,
        cols=cols,
        headings=headings,
    )
    collision_loss = compute_collision_loss(
        collision_logits=collision_logits_seq,
        collision_targets=collisions,
    )

    # Rollout targets
    rollout_targets = sample_rollout_start_and_targets(
        observations=observations,
        z_seq=z_seq,
        h_seq=h_seq,
        actions=actions,
        rows=rows,
        cols=cols,
        headings=headings,
        collisions=collisions,
        rollout_start_t_min=cfg.rollout_start_t_min,
        rollout_horizon=cfg.rollout_horizon,
    )

    z_start = rollout_targets["z_start"]
    h_start = rollout_targets["h_start"]
    action_roll = rollout_targets["action_roll"]
    z_target = rollout_targets["z_target"]
    obs_target = rollout_targets["obs_target"]
    row_target = rollout_targets["row_target"]
    col_target = rollout_targets["col_target"]
    heading_target = rollout_targets["heading_target"]
    collision_target = rollout_targets["collision_target"].long()

    roll = model.rollout_from_filtered_state(
        z_start=z_start,
        h_start=h_start,
        action_seq=action_roll,
    )

    z_roll = roll["z_roll"]
    recon_roll = roll["recon_roll"]
    row_logits_roll = roll["row_logits_roll"]
    col_logits_roll = roll["col_logits_roll"]
    heading_logits_roll = roll["heading_logits_roll"]
    collision_logits_roll = roll["collision_logits_roll"]

    roll_latent_loss = F.mse_loss(z_roll, z_target)
    roll_recon_loss = compute_recon_loss(recon_roll, obs_target)
    roll_pose_loss = compute_pose_loss(
        row_logits=row_logits_roll,
        col_logits=col_logits_roll,
        heading_logits=heading_logits_roll,
        rows=row_target,
        cols=col_target,
        headings=heading_target,
    )
    roll_collision_loss = compute_collision_loss(
        collision_logits=collision_logits_roll,
        collision_targets=collision_target,
    )

    z_start_expanded = z_start.unsqueeze(1)  # [B,1,Z]
    z_prev_true = torch.cat([z_start_expanded, z_target[:, :-1, :]], dim=1)  # [B,K,Z]
    roll_collision_stay_loss = masked_latent_stay_loss(
        z_pred_next=z_roll,
        z_should_stay=z_prev_true,
        collision_targets=collision_target,
    )

    total_loss = (
        cfg.w_recon * recon_loss +
        cfg.w_pose * pose_loss +
        cfg.w_roll_latent * roll_latent_loss +
        cfg.w_roll_pose * roll_pose_loss +
        cfg.w_roll_recon * roll_recon_loss +
        cfg.w_collision * collision_loss +
        cfg.w_roll_collision * roll_collision_loss +
        cfg.w_roll_collision_stay * roll_collision_stay_loss
    )

    metrics = {
        "loss_total": float(total_loss.detach().item()),
        "loss_recon": float(recon_loss.detach().item()),
        "loss_pose": float(pose_loss.detach().item()),
        "loss_roll_latent": float(roll_latent_loss.detach().item()),
        "loss_roll_pose": float(roll_pose_loss.detach().item()),
        "loss_roll_recon": float(roll_recon_loss.detach().item()),
        "loss_collision": float(collision_loss.detach().item()),
        "loss_roll_collision": float(roll_collision_loss.detach().item()),
        "loss_roll_collision_stay": float(roll_collision_stay_loss.detach().item()),
    }
    return total_loss, metrics


# ============================================================
# Epoch runner
# ============================================================

def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved



def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = metrics_list[0].keys()
    out = {}
    for k in keys:
        out[k] = float(sum(m[k] for m in metrics_list) / len(metrics_list))
    return out



def run_epoch(
    model: WorldModelV2,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    cfg: TrainV2Config,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    all_metrics: List[Dict[str, float]] = []

    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(cfg.use_amp and device.type == "cuda"),
        ):
            loss, metrics = compute_batch_losses(model, batch, cfg)

        if train:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

        all_metrics.append(metrics)

        if train and ((batch_idx + 1) % cfg.print_every == 0):
            print(
                f"  [batch {batch_idx + 1:4d}] "
                f"loss={metrics['loss_total']:.4f} | "
                f"recon={metrics['loss_recon']:.4f} | "
                f"pose={metrics['loss_pose']:.4f} | "
                f"roll_lat={metrics['loss_roll_latent']:.4f} | "
                f"roll_pose={metrics['loss_roll_pose']:.4f} | "
                f"roll_recon={metrics['loss_roll_recon']:.4f} | "
                f"coll={metrics['loss_collision']:.4f} | "
                f"roll_coll={metrics['loss_roll_collision']:.4f} | "
                f"roll_stay={metrics['loss_roll_collision_stay']:.4f}"
            )

    return average_metrics(all_metrics)


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(
    save_path: Path,
    model: WorldModelV2,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: TrainV2Config,
    model_cfg: ModelV2Config,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_cfg": asdict(cfg),
        "model_cfg": asdict(model_cfg),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }
    torch.save(payload, save_path)


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = TrainV2Config()

    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()
    print(f"Using device: {device}")

    loader_cfg = LoaderConfig(
        dataset_path=cfg.dataset_path,
        val_fraction=cfg.val_fraction,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle_train=cfg.shuffle_train,
        seed=cfg.seed,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
    )

    full_dataset, train_loader, val_loader = build_dataloaders(loader_cfg)
    dataset_summary = full_dataset.summary()

    print("Dataset summary:")
    for k, v in dataset_summary.items():
        print(f"  {k}: {v}")

    model_cfg = ModelV2Config(
        obs_channels=dataset_summary["channels"],
        obs_height=dataset_summary["height"],
        obs_width=dataset_summary["width"],
        num_actions=4,
        num_row_classes=dataset_summary["num_row_classes"],
        num_col_classes=dataset_summary["num_col_classes"],
        num_heading_classes=dataset_summary["num_heading_classes"],
        action_emb_dim=16,
        encoder_feat_dim=128,
        gru_hidden_dim=256,
        attn_num_heads=4,
        attn_num_layers=2,
        attn_ff_dim=512,
        attn_dropout=0.1,
        latent_dim=128,
        decoder_base_channels=128,
        collision_head_hidden_dim=128,
        max_seq_len=max(128, dataset_summary["seq_len"] + 8),
    )

    model = WorldModelV2(model_cfg).to(device)
    if cfg.use_channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda") if (cfg.use_amp and device.type == "cuda") else None

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(save_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(model_cfg), f, indent=2)

    best_val_loss = math.inf

    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\nEpoch {epoch:03d}/{cfg.num_epochs}")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            train=True,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            cfg=cfg,
            train=False,
        )

        print(
            f"Train | "
            f"loss={train_metrics['loss_total']:.4f} | "
            f"recon={train_metrics['loss_recon']:.4f} | "
            f"pose={train_metrics['loss_pose']:.4f} | "
            f"roll_lat={train_metrics['loss_roll_latent']:.4f} | "
            f"roll_pose={train_metrics['loss_roll_pose']:.4f} | "
            f"roll_recon={train_metrics['loss_roll_recon']:.4f} | "
            f"coll={train_metrics['loss_collision']:.4f} | "
            f"roll_coll={train_metrics['loss_roll_collision']:.4f} | "
            f"roll_stay={train_metrics['loss_roll_collision_stay']:.4f}"
        )
        print(
            f"Val   | "
            f"loss={val_metrics['loss_total']:.4f} | "
            f"recon={val_metrics['loss_recon']:.4f} | "
            f"pose={val_metrics['loss_pose']:.4f} | "
            f"roll_lat={val_metrics['loss_roll_latent']:.4f} | "
            f"roll_pose={val_metrics['loss_roll_pose']:.4f} | "
            f"roll_recon={val_metrics['loss_roll_recon']:.4f} | "
            f"coll={val_metrics['loss_collision']:.4f} | "
            f"roll_coll={val_metrics['loss_roll_collision']:.4f} | "
            f"roll_stay={val_metrics['loss_roll_collision_stay']:.4f}"
        )

        if cfg.save_every_epoch:
            save_checkpoint(
                save_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch,
                cfg,
                model_cfg,
                train_metrics,
                val_metrics,
            )

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            save_checkpoint(
                save_dir / "best_model.pt",
                model,
                optimizer,
                epoch,
                cfg,
                model_cfg,
                train_metrics,
                val_metrics,
            )
            print(f"  Saved new best model with val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

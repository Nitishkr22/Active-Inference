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
from torch.utils.data import DataLoader

from dataset_loader import LoaderConfig, build_dataloaders
from model_v3 import ModelV3Config, WorldModelV3


# ============================================================
# Config
# ============================================================

@dataclass
class TrainV3Config:
    # Data
    dataset_path: str = "../../dataset/train_dataset_v7.npz"
    val_fraction: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last_train: bool = False
    drop_last_val: bool = False

    # Optimization
    lr: float = 5e-4
    weight_decay: float = 1e-5
    num_epochs: int = 40
    grad_clip_norm: float = 1.0

    # Base rollout settings
    rollout_start_t_min: int = 4
    rollout_horizon: int = 5

    # Filtering loss weights
    w_recon: float = 0.5
    w_pose: float = 2.0
    w_collision: float = 0.3

    # Rollout loss weights
    w_roll_pose: float = 1.0
    w_roll_recon: float = 0.5
    w_roll_collision: float = 0.5
    w_roll_stay: float = 0.5
    w_context_stability: float = 0.1

    # Entropy regularization
    # Small positive value encourages sharper beliefs.
    # Keep small; too large can make model confidently wrong.
    w_entropy: float = 0.005

    # Curriculum
    use_curriculum: bool = True

    # Runtime
    print_every: int = 50
    save_dir: str = "./checkpoints_v32"
    save_every_epoch: bool = False
    use_amp: bool = True
    deterministic: bool = False
    seed: int = 42


# ============================================================
# Curriculum schedule
# ============================================================

@dataclass
class CurriculumState:
    rollout_start_t_min: int
    rollout_horizon: int

    w_roll_pose: float
    w_roll_recon: float
    w_roll_collision: float
    w_roll_stay: float
    w_context_stability: float


def get_curriculum_state(epoch: int, cfg: TrainV3Config) -> CurriculumState:
    """
    Stage 1: learn clean belief first, short rollout.
    Stage 2: medium rollout.
    Stage 3: long rollout fine-tuning.
    """
    if not cfg.use_curriculum:
        return CurriculumState(
            rollout_start_t_min=cfg.rollout_start_t_min,
            rollout_horizon=cfg.rollout_horizon,
            w_roll_pose=cfg.w_roll_pose,
            w_roll_recon=cfg.w_roll_recon,
            w_roll_collision=cfg.w_roll_collision,
            w_roll_stay=cfg.w_roll_stay,
            w_context_stability=cfg.w_context_stability,
        )

    if epoch <= 10:
        return CurriculumState(
            rollout_start_t_min=4,
            rollout_horizon=3,
            w_roll_pose=0.5,
            w_roll_recon=0.25,
            w_roll_collision=0.25,
            w_roll_stay=0.25,
            w_context_stability=0.05,
        )

    if epoch <= 25:
        return CurriculumState(
            rollout_start_t_min=6,
            rollout_horizon=5,
            w_roll_pose=1.0,
            w_roll_recon=0.5,
            w_roll_collision=0.5,
            w_roll_stay=0.5,
            w_context_stability=0.1,
        )

    return CurriculumState(
        rollout_start_t_min=8,
        rollout_horizon=10,
        w_roll_pose=1.5,
        w_roll_recon=0.75,
        w_roll_collision=0.75,
        w_roll_stay=0.75,
        w_context_stability=0.1,
    )


# ============================================================
# Reproducibility / device
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
    B, T, _ = row_logits.shape

    row_loss = F.cross_entropy(
        row_logits.reshape(B * T, -1),
        rows.reshape(B * T).long(),
    )
    col_loss = F.cross_entropy(
        col_logits.reshape(B * T, -1),
        cols.reshape(B * T).long(),
    )
    heading_loss = F.cross_entropy(
        heading_logits.reshape(B * T, -1),
        headings.reshape(B * T).long(),
    )

    return row_loss + col_loss + heading_loss


def compute_collision_loss(
    collision_logits: torch.Tensor,
    collision_targets: torch.Tensor,
) -> torch.Tensor:
    B, T, C = collision_logits.shape
    assert C == 2

    return F.cross_entropy(
        collision_logits.reshape(B * T, C),
        collision_targets.reshape(B * T).long(),
    )


def compute_recon_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def compute_belief_entropy_loss(
    row_probs: torch.Tensor,
    col_probs: torch.Tensor,
    heading_probs: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-8

    row_entropy = -(row_probs * torch.log(row_probs.clamp_min(eps))).sum(dim=-1).mean()
    col_entropy = -(col_probs * torch.log(col_probs.clamp_min(eps))).sum(dim=-1).mean()
    heading_entropy = -(heading_probs * torch.log(heading_probs.clamp_min(eps))).sum(dim=-1).mean()

    return row_entropy + col_entropy + heading_entropy


def compute_factor_stay_loss(
    row_probs_pred: torch.Tensor,
    col_probs_pred: torch.Tensor,
    heading_probs_pred: torch.Tensor,
    row_probs_prev_true: torch.Tensor,
    col_probs_prev_true: torch.Tensor,
    heading_probs_prev_true: torch.Tensor,
    context_pred: torch.Tensor,
    context_prev_true: torch.Tensor,
    collision_targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    If collision occurs, next predicted belief should stay close to previous belief.
    """
    mask = collision_targets.float()

    if mask.sum().item() < 1:
        zero = row_probs_pred.new_tensor(0.0)
        return zero, zero

    eps = 1e-8

    row_kl = F.kl_div(
        torch.log(row_probs_pred.clamp_min(eps)),
        row_probs_prev_true,
        reduction="none",
    ).sum(dim=-1)

    col_kl = F.kl_div(
        torch.log(col_probs_pred.clamp_min(eps)),
        col_probs_prev_true,
        reduction="none",
    ).sum(dim=-1)

    heading_kl = F.kl_div(
        torch.log(heading_probs_pred.clamp_min(eps)),
        heading_probs_prev_true,
        reduction="none",
    ).sum(dim=-1)

    factor_loss = row_kl + col_kl + heading_kl
    factor_loss = (factor_loss * mask).sum() / mask.sum()

    context_mse = ((context_pred - context_prev_true) ** 2).mean(dim=-1)
    context_mse = (context_mse * mask).sum() / mask.sum()

    return factor_loss, context_mse


# ============================================================
# Metrics
# ============================================================

def compute_pose_metrics(
    row_logits: torch.Tensor,
    col_logits: torch.Tensor,
    heading_logits: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    headings: torch.Tensor,
    prefix: str,
) -> Dict[str, float]:
    row_pred = row_logits.argmax(dim=-1)
    col_pred = col_logits.argmax(dim=-1)
    heading_pred = heading_logits.argmax(dim=-1)

    row_ok = row_pred.eq(rows)
    col_ok = col_pred.eq(cols)
    heading_ok = heading_pred.eq(headings)
    joint_ok = row_ok & col_ok & heading_ok

    return {
        f"{prefix}_row_acc": float(row_ok.float().mean().detach().item()),
        f"{prefix}_col_acc": float(col_ok.float().mean().detach().item()),
        f"{prefix}_heading_acc": float(heading_ok.float().mean().detach().item()),
        f"{prefix}_joint_acc": float(joint_ok.float().mean().detach().item()),
    }


def compute_entropy_metrics(
    row_probs: torch.Tensor,
    col_probs: torch.Tensor,
    heading_probs: torch.Tensor,
    prefix: str,
) -> Dict[str, float]:
    eps = 1e-8

    row_entropy = -(row_probs * torch.log(row_probs.clamp_min(eps))).sum(dim=-1).mean()
    col_entropy = -(col_probs * torch.log(col_probs.clamp_min(eps))).sum(dim=-1).mean()
    heading_entropy = -(heading_probs * torch.log(heading_probs.clamp_min(eps))).sum(dim=-1).mean()

    return {
        f"{prefix}_row_entropy": float(row_entropy.detach().item()),
        f"{prefix}_col_entropy": float(col_entropy.detach().item()),
        f"{prefix}_heading_entropy": float(heading_entropy.detach().item()),
    }


# ============================================================
# Rollout target builder
# ============================================================

def sample_rollout_targets(
    row_probs_seq: torch.Tensor,
    col_probs_seq: torch.Tensor,
    heading_probs_seq: torch.Tensor,
    context_seq: torch.Tensor,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    headings: torch.Tensor,
    collisions: torch.Tensor,
    rollout_start_t_min: int,
    rollout_horizon: int,
) -> Dict[str, torch.Tensor]:
    B, T, _ = row_probs_seq.shape
    K = rollout_horizon

    max_valid_t = T - K - 1

    if max_valid_t < rollout_start_t_min:
        raise ValueError(
            f"rollout_start_t_min={rollout_start_t_min} too large for T={T}, K={K}. "
            f"Need rollout_start_t_min <= {max_valid_t}."
        )

    row_probs_start_list = []
    col_probs_start_list = []
    heading_probs_start_list = []
    context_start_list = []
    action_roll_list = []

    row_target_list = []
    col_target_list = []
    heading_target_list = []
    obs_target_list = []
    collision_target_list = []

    row_probs_target_list = []
    col_probs_target_list = []
    heading_probs_target_list = []
    context_target_list = []

    start_index_list = []

    for b in range(B):
        t = random.randint(rollout_start_t_min, max_valid_t)
        start_index_list.append(t)

        row_probs_start_list.append(row_probs_seq[b, t])
        col_probs_start_list.append(col_probs_seq[b, t])
        heading_probs_start_list.append(heading_probs_seq[b, t])
        context_start_list.append(context_seq[b, t])

        action_roll_list.append(actions[b, t:t + K])

        row_target_list.append(rows[b, t + 1:t + 1 + K])
        col_target_list.append(cols[b, t + 1:t + 1 + K])
        heading_target_list.append(headings[b, t + 1:t + 1 + K])
        obs_target_list.append(observations[b, t + 1:t + 1 + K])
        collision_target_list.append(collisions[b, t:t + K])

        row_probs_target_list.append(row_probs_seq[b, t + 1:t + 1 + K])
        col_probs_target_list.append(col_probs_seq[b, t + 1:t + 1 + K])
        heading_probs_target_list.append(heading_probs_seq[b, t + 1:t + 1 + K])
        context_target_list.append(context_seq[b, t + 1:t + 1 + K])

    return {
        "row_probs_start": torch.stack(row_probs_start_list, dim=0),
        "col_probs_start": torch.stack(col_probs_start_list, dim=0),
        "heading_probs_start": torch.stack(heading_probs_start_list, dim=0),
        "context_start": torch.stack(context_start_list, dim=0),

        "action_roll": torch.stack(action_roll_list, dim=0),

        "row_target": torch.stack(row_target_list, dim=0),
        "col_target": torch.stack(col_target_list, dim=0),
        "heading_target": torch.stack(heading_target_list, dim=0),
        "obs_target": torch.stack(obs_target_list, dim=0),
        "collision_target": torch.stack(collision_target_list, dim=0),

        "row_probs_target": torch.stack(row_probs_target_list, dim=0),
        "col_probs_target": torch.stack(col_probs_target_list, dim=0),
        "heading_probs_target": torch.stack(heading_probs_target_list, dim=0),
        "context_target": torch.stack(context_target_list, dim=0),

        "start_index": torch.tensor(
            start_index_list,
            dtype=torch.long,
            device=row_probs_seq.device,
        ),
    }


# ============================================================
# Batch loss computation
# ============================================================

def compute_batch_losses(
    model: WorldModelV3,
    batch: Dict[str, torch.Tensor],
    cfg: TrainV3Config,
    curriculum: CurriculumState,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    observations = batch["observations"]
    actions = batch["actions"]
    rows = batch["rows"].long()
    cols = batch["cols"].long()
    headings = batch["headings"].long()
    collisions = batch["collisions"].long()

    filt = model.forward_filter(observations, actions)

    row_logits_seq = filt["row_logits_seq"]
    col_logits_seq = filt["col_logits_seq"]
    heading_logits_seq = filt["heading_logits_seq"]

    row_probs_seq = filt["row_probs_seq"]
    col_probs_seq = filt["col_probs_seq"]
    heading_probs_seq = filt["heading_probs_seq"]

    context_seq = filt["context_seq"]
    recon_seq = filt["recon_seq"]
    collision_logits_seq = filt["collision_logits_seq"]

    # -----------------------------
    # Filtering losses
    # -----------------------------
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
        collision_logits_seq,
        collisions,
    )

    entropy_loss = compute_belief_entropy_loss(
        row_probs_seq,
        col_probs_seq,
        heading_probs_seq,
    )

    # -----------------------------
    # Rollout targets
    # -----------------------------
    rollout_targets = sample_rollout_targets(
        row_probs_seq=row_probs_seq,
        col_probs_seq=col_probs_seq,
        heading_probs_seq=heading_probs_seq,
        context_seq=context_seq,
        observations=observations,
        actions=actions,
        rows=rows,
        cols=cols,
        headings=headings,
        collisions=collisions,
        rollout_start_t_min=curriculum.rollout_start_t_min,
        rollout_horizon=curriculum.rollout_horizon,
    )

    roll = model.rollout_from_filtered_state(
        row_probs_start=rollout_targets["row_probs_start"],
        col_probs_start=rollout_targets["col_probs_start"],
        heading_probs_start=rollout_targets["heading_probs_start"],
        context_start=rollout_targets["context_start"],
        action_seq=rollout_targets["action_roll"],
    )

    # -----------------------------
    # Rollout losses
    # -----------------------------
    roll_pose_loss = compute_pose_loss(
        row_logits=roll["row_logits_roll"],
        col_logits=roll["col_logits_roll"],
        heading_logits=roll["heading_logits_roll"],
        rows=rollout_targets["row_target"],
        cols=rollout_targets["col_target"],
        headings=rollout_targets["heading_target"],
    )

    roll_recon_loss = compute_recon_loss(
        roll["recon_roll"],
        rollout_targets["obs_target"],
    )

    roll_collision_loss = compute_collision_loss(
        roll["collision_logits_roll"],
        rollout_targets["collision_target"],
    )

    row_probs_prev_true = torch.cat(
        [
            rollout_targets["row_probs_start"].unsqueeze(1),
            rollout_targets["row_probs_target"][:, :-1, :],
        ],
        dim=1,
    )

    col_probs_prev_true = torch.cat(
        [
            rollout_targets["col_probs_start"].unsqueeze(1),
            rollout_targets["col_probs_target"][:, :-1, :],
        ],
        dim=1,
    )

    heading_probs_prev_true = torch.cat(
        [
            rollout_targets["heading_probs_start"].unsqueeze(1),
            rollout_targets["heading_probs_target"][:, :-1, :],
        ],
        dim=1,
    )

    context_prev_true = torch.cat(
        [
            rollout_targets["context_start"].unsqueeze(1),
            rollout_targets["context_target"][:, :-1, :],
        ],
        dim=1,
    )

    roll_stay_loss, context_stability_loss = compute_factor_stay_loss(
        row_probs_pred=roll["row_probs_roll"],
        col_probs_pred=roll["col_probs_roll"],
        heading_probs_pred=roll["heading_probs_roll"],
        row_probs_prev_true=row_probs_prev_true,
        col_probs_prev_true=col_probs_prev_true,
        heading_probs_prev_true=heading_probs_prev_true,
        context_pred=roll["context_roll"],
        context_prev_true=context_prev_true,
        collision_targets=rollout_targets["collision_target"],
    )

    # -----------------------------
    # Total loss
    # -----------------------------
    total_loss = (
        cfg.w_recon * recon_loss
        + cfg.w_pose * pose_loss
        + cfg.w_collision * collision_loss
        + curriculum.w_roll_pose * roll_pose_loss
        + curriculum.w_roll_recon * roll_recon_loss
        + curriculum.w_roll_collision * roll_collision_loss
        + curriculum.w_roll_stay * roll_stay_loss
        + curriculum.w_context_stability * context_stability_loss
        + cfg.w_entropy * entropy_loss
    )

    metrics: Dict[str, float] = {
        "loss_total": float(total_loss.detach().item()),
        "loss_recon": float(recon_loss.detach().item()),
        "loss_pose": float(pose_loss.detach().item()),
        "loss_collision": float(collision_loss.detach().item()),
        "loss_entropy": float(entropy_loss.detach().item()),

        "loss_roll_pose": float(roll_pose_loss.detach().item()),
        "loss_roll_recon": float(roll_recon_loss.detach().item()),
        "loss_roll_collision": float(roll_collision_loss.detach().item()),
        "loss_roll_stay": float(roll_stay_loss.detach().item()),
        "loss_context_stability": float(context_stability_loss.detach().item()),

        "rollout_start_t_min": float(curriculum.rollout_start_t_min),
        "rollout_horizon": float(curriculum.rollout_horizon),
    }

    metrics.update(
        compute_pose_metrics(
            row_logits_seq,
            col_logits_seq,
            heading_logits_seq,
            rows,
            cols,
            headings,
            prefix="filter",
        )
    )

    metrics.update(
        compute_pose_metrics(
            roll["row_logits_roll"],
            roll["col_logits_roll"],
            roll["heading_logits_roll"],
            rollout_targets["row_target"],
            rollout_targets["col_target"],
            rollout_targets["heading_target"],
            prefix="roll",
        )
    )

    metrics.update(
        compute_entropy_metrics(
            row_probs_seq,
            col_probs_seq,
            heading_probs_seq,
            prefix="filter",
        )
    )

    return total_loss, metrics


# ============================================================
# Epoch runner
# ============================================================

def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    out = {}

    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v

    return out


def average_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = metrics_list[0].keys()
    return {
        k: float(sum(m[k] for m in metrics_list) / len(metrics_list))
        for k in keys
    }


def run_epoch(
    model: WorldModelV3,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    cfg: TrainV3Config,
    curriculum: CurriculumState,
    train: bool,
) -> Dict[str, float]:
    model.train(train)
    all_metrics: List[Dict[str, float]] = []

    grad_context = torch.enable_grad() if train else torch.no_grad()

    with grad_context:
        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(cfg.use_amp and device.type == "cuda"),
            ):
                loss, metrics = compute_batch_losses(
                    model=model,
                    batch=batch,
                    cfg=cfg,
                    curriculum=curriculum,
                )

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.grad_clip_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg.grad_clip_norm,
                    )
                    optimizer.step()

            all_metrics.append(metrics)

            if train and ((batch_idx + 1) % cfg.print_every == 0):
                print(
                    f"  [batch {batch_idx + 1:4d}] "
                    f"loss={metrics['loss_total']:.4f} | "
                    f"pose={metrics['loss_pose']:.4f} | "
                    f"joint={metrics['filter_joint_acc']:.3f} | "
                    f"roll_pose={metrics['loss_roll_pose']:.4f} | "
                    f"roll_joint={metrics['roll_joint_acc']:.3f} | "
                    f"recon={metrics['loss_recon']:.4f} | "
                    f"coll={metrics['loss_collision']:.4f} | "
                    f"roll_coll={metrics['loss_roll_collision']:.4f} | "
                    f"stay={metrics['loss_roll_stay']:.4f} | "
                    f"Hr={metrics['filter_row_entropy']:.3f} | "
                    f"Hc={metrics['filter_col_entropy']:.3f} | "
                    f"Hh={metrics['filter_heading_entropy']:.3f}"
                )

    return average_metrics(all_metrics)


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(
    save_path: Path,
    model: WorldModelV3,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: TrainV3Config,
    model_cfg: ModelV3Config,
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


def print_epoch_summary(
    name: str,
    metrics: Dict[str, float],
) -> None:
    print(
        f"{name:<5} | "
        f"loss={metrics['loss_total']:.4f} | "
        f"pose={metrics['loss_pose']:.4f} | "
        f"joint={metrics['filter_joint_acc']:.3f} | "
        f"row={metrics['filter_row_acc']:.3f} | "
        f"col={metrics['filter_col_acc']:.3f} | "
        f"head={metrics['filter_heading_acc']:.3f} | "
        f"roll_pose={metrics['loss_roll_pose']:.4f} | "
        f"roll_joint={metrics['roll_joint_acc']:.3f} | "
        f"recon={metrics['loss_recon']:.4f} | "
        f"roll_recon={metrics['loss_roll_recon']:.4f} | "
        f"coll={metrics['loss_collision']:.4f} | "
        f"roll_coll={metrics['loss_roll_collision']:.4f} | "
        f"stay={metrics['loss_roll_stay']:.4f} | "
        f"ctx={metrics['loss_context_stability']:.4f} | "
        f"H=({metrics['filter_row_entropy']:.3f},"
        f"{metrics['filter_col_entropy']:.3f},"
        f"{metrics['filter_heading_entropy']:.3f})"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = TrainV3Config()

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
        drop_last_train=cfg.drop_last_train,
        drop_last_val=cfg.drop_last_val,
    )

    full_dataset, train_loader, val_loader = build_dataloaders(loader_cfg)
    summary = full_dataset.summary()

    print("Dataset summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    model_cfg = ModelV3Config(
        obs_channels=summary["channels"],
        obs_height=summary["height"],
        obs_width=summary["width"],

        num_actions=4,
        num_row_classes=summary["num_row_classes"],
        num_col_classes=summary["num_col_classes"],
        num_heading_classes=summary["num_heading_classes"],

        action_emb_dim=16,
        encoder_feat_dim=128,

        gru_hidden_dim=256,

        attn_num_heads=4,
        attn_num_layers=2,
        attn_ff_dim=512,
        attn_dropout=0.1,
        max_seq_len=max(128, summary["seq_len"] + 8),

        context_dim=64,
        factor_hidden_dim=128,

        transition_hidden_dim=128,
        collision_hidden_dim=128,

        decoder_hidden_dim=256,
        decoder_base_channels=128,
    )

    model = WorldModelV3(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scaler = (
        torch.amp.GradScaler("cuda")
        if (cfg.use_amp and device.type == "cuda")
        else None
    )

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    with open(save_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(model_cfg), f, indent=2)

    best_val_loss = math.inf
    best_val_joint = -math.inf

    for epoch in range(1, cfg.num_epochs + 1):
        curriculum = get_curriculum_state(epoch, cfg)

        print(f"\nEpoch {epoch:03d}/{cfg.num_epochs}")
        print(
            f"Curriculum: start_t_min={curriculum.rollout_start_t_min}, "
            f"horizon={curriculum.rollout_horizon}, "
            f"w_roll_pose={curriculum.w_roll_pose}"
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            cfg=cfg,
            curriculum=curriculum,
            train=True,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            cfg=cfg,
            curriculum=curriculum,
            train=False,
        )

        print_epoch_summary("Train", train_metrics)
        print_epoch_summary("Val", val_metrics)

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

        val_loss = val_metrics["loss_total"]
        val_joint = val_metrics["filter_joint_acc"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                save_dir / "best_model_v32.pt",
                model,
                optimizer,
                epoch,
                cfg,
                model_cfg,
                train_metrics,
                val_metrics,
            )

            print(f"  Saved new best model by val loss: {best_val_loss:.4f}")

        if val_joint > best_val_joint:
            best_val_joint = val_joint

            save_checkpoint(
                save_dir / "best_belief_model_v32.pt",
                model,
                optimizer,
                epoch,
                cfg,
                model_cfg,
                train_metrics,
                val_metrics,
            )

            print(f"  Saved new best belief model by joint acc: {best_val_joint:.4f}")


if __name__ == "__main__":
    main()
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple
import csv
import json
import math
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_loader import LoaderConfig, build_dataloaders
from model_v5 import ModelV5Config, WorldModelV5


# ============================================================
# Training logger  (identical to V4)
# ============================================================

class TrainingLogger:
    FIXED_COLS = [
        "epoch", "timestamp", "elapsed_s",
        "kl_weight",
        "curriculum_start_t_min", "curriculum_horizon", "curriculum_w_roll_pose",
    ]

    def __init__(self, save_dir: Path) -> None:
        self.save_dir = save_dir
        self.csv_path = save_dir / "training_log.csv"
        self._fieldnames: List[str] = []
        self._writer: csv.DictWriter | None = None
        self._file = None
        self._start_time: float = time.time()

    def _ensure_open(self, row: dict) -> None:
        if self._writer is not None:
            return
        self._fieldnames = self.FIXED_COLS + sorted(
            k for k in row if k not in self.FIXED_COLS
        )
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames, extrasaction="ignore")
        self._writer.writeheader()

    def log_epoch(
        self,
        epoch: int,
        kl_weight: float,
        curriculum: "CurriculumState",
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_elapsed_s: float,
    ) -> None:
        row: Dict = {
            "epoch": epoch,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_s": round(epoch_elapsed_s, 2),
            "kl_weight": round(kl_weight, 4),
            "curriculum_start_t_min": curriculum.rollout_start_t_min,
            "curriculum_horizon": curriculum.rollout_horizon,
            "curriculum_w_roll_pose": curriculum.w_roll_pose,
        }
        for k, v in train_metrics.items():
            row[f"train_{k}"] = round(float(v), 6) if isinstance(v, float) else v
        for k, v in val_metrics.items():
            row[f"val_{k}"] = round(float(v), 6) if isinstance(v, float) else v
        self._ensure_open(row)
        self._writer.writerow(row)
        self._file.flush()

    def write_best_metrics(self, epoch, reason, val_metrics, train_metrics) -> None:
        payload = {
            "epoch": epoch, "reason": reason,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "val": {k: round(float(v), 6) for k, v in val_metrics.items()},
            "train": {k: round(float(v), 6) for k, v in train_metrics.items()},
        }
        with open(self.save_dir / "best_metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def finish(self, final_epoch, best_val_loss, best_val_joint, train_metrics, val_metrics) -> None:
        total_s = time.time() - self._start_time
        summary = {
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_training_seconds": round(total_s, 1),
            "total_training_minutes": round(total_s / 60, 2),
            "final_epoch": final_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "best_val_joint_acc": round(best_val_joint, 6),
            "final_train_metrics": {k: round(float(v), 6) for k, v in train_metrics.items()},
            "final_val_metrics": {k: round(float(v), 6) for k, v in val_metrics.items()},
        }
        with open(self.save_dir / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        if self._file is not None:
            self._file.close()
        print(f"\nTraining complete. Total time: {total_s/60:.1f} min")


# ============================================================
# Config
# ============================================================

@dataclass
class TrainV5Config:
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

    # KL / uncertainty losses
    w_kl_context: float = 0.01
    w_vfe_kl: float = 0.02
    kl_warmup_epochs: int = 5

    # Observation noise injection
    obs_noise_sigma: float = 0.07

    # Predictive prior
    use_predictive_prior: bool = True

    # Curriculum
    use_curriculum: bool = True

    # Runtime
    print_every: int = 50
    save_dir: str = "./checkpoints_v5"
    save_every_epoch: bool = False
    use_amp: bool = True
    deterministic: bool = False
    seed: int = 42


# ============================================================
# Curriculum  (identical to V4)
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


def get_curriculum_state(epoch: int, cfg: TrainV5Config) -> CurriculumState:
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
        return CurriculumState(4, 3, 0.5, 0.25, 0.25, 0.25, 0.05)
    if epoch <= 25:
        return CurriculumState(6, 5, 1.0, 0.5, 0.5, 0.5, 0.1)
    return CurriculumState(8, 10, 1.5, 0.75, 0.75, 0.75, 0.1)


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
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Loss helpers  (identical to V4)
# ============================================================

def compute_pose_loss(row_logits, col_logits, heading_logits, rows, cols, headings):
    B, T, _ = row_logits.shape
    return (
        F.cross_entropy(row_logits.reshape(B * T, -1), rows.reshape(B * T).long())
        + F.cross_entropy(col_logits.reshape(B * T, -1), cols.reshape(B * T).long())
        + F.cross_entropy(heading_logits.reshape(B * T, -1), headings.reshape(B * T).long())
    )


def compute_collision_loss(collision_logits, collision_targets):
    B, T, C = collision_logits.shape
    return F.cross_entropy(collision_logits.reshape(B * T, C), collision_targets.reshape(B * T).long())


def compute_recon_loss(pred, target):
    return F.l1_loss(pred, target)


def compute_factor_stay_loss(
    row_probs_pred, col_probs_pred, heading_probs_pred,
    row_probs_prev_true, col_probs_prev_true, heading_probs_prev_true,
    context_pred, context_prev_true, collision_targets,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = collision_targets.float()
    if mask.sum().item() < 1:
        zero = row_probs_pred.new_tensor(0.0)
        return zero, zero

    eps = 1e-8
    kl_row = F.kl_div(torch.log(row_probs_pred.clamp_min(eps)), row_probs_prev_true, reduction="none").sum(dim=-1)
    kl_col = F.kl_div(torch.log(col_probs_pred.clamp_min(eps)), col_probs_prev_true, reduction="none").sum(dim=-1)
    kl_hdg = F.kl_div(torch.log(heading_probs_pred.clamp_min(eps)), heading_probs_prev_true, reduction="none").sum(dim=-1)
    factor_loss = ((kl_row + kl_col + kl_hdg) * mask).sum() / mask.sum()

    ctx_mse = (((context_pred - context_prev_true) ** 2).mean(dim=-1) * mask).sum() / mask.sum()
    return factor_loss, ctx_mse


def compute_vfe_kl_loss(
    posterior_row_probs, posterior_col_probs, posterior_heading_probs,
    prior_row_probs, prior_col_probs, prior_heading_probs,
) -> torch.Tensor:
    eps = 1e-8
    post_row = posterior_row_probs[:, 1:, :].detach()
    post_col = posterior_col_probs[:, 1:, :].detach()
    post_hdg = posterior_heading_probs[:, 1:, :].detach()
    return (
        F.kl_div(torch.log(prior_row_probs.clamp_min(eps)), post_row, reduction="batchmean")
        + F.kl_div(torch.log(prior_col_probs.clamp_min(eps)), post_col, reduction="batchmean")
        + F.kl_div(torch.log(prior_heading_probs.clamp_min(eps)), post_hdg, reduction="batchmean")
    )


# ============================================================
# Metrics  (identical to V4)
# ============================================================

def compute_pose_metrics(row_logits, col_logits, heading_logits, rows, cols, headings, prefix):
    row_ok = row_logits.argmax(dim=-1).eq(rows)
    col_ok = col_logits.argmax(dim=-1).eq(cols)
    hdg_ok = heading_logits.argmax(dim=-1).eq(headings)
    return {
        f"{prefix}_row_acc":     float(row_ok.float().mean().item()),
        f"{prefix}_col_acc":     float(col_ok.float().mean().item()),
        f"{prefix}_heading_acc": float(hdg_ok.float().mean().item()),
        f"{prefix}_joint_acc":   float((row_ok & col_ok & hdg_ok).float().mean().item()),
    }


def compute_entropy_metrics(row_probs, col_probs, heading_probs, prefix):
    eps = 1e-8
    def _h(p):
        return float(-(p * p.clamp_min(eps).log()).sum(dim=-1).mean().item())
    return {
        f"{prefix}_row_entropy":     _h(row_probs),
        f"{prefix}_col_entropy":     _h(col_probs),
        f"{prefix}_heading_entropy": _h(heading_probs),
    }


# ============================================================
# Rollout target builder  (identical to V4)
# ============================================================

def sample_rollout_targets(
    row_probs_seq, col_probs_seq, heading_probs_seq, context_seq,
    observations, actions, rows, cols, headings, collisions,
    rollout_start_t_min, rollout_horizon,
) -> Dict[str, torch.Tensor]:
    B, T, _ = row_probs_seq.shape
    K = rollout_horizon
    max_valid_t = T - K - 1
    if max_valid_t < rollout_start_t_min:
        raise ValueError(
            f"rollout_start_t_min={rollout_start_t_min} too large for T={T}, K={K}."
        )

    rps, cps, hps, cs, ars = [], [], [], [], []
    rt, ct, ht, ot, colt = [], [], [], [], []
    rpt, cpt, hpt, ctx_t, si = [], [], [], [], []

    for b in range(B):
        t = random.randint(rollout_start_t_min, max_valid_t)
        si.append(t)
        rps.append(row_probs_seq[b, t])
        cps.append(col_probs_seq[b, t])
        hps.append(heading_probs_seq[b, t])
        cs.append(context_seq[b, t])
        ars.append(actions[b, t:t + K])
        rt.append(rows[b, t + 1:t + 1 + K])
        ct.append(cols[b, t + 1:t + 1 + K])
        ht.append(headings[b, t + 1:t + 1 + K])
        ot.append(observations[b, t + 1:t + 1 + K])
        colt.append(collisions[b, t:t + K])
        rpt.append(row_probs_seq[b, t + 1:t + 1 + K])
        cpt.append(col_probs_seq[b, t + 1:t + 1 + K])
        hpt.append(heading_probs_seq[b, t + 1:t + 1 + K])
        ctx_t.append(context_seq[b, t + 1:t + 1 + K])

    return {
        "row_probs_start": torch.stack(rps),
        "col_probs_start": torch.stack(cps),
        "heading_probs_start": torch.stack(hps),
        "context_start": torch.stack(cs),
        "action_roll": torch.stack(ars),
        "row_target": torch.stack(rt),
        "col_target": torch.stack(ct),
        "heading_target": torch.stack(ht),
        "obs_target": torch.stack(ot),
        "collision_target": torch.stack(colt),
        "row_probs_target": torch.stack(rpt),
        "col_probs_target": torch.stack(cpt),
        "heading_probs_target": torch.stack(hpt),
        "context_target": torch.stack(ctx_t),
        "start_index": torch.tensor(si, dtype=torch.long, device=row_probs_seq.device),
    }


# ============================================================
# Batch loss computation
# ============================================================

def compute_batch_losses(
    model: WorldModelV5,
    batch: Dict[str, torch.Tensor],
    cfg: TrainV5Config,
    curriculum: CurriculumState,
    kl_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    observations = batch["observations"]
    actions      = batch["actions"]
    rows         = batch["rows"].long()
    cols         = batch["cols"].long()
    headings     = batch["headings"].long()
    collisions   = batch["collisions"].long()

    # forward_filter runs decoder only when model.training == True
    filt = model.forward_filter(observations, actions)

    row_logits_seq    = filt["row_logits_seq"]
    col_logits_seq    = filt["col_logits_seq"]
    heading_logits_seq = filt["heading_logits_seq"]
    row_probs_seq     = filt["row_probs_seq"]
    col_probs_seq     = filt["col_probs_seq"]
    heading_probs_seq  = filt["heading_probs_seq"]
    context_seq        = filt["context_seq"]
    recon_seq          = filt["recon_seq"]         # None in eval mode (val epoch)
    collision_logits_seq = filt["collision_logits_seq"]

    # ---- Filtering losses ----
    pose_loss      = compute_pose_loss(row_logits_seq, col_logits_seq, heading_logits_seq, rows, cols, headings)
    collision_loss = compute_collision_loss(collision_logits_seq, collisions)
    recon_loss     = compute_recon_loss(recon_seq, observations) if recon_seq is not None else row_logits_seq.new_tensor(0.0)

    # ---- V4/V5: KL losses ----
    kl_context = model.compute_context_kl(filt["context_mu_seq"], filt["context_logvar_seq"])

    if cfg.use_predictive_prior:
        kl_vfe = compute_vfe_kl_loss(
            row_probs_seq, col_probs_seq, heading_probs_seq,
            filt["transition_prior_row_probs"],
            filt["transition_prior_col_probs"],
            filt["transition_prior_heading_probs"],
        )
    else:
        kl_vfe = row_logits_seq.new_tensor(0.0)

    # ---- Rollout ----
    rollout_targets = sample_rollout_targets(
        row_probs_seq, col_probs_seq, heading_probs_seq, context_seq,
        observations, actions, rows, cols, headings, collisions,
        curriculum.rollout_start_t_min, curriculum.rollout_horizon,
    )

    roll = model.rollout_from_filtered_state(
        row_probs_start=rollout_targets["row_probs_start"],
        col_probs_start=rollout_targets["col_probs_start"],
        heading_probs_start=rollout_targets["heading_probs_start"],
        context_start=rollout_targets["context_start"],
        action_seq=rollout_targets["action_roll"],
    )

    roll_pose_loss = compute_pose_loss(
        roll["row_logits_roll"], roll["col_logits_roll"], roll["heading_logits_roll"],
        rollout_targets["row_target"], rollout_targets["col_target"], rollout_targets["heading_target"],
    )
    roll_recon_loss     = compute_recon_loss(roll["recon_roll"], rollout_targets["obs_target"])
    roll_collision_loss = compute_collision_loss(roll["collision_logits_roll"], rollout_targets["collision_target"])

    row_probs_prev_true = torch.cat([
        rollout_targets["row_probs_start"].unsqueeze(1),
        rollout_targets["row_probs_target"][:, :-1, :],
    ], dim=1)
    col_probs_prev_true = torch.cat([
        rollout_targets["col_probs_start"].unsqueeze(1),
        rollout_targets["col_probs_target"][:, :-1, :],
    ], dim=1)
    heading_probs_prev_true = torch.cat([
        rollout_targets["heading_probs_start"].unsqueeze(1),
        rollout_targets["heading_probs_target"][:, :-1, :],
    ], dim=1)
    context_prev_true = torch.cat([
        rollout_targets["context_start"].unsqueeze(1),
        rollout_targets["context_target"][:, :-1, :],
    ], dim=1)

    roll_stay_loss, context_stability_loss = compute_factor_stay_loss(
        roll["row_probs_roll"], roll["col_probs_roll"], roll["heading_probs_roll"],
        row_probs_prev_true, col_probs_prev_true, heading_probs_prev_true,
        roll["context_roll"], context_prev_true, rollout_targets["collision_target"],
    )

    # ---- Total ----
    # w_recon is zero-contribution in validation (recon_loss=0 when decoder skipped)
    total_loss = (
        cfg.w_recon       * recon_loss
        + cfg.w_pose      * pose_loss
        + cfg.w_collision * collision_loss
        + curriculum.w_roll_pose        * roll_pose_loss
        + curriculum.w_roll_recon       * roll_recon_loss
        + curriculum.w_roll_collision   * roll_collision_loss
        + curriculum.w_roll_stay        * roll_stay_loss
        + curriculum.w_context_stability * context_stability_loss
        + kl_weight * cfg.w_kl_context  * kl_context
        + kl_weight * cfg.w_vfe_kl      * kl_vfe
    )

    metrics: Dict[str, float] = {
        "loss_total":            float(total_loss.detach().item()),
        "loss_recon":            float(recon_loss.detach().item()),
        "loss_pose":             float(pose_loss.detach().item()),
        "loss_collision":        float(collision_loss.detach().item()),
        "loss_kl_context":       float(kl_context.detach().item()),
        "loss_kl_vfe":           float(kl_vfe.detach().item()),
        "loss_roll_pose":        float(roll_pose_loss.detach().item()),
        "loss_roll_recon":       float(roll_recon_loss.detach().item()),
        "loss_roll_collision":   float(roll_collision_loss.detach().item()),
        "loss_roll_stay":        float(roll_stay_loss.detach().item()),
        "loss_context_stability": float(context_stability_loss.detach().item()),
        "rollout_start_t_min":   float(curriculum.rollout_start_t_min),
        "rollout_horizon":       float(curriculum.rollout_horizon),
        "kl_weight":             float(kl_weight),
    }
    metrics.update(compute_pose_metrics(row_logits_seq, col_logits_seq, heading_logits_seq, rows, cols, headings, "filter"))
    metrics.update(compute_pose_metrics(
        roll["row_logits_roll"], roll["col_logits_roll"], roll["heading_logits_roll"],
        rollout_targets["row_target"], rollout_targets["col_target"], rollout_targets["heading_target"], "roll",
    ))
    metrics.update(compute_entropy_metrics(row_probs_seq, col_probs_seq, heading_probs_seq, "filter"))

    return total_loss, metrics


# ============================================================
# Observation noise  (identical to V4c)
# ============================================================

def add_obs_noise(obs: torch.Tensor, sigma: float) -> torch.Tensor:
    return (obs + torch.randn_like(obs) * sigma).clamp_(0.0, 1.0)


# ============================================================
# Epoch runner
# ============================================================

def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def average_metrics(metrics_list):
    keys = metrics_list[0].keys()
    return {k: float(sum(m[k] for m in metrics_list) / len(metrics_list)) for k in keys}


def run_epoch(
    model: WorldModelV5,
    loader: DataLoader,
    optimizer,
    scaler,
    device: torch.device,
    cfg: TrainV5Config,
    curriculum: CurriculumState,
    train: bool,
    kl_weight: float = 1.0,
) -> Dict[str, float]:
    model.train(train)
    all_metrics = []

    with (torch.enable_grad() if train else torch.no_grad()):
        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            if train and cfg.obs_noise_sigma > 0.0:
                batch["observations"] = add_obs_noise(batch["observations"], cfg.obs_noise_sigma)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(cfg.use_amp and device.type == "cuda")):
                loss, metrics = compute_batch_losses(model, batch, cfg, curriculum, kl_weight)

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
                    f"pose={metrics['loss_pose']:.4f} | "
                    f"joint={metrics['filter_joint_acc']:.3f} | "
                    f"roll_joint={metrics['roll_joint_acc']:.3f} | "
                    f"recon={metrics['loss_recon']:.4f} | "
                    f"kl_ctx={metrics['loss_kl_context']:.5f} | "
                    f"kl_vfe={metrics['loss_kl_vfe']:.5f} | "
                    f"H=({metrics['filter_row_entropy']:.3f},"
                    f"{metrics['filter_col_entropy']:.3f},"
                    f"{metrics['filter_heading_entropy']:.3f})"
                )

    return average_metrics(all_metrics)


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(save_path, model, optimizer, epoch, cfg, model_cfg, train_metrics, val_metrics):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_cfg": asdict(cfg),
        "model_cfg": asdict(model_cfg),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }, save_path)


def print_epoch_summary(name, metrics):
    print(
        f"{name:<5} | "
        f"loss={metrics['loss_total']:.4f} | "
        f"pose={metrics['loss_pose']:.4f} | "
        f"joint={metrics['filter_joint_acc']:.3f} | "
        f"roll_joint={metrics['roll_joint_acc']:.3f} | "
        f"recon={metrics['loss_recon']:.4f} | "
        f"kl_ctx={metrics['loss_kl_context']:.5f} | "
        f"kl_vfe={metrics['loss_kl_vfe']:.5f} | "
        f"H=({metrics['filter_row_entropy']:.3f},"
        f"{metrics['filter_col_entropy']:.3f},"
        f"{metrics['filter_heading_entropy']:.3f})"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = TrainV5Config()
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

    model_cfg = ModelV5Config(
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
        context_dim=32,
        factor_hidden_dim=128,
        transition_hidden_dim=128,
        collision_hidden_dim=128,
        decoder_hidden_dim=256,
        decoder_base_channels=128,
        use_predictive_prior=cfg.use_predictive_prior,
    )

    model = WorldModelV5(model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if (cfg.use_amp and device.type == "cuda") else None

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "train_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(save_dir / "model_config.json", "w") as f:
        json.dump(asdict(model_cfg), f, indent=2)

    logger = TrainingLogger(save_dir)
    best_val_loss  = math.inf
    best_val_joint = -math.inf
    last_train_metrics: Dict[str, float] = {}
    last_val_metrics:   Dict[str, float] = {}

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.time()
        curriculum  = get_curriculum_state(epoch, cfg)
        kl_weight   = min(1.0, epoch / max(1, cfg.kl_warmup_epochs))

        print(f"\nEpoch {epoch:03d}/{cfg.num_epochs}")
        print(
            f"Curriculum: start_t_min={curriculum.rollout_start_t_min}, "
            f"horizon={curriculum.rollout_horizon} | kl_weight={kl_weight:.3f}"
        )

        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, cfg, curriculum, train=True, kl_weight=kl_weight)
        val_metrics   = run_epoch(model, val_loader, None, None, device, cfg, curriculum, train=False, kl_weight=kl_weight)

        epoch_elapsed = time.time() - epoch_start
        print_epoch_summary("Train", train_metrics)
        print_epoch_summary("Val  ", val_metrics)
        print(f"  Epoch time: {epoch_elapsed:.1f}s")

        logger.log_epoch(epoch, kl_weight, curriculum, train_metrics, val_metrics, epoch_elapsed)
        last_train_metrics = train_metrics
        last_val_metrics   = val_metrics

        if cfg.save_every_epoch:
            save_checkpoint(save_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, cfg, model_cfg, train_metrics, val_metrics)

        val_loss  = val_metrics["loss_total"]
        val_joint = val_metrics["filter_joint_acc"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(save_dir / "best_model_v5.pt", model, optimizer, epoch, cfg, model_cfg, train_metrics, val_metrics)
            logger.write_best_metrics(epoch, "best_val_loss", val_metrics, train_metrics)
            print(f"  Saved best model (val loss {best_val_loss:.4f})")

        if val_joint > best_val_joint:
            best_val_joint = val_joint
            save_checkpoint(save_dir / "best_belief_model_v5.pt", model, optimizer, epoch, cfg, model_cfg, train_metrics, val_metrics)
            logger.write_best_metrics(epoch, "best_val_joint_acc", val_metrics, train_metrics)
            print(f"  Saved best belief model (joint acc {best_val_joint:.4f})")

    logger.finish(cfg.num_epochs, best_val_loss, best_val_joint, last_train_metrics, last_val_metrics)


if __name__ == "__main__":
    main()

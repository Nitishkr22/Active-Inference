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
from model_v4 import ModelV4Config, WorldModelV4


# ============================================================
# Training logger
# ============================================================

class TrainingLogger:
    """
    Writes one CSV row per epoch and a JSON summary at the end.

    Files written to save_dir:
      training_log.csv      — full per-epoch metrics (train + val)
      training_summary.json — written at the end of training
      best_metrics.json     — updated each time a new best checkpoint is saved
    """

    # Columns written in order.  Prefixed metrics come from the metrics dicts
    # returned by run_epoch; we write all of them with "train_" / "val_" prefix.
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

    # ------------------------------------------------------------------
    def _ensure_open(self, row: dict) -> None:
        if self._writer is not None:
            return
        self._fieldnames = self.FIXED_COLS + sorted(
            k for k in row if k not in self.FIXED_COLS
        )
        self._file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._fieldnames, extrasaction="ignore"
        )
        self._writer.writeheader()

    # ------------------------------------------------------------------
    def log_epoch(
        self,
        epoch: int,
        kl_weight: float,
        curriculum: "CurriculumState",
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_elapsed_s: float,
    ) -> None:
        row: Dict[str, object] = {
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

    # ------------------------------------------------------------------
    def write_best_metrics(
        self,
        epoch: int,
        reason: str,
        val_metrics: Dict[str, float],
        train_metrics: Dict[str, float],
    ) -> None:
        payload = {
            "epoch": epoch,
            "reason": reason,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "val": {k: round(float(v), 6) for k, v in val_metrics.items()},
            "train": {k: round(float(v), 6) for k, v in train_metrics.items()},
        }
        with open(self.save_dir / "best_metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ------------------------------------------------------------------
    def finish(
        self,
        final_epoch: int,
        best_val_loss: float,
        best_val_joint: float,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
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
        print(f"  Logs  : {self.csv_path}")
        print(f"  Summary: {self.save_dir / 'training_summary.json'}")


# ============================================================
# Config
# ============================================================

@dataclass
class TrainV4Config:
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

    # V4c: uncertainty losses (same as V4b)
    w_kl_context: float = 0.01    # KL( N(mu,sigma) || N(0,I) ) for context VAE
    w_vfe_kl: float = 0.02        # KL( posterior || transition prior ) — VFE term
    kl_warmup_epochs: int = 5     # epochs to linearly ramp KL weights from 0 → full

    # V4c: observation noise injection during training only.
    # Forces the encoder to maintain genuine uncertainty in discrete pose beliefs
    # rather than collapsing to near-deterministic spikes.  Validation always
    # uses clean observations so metrics remain comparable across versions.
    # sigma=0.07 is a conservative start — impactful but not overwhelming.
    obs_noise_sigma: float = 0.07

    # V4d: context logvar coupling.
    # Trains context_logvar to track reconstruction difficulty: high recon error
    # → high logvar (genuine uncertainty); low recon error → low logvar (confident).
    # When > 0.0 this is V4d training — change save_dir to ./checkpoints_v4d.
    # Uses Pearson correlation between per-step recon L1 error and mean logvar.
    # Start with 0.05; the loss is bounded in [-1, 1] (negated correlation).
    w_logvar_coupling: float = 0.05

    # V4e: random pixel masking augmentation (training only).
    # A fraction of pixels are zeroed per frame, independently per (B, T) sample.
    # This forces the encoder to maintain calibrated uncertainty when structural
    # image information is missing — bridging the train/eval distribution gap.
    # Set to 0.0 to disable (V4d behaviour).
    # 0.20 exposes the model to partial masking without being too harsh.
    obs_mask_fraction: float = 0.20

    # V4: predictive prior toggle
    use_predictive_prior: bool = True

    # Curriculum
    use_curriculum: bool = True

    # Runtime
    print_every: int = 50
    save_dir: str = "./checkpoints_v4e"
    save_every_epoch: bool = False
    use_amp: bool = True
    deterministic: bool = False
    seed: int = 42


# ============================================================
# Curriculum schedule  (identical to V3)
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


def get_curriculum_state(epoch: int, cfg: TrainV4Config) -> CurriculumState:
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
    row_loss = F.cross_entropy(row_logits.reshape(B * T, -1), rows.reshape(B * T).long())
    col_loss = F.cross_entropy(col_logits.reshape(B * T, -1), cols.reshape(B * T).long())
    heading_loss = F.cross_entropy(heading_logits.reshape(B * T, -1), headings.reshape(B * T).long())
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
    """If collision occurs, next predicted belief should stay close to previous belief."""
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


def compute_vfe_kl_loss(
    posterior_row_probs: torch.Tensor,       # [B, T, R]  — t=1..T
    posterior_col_probs: torch.Tensor,       # [B, T, C]
    posterior_heading_probs: torch.Tensor,   # [B, T, Hd]
    prior_row_probs: torch.Tensor,           # [B, T-1, R]
    prior_col_probs: torch.Tensor,           # [B, T-1, C]
    prior_heading_probs: torch.Tensor,       # [B, T-1, Hd]
) -> torch.Tensor:
    """
    KL( Q(s_t | o_{1:t}) || P(s_t | s_{t-1}, a_{t-1}) ) summed over factors,
    averaged over batch and time.

    Posterior is detached to avoid propagating gradients back through the
    Bayesian update loop into the transition model weights (which are already
    supervised by the rollout losses).
    """
    eps = 1e-8
    # Use posterior for t=1..T, prior covers t=1..T (i.e. T-1 steps)
    post_row = posterior_row_probs[:, 1:, :].detach()       # [B, T-1, R]
    post_col = posterior_col_probs[:, 1:, :].detach()
    post_heading = posterior_heading_probs[:, 1:, :].detach()

    kl_row = F.kl_div(
        torch.log(prior_row_probs.clamp_min(eps)),
        post_row,
        reduction="batchmean",
    )
    kl_col = F.kl_div(
        torch.log(prior_col_probs.clamp_min(eps)),
        post_col,
        reduction="batchmean",
    )
    kl_heading = F.kl_div(
        torch.log(prior_heading_probs.clamp_min(eps)),
        post_heading,
        reduction="batchmean",
    )

    return kl_row + kl_col + kl_heading


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
# Rollout target builder  (identical to V3)
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

    row_probs_start_list, col_probs_start_list, heading_probs_start_list = [], [], []
    context_start_list, action_roll_list = [], []
    row_target_list, col_target_list, heading_target_list = [], [], []
    obs_target_list, collision_target_list = [], []
    row_probs_target_list, col_probs_target_list, heading_probs_target_list = [], [], []
    context_target_list, start_index_list = [], []

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
        "row_probs_start": torch.stack(row_probs_start_list),
        "col_probs_start": torch.stack(col_probs_start_list),
        "heading_probs_start": torch.stack(heading_probs_start_list),
        "context_start": torch.stack(context_start_list),
        "action_roll": torch.stack(action_roll_list),
        "row_target": torch.stack(row_target_list),
        "col_target": torch.stack(col_target_list),
        "heading_target": torch.stack(heading_target_list),
        "obs_target": torch.stack(obs_target_list),
        "collision_target": torch.stack(collision_target_list),
        "row_probs_target": torch.stack(row_probs_target_list),
        "col_probs_target": torch.stack(col_probs_target_list),
        "heading_probs_target": torch.stack(heading_probs_target_list),
        "context_target": torch.stack(context_target_list),
        "start_index": torch.tensor(start_index_list, dtype=torch.long,
                                    device=row_probs_seq.device),
    }


# ============================================================
# Batch loss computation
# ============================================================

def compute_batch_losses(
    model: WorldModelV4,
    batch: Dict[str, torch.Tensor],
    cfg: TrainV4Config,
    curriculum: CurriculumState,
    kl_weight: float = 1.0,
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

    # ---- Filtering losses ----
    recon_loss = compute_recon_loss(recon_seq, observations)

    pose_loss = compute_pose_loss(
        row_logits=row_logits_seq,
        col_logits=col_logits_seq,
        heading_logits=heading_logits_seq,
        rows=rows,
        cols=cols,
        headings=headings,
    )

    collision_loss = compute_collision_loss(collision_logits_seq, collisions)

    # ---- V4: Context KL loss (VAE) ----
    # KL( N(mu, sigma) || N(0, I) ) — encourages the context to be
    # uncertain in regions with weak evidence, confident elsewhere.
    kl_context = model.compute_context_kl(
        filt["context_mu_seq"],
        filt["context_logvar_seq"],
    )

    # ---- V4: VFE KL loss (predictive prior) ----
    # KL( posterior || transition_prior ) — ensures the posterior
    # stays consistent with what the transition model predicted.
    if cfg.use_predictive_prior:
        kl_vfe = compute_vfe_kl_loss(
            posterior_row_probs=row_probs_seq,
            posterior_col_probs=col_probs_seq,
            posterior_heading_probs=heading_probs_seq,
            prior_row_probs=filt["transition_prior_row_probs"],
            prior_col_probs=filt["transition_prior_col_probs"],
            prior_heading_probs=filt["transition_prior_heading_probs"],
        )
    else:
        kl_vfe = row_logits_seq.new_tensor(0.0)

    # ---- Rollout targets ----
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

    # ---- Rollout losses ----
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

    # ---- V4d: context logvar coupling ----
    # Maximise Pearson correlation between per-step reconstruction L1 error
    # and mean context logvar.  Loss = -correlation ∈ [-1, 1].
    if cfg.w_logvar_coupling > 0.0:
        recon_err = F.l1_loss(recon_seq, observations, reduction='none').mean(dim=[2, 3, 4])  # [B, T]
        logvar_mean = filt["context_logvar_seq"].mean(dim=-1)  # [B, T]
        x_flat = recon_err.detach().reshape(-1)
        y_flat = logvar_mean.reshape(-1)
        x_c = x_flat - x_flat.mean()
        y_c = y_flat - y_flat.mean()
        logvar_coupling_loss = -(x_c * y_c).sum() / (
            (x_c.pow(2).sum().sqrt() * y_c.pow(2).sum().sqrt()).clamp_min(1e-8)
        )
    else:
        logvar_coupling_loss = row_logits_seq.new_tensor(0.0)

    # ---- Total loss ----
    # kl_weight is annealed from 0→1 over kl_warmup_epochs so that the model
    # first learns accurate beliefs before the KL regularisation kicks in.
    total_loss = (
        cfg.w_recon * recon_loss
        + cfg.w_pose * pose_loss
        + cfg.w_collision * collision_loss
        + curriculum.w_roll_pose * roll_pose_loss
        + curriculum.w_roll_recon * roll_recon_loss
        + curriculum.w_roll_collision * roll_collision_loss
        + curriculum.w_roll_stay * roll_stay_loss
        + curriculum.w_context_stability * context_stability_loss
        + kl_weight * cfg.w_kl_context * kl_context
        + kl_weight * cfg.w_vfe_kl * kl_vfe
        + cfg.w_logvar_coupling * logvar_coupling_loss
    )

    metrics: Dict[str, float] = {
        "loss_total": float(total_loss.detach().item()),
        "loss_recon": float(recon_loss.detach().item()),
        "loss_pose": float(pose_loss.detach().item()),
        "loss_collision": float(collision_loss.detach().item()),
        "loss_kl_context": float(kl_context.detach().item()),
        "loss_kl_vfe": float(kl_vfe.detach().item()),
        "loss_roll_pose": float(roll_pose_loss.detach().item()),
        "loss_roll_recon": float(roll_recon_loss.detach().item()),
        "loss_roll_collision": float(roll_collision_loss.detach().item()),
        "loss_roll_stay": float(roll_stay_loss.detach().item()),
        "loss_context_stability": float(context_stability_loss.detach().item()),
        "loss_logvar_coupling": float(logvar_coupling_loss.detach().item()),
        "rollout_start_t_min": float(curriculum.rollout_start_t_min),
        "rollout_horizon": float(curriculum.rollout_horizon),
        "kl_weight": float(kl_weight),
    }

    metrics.update(compute_pose_metrics(
        row_logits_seq, col_logits_seq, heading_logits_seq,
        rows, cols, headings, prefix="filter",
    ))
    metrics.update(compute_pose_metrics(
        roll["row_logits_roll"], roll["col_logits_roll"], roll["heading_logits_roll"],
        rollout_targets["row_target"], rollout_targets["col_target"],
        rollout_targets["heading_target"], prefix="roll",
    ))
    metrics.update(compute_entropy_metrics(
        row_probs_seq, col_probs_seq, heading_probs_seq, prefix="filter",
    ))

    return total_loss, metrics


# ============================================================
# Observation noise (training only)
# ============================================================

def add_obs_noise(
    obs: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Add i.i.d. Gaussian noise to a float32 observation tensor and clamp to [0,1].
    Applied only during training so validation metrics reflect clean performance.
    obs shape: [B, T, C, H, W]
    """
    return (obs + torch.randn_like(obs) * sigma).clamp_(0.0, 1.0)


def add_obs_mask(
    obs: torch.Tensor,
    mask_fraction: float,
) -> torch.Tensor:
    """
    Randomly zero out a fraction of pixels per frame.
    The binary mask is sampled independently for each (B, T) frame and
    broadcast across channels, matching the eval masking scheme.
    obs shape: [B, T, C, H, W]
    """
    B, T, C, H, W = obs.shape
    keep_prob = 1.0 - mask_fraction
    mask = torch.bernoulli(
        torch.full((B, T, 1, H, W), keep_prob, device=obs.device, dtype=obs.dtype)
    )
    return obs * mask


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
    model: WorldModelV4,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    cfg: TrainV4Config,
    curriculum: CurriculumState,
    train: bool,
    kl_weight: float = 1.0,
) -> Dict[str, float]:
    model.train(train)
    all_metrics: List[Dict[str, float]] = []

    grad_context = torch.enable_grad() if train else torch.no_grad()

    with grad_context:
        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            # Corrupt observations during training so the model learns to maintain
            # genuine uncertainty in pose beliefs rather than collapsing to spikes.
            # Validation batches are always left clean.
            if train and cfg.obs_noise_sigma > 0.0:
                batch["observations"] = add_obs_noise(
                    batch["observations"], cfg.obs_noise_sigma
                )
            if train and cfg.obs_mask_fraction > 0.0:
                batch["observations"] = add_obs_mask(
                    batch["observations"], cfg.obs_mask_fraction
                )

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
                    kl_weight=kl_weight,
                )

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
                    f"roll_pose={metrics['loss_roll_pose']:.4f} | "
                    f"roll_joint={metrics['roll_joint_acc']:.3f} | "
                    f"recon={metrics['loss_recon']:.4f} | "
                    f"coll={metrics['loss_collision']:.4f} | "
                    f"kl_ctx={metrics['loss_kl_context']:.5f} | "
                    f"kl_vfe={metrics['loss_kl_vfe']:.5f} | "
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
    model: WorldModelV4,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    cfg: TrainV4Config,
    model_cfg: ModelV4Config,
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


def print_epoch_summary(name: str, metrics: Dict[str, float]) -> None:
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
        f"kl_ctx={metrics['loss_kl_context']:.5f} | "
        f"kl_vfe={metrics['loss_kl_vfe']:.5f} | "
        f"kl_w={metrics['kl_weight']:.3f} | "
        f"H=({metrics['filter_row_entropy']:.3f},"
        f"{metrics['filter_col_entropy']:.3f},"
        f"{metrics['filter_heading_entropy']:.3f})"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = TrainV4Config()

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

    model_cfg = ModelV4Config(
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
        use_predictive_prior=cfg.use_predictive_prior,
    )

    model = WorldModelV4(model_cfg).to(device)

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

    logger = TrainingLogger(save_dir)

    best_val_loss = math.inf
    best_val_joint = -math.inf
    last_train_metrics: Dict[str, float] = {}
    last_val_metrics: Dict[str, float] = {}

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.time()
        curriculum = get_curriculum_state(epoch, cfg)

        # KL annealing: ramp from 0 → 1 over kl_warmup_epochs
        kl_weight = min(1.0, epoch / max(1, cfg.kl_warmup_epochs))

        print(f"\nEpoch {epoch:03d}/{cfg.num_epochs}")
        print(
            f"Curriculum: start_t_min={curriculum.rollout_start_t_min}, "
            f"horizon={curriculum.rollout_horizon}, "
            f"w_roll_pose={curriculum.w_roll_pose} | "
            f"kl_weight={kl_weight:.3f}"
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
            kl_weight=kl_weight,
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
            kl_weight=kl_weight,
        )

        epoch_elapsed = time.time() - epoch_start
        print_epoch_summary("Train", train_metrics)
        print_epoch_summary("Val  ", val_metrics)
        print(f"  Epoch time: {epoch_elapsed:.1f}s")

        logger.log_epoch(
            epoch=epoch,
            kl_weight=kl_weight,
            curriculum=curriculum,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            epoch_elapsed_s=epoch_elapsed,
        )

        last_train_metrics = train_metrics
        last_val_metrics = val_metrics

        if cfg.save_every_epoch:
            save_checkpoint(
                save_dir / f"epoch_{epoch:03d}.pt",
                model, optimizer, epoch, cfg, model_cfg,
                train_metrics, val_metrics,
            )

        val_loss = val_metrics["loss_total"]
        val_joint = val_metrics["filter_joint_acc"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                save_dir / "best_model_v4.pt",
                model, optimizer, epoch, cfg, model_cfg,
                train_metrics, val_metrics,
            )
            logger.write_best_metrics(
                epoch=epoch,
                reason="best_val_loss",
                val_metrics=val_metrics,
                train_metrics=train_metrics,
            )
            print(f"  Saved new best model by val loss: {best_val_loss:.4f}")

        if val_joint > best_val_joint:
            best_val_joint = val_joint
            save_checkpoint(
                save_dir / "best_belief_model_v4.pt",
                model, optimizer, epoch, cfg, model_cfg,
                train_metrics, val_metrics,
            )
            logger.write_best_metrics(
                epoch=epoch,
                reason="best_val_joint_acc",
                val_metrics=val_metrics,
                train_metrics=train_metrics,
            )
            print(f"  Saved new best belief model by joint acc: {best_val_joint:.4f}")

    logger.finish(
        final_epoch=cfg.num_epochs,
        best_val_loss=best_val_loss,
        best_val_joint=best_val_joint,
        train_metrics=last_train_metrics,
        val_metrics=last_val_metrics,
    )


if __name__ == "__main__":
    main()

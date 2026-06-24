"""
train_compressed.py — Training script for WorldModelCompressed.

Usage:
    python train_compressed.py --config CL1
    python train_compressed.py --config CL2-DW --seed 0
    python train_compressed.py --gru 64 --feat 32 --ctx 16 --save_dir ./checkpoints/custom

The training logic is identical to train_v5.py; only the model import and config
construction change. All augmentations (noise + masking + logvar coupling) from
V5f are included by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from compression_configs import CONFIGS
from dataset_loader import LoaderConfig, build_dataloaders
from model_compressed import CompressedModelConfig, WorldModelCompressed


# ============================================================
# Training Config
# ============================================================

@dataclass
class TrainCompressedConfig:
    dataset_path:       str  = "../../dataset/train_dataset_v7.npz"
    val_fraction:       float = 0.1
    batch_size:         int   = 32
    num_workers:        int   = 4
    pin_memory:         bool  = True
    shuffle_train:      bool  = True
    persistent_workers: bool  = True
    prefetch_factor:    int   = 2
    drop_last_train:    bool  = False
    drop_last_val:      bool  = False

    lr:              float = 5e-4
    weight_decay:    float = 1e-5
    num_epochs:      int   = 40
    grad_clip_norm:  float = 1.0

    rollout_start_t_min: int = 4
    rollout_horizon:     int = 5

    w_recon:      float = 0.5
    w_pose:       float = 2.0
    w_collision:  float = 0.3
    w_roll_pose:  float = 1.0
    w_roll_recon: float = 0.5
    w_roll_collision: float = 0.5
    w_roll_stay:  float = 0.5
    w_context_stability: float = 0.1

    w_kl_context:     float = 0.01
    w_vfe_kl:         float = 0.02
    kl_warmup_epochs: int   = 5

    obs_noise_sigma_max: float = 0.20
    obs_mask_fraction:   float = 0.20
    w_logvar_coupling:   float = 0.05

    use_predictive_prior: bool = True
    use_curriculum:       bool = True

    print_every:      int  = 50
    save_dir:         str  = "./checkpoints/CL0"
    save_every_epoch: bool = False
    use_amp:          bool = True
    deterministic:    bool = False
    seed:             int  = 42


# ============================================================
# Curriculum
# ============================================================

@dataclass
class CurriculumState:
    rollout_start_t_min:   int
    rollout_horizon:       int
    w_roll_pose:           float
    w_roll_recon:          float
    w_roll_collision:      float
    w_roll_stay:           float
    w_context_stability:   float


def get_curriculum_state(epoch: int, cfg: TrainCompressedConfig) -> CurriculumState:
    if not cfg.use_curriculum:
        return CurriculumState(
            cfg.rollout_start_t_min, cfg.rollout_horizon,
            cfg.w_roll_pose, cfg.w_roll_recon, cfg.w_roll_collision,
            cfg.w_roll_stay, cfg.w_context_stability,
        )
    if epoch <= 10:
        return CurriculumState(4,  3,  0.5,  0.25, 0.25, 0.25, 0.05)
    if epoch <= 25:
        return CurriculumState(6,  5,  1.0,  0.5,  0.5,  0.5,  0.1)
    return     CurriculumState(8,  10, 1.5,  0.75, 0.75, 0.75, 0.1)


# ============================================================
# Utilities
# ============================================================

def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_obs_noise(obs: torch.Tensor, sigma: float) -> torch.Tensor:
    return (obs + torch.randn_like(obs) * sigma).clamp_(0.0, 1.0)


def add_obs_mask(obs: torch.Tensor, mask_fraction: float) -> torch.Tensor:
    B, T, C, H, W = obs.shape
    keep = 1.0 - mask_fraction
    mask = torch.bernoulli(torch.full((B, T, 1, H, W), keep, device=obs.device, dtype=obs.dtype))
    return obs * mask


def move_batch_to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def average_metrics(metrics_list):
    keys = metrics_list[0].keys()
    return {k: float(sum(m[k] for m in metrics_list) / len(metrics_list)) for k in keys}


# ============================================================
# Loss helpers
# ============================================================

def compute_pose_loss(row_logits, col_logits, heading_logits, rows, cols, headings):
    B, T, _ = row_logits.shape
    return (
        F.cross_entropy(row_logits.reshape(B*T, -1),     rows.reshape(B*T).long())
        + F.cross_entropy(col_logits.reshape(B*T, -1),   cols.reshape(B*T).long())
        + F.cross_entropy(heading_logits.reshape(B*T,-1), headings.reshape(B*T).long())
    )


def compute_collision_loss(collision_logits, collision_targets):
    B, T, C = collision_logits.shape
    return F.cross_entropy(collision_logits.reshape(B*T, C), collision_targets.reshape(B*T).long())


def compute_recon_loss(pred, target):
    return F.l1_loss(pred, target)


def compute_factor_stay_loss(
    row_probs_pred, col_probs_pred, heading_probs_pred,
    row_probs_prev_true, col_probs_prev_true, heading_probs_prev_true,
    context_pred, context_prev_true, collision_targets,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = collision_targets.float()
    if mask.sum().item() < 1:
        z = row_probs_pred.new_tensor(0.0)
        return z, z
    eps = 1e-8
    kl_row = F.kl_div(torch.log(row_probs_pred.clamp_min(eps)),     row_probs_prev_true,     reduction="none").sum(dim=-1)
    kl_col = F.kl_div(torch.log(col_probs_pred.clamp_min(eps)),     col_probs_prev_true,     reduction="none").sum(dim=-1)
    kl_hdg = F.kl_div(torch.log(heading_probs_pred.clamp_min(eps)), heading_probs_prev_true, reduction="none").sum(dim=-1)
    factor_loss = ((kl_row + kl_col + kl_hdg) * mask).sum() / mask.sum()
    ctx_mse     = (((context_pred - context_prev_true)**2).mean(dim=-1) * mask).sum() / mask.sum()
    return factor_loss, ctx_mse


def compute_vfe_kl_loss(
    post_row, post_col, post_hdg,
    prior_row, prior_col, prior_hdg,
) -> torch.Tensor:
    eps = 1e-8
    return (
        F.kl_div(torch.log(prior_row.clamp_min(eps)), post_row[:, 1:, :].detach(), reduction="batchmean")
        + F.kl_div(torch.log(prior_col.clamp_min(eps)), post_col[:, 1:, :].detach(), reduction="batchmean")
        + F.kl_div(torch.log(prior_hdg.clamp_min(eps)), post_hdg[:, 1:, :].detach(), reduction="batchmean")
    )


# ============================================================
# Pose metrics
# ============================================================

def compute_pose_metrics(row_logits, col_logits, heading_logits, rows, cols, headings, prefix):
    row_ok = row_logits.argmax(dim=-1).eq(rows)
    col_ok = col_logits.argmax(dim=-1).eq(cols)
    hdg_ok = heading_logits.argmax(dim=-1).eq(headings)
    return {
        f"{prefix}_row_acc":     float(row_ok.float().mean()),
        f"{prefix}_col_acc":     float(col_ok.float().mean()),
        f"{prefix}_heading_acc": float(hdg_ok.float().mean()),
        f"{prefix}_joint_acc":   float((row_ok & col_ok & hdg_ok).float().mean()),
    }


def compute_entropy_metrics(row_probs, col_probs, heading_probs, prefix):
    eps = 1e-8
    def _h(p): return float(-(p * p.clamp_min(eps).log()).sum(dim=-1).mean())
    return {
        f"{prefix}_row_entropy":     _h(row_probs),
        f"{prefix}_col_entropy":     _h(col_probs),
        f"{prefix}_heading_entropy": _h(heading_probs),
    }


# ============================================================
# Rollout target builder
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
        raise ValueError(f"rollout_start_t_min={rollout_start_t_min} too large for T={T}, K={K}.")

    rps, cps, hps, cs, ars = [], [], [], [], []
    rt, ct, ht, ot, colt   = [], [], [], [], []
    rpt, cpt, hpt, ctx_t, si = [], [], [], [], []

    for b in range(B):
        t = random.randint(rollout_start_t_min, max_valid_t)
        si.append(t)
        rps.append(row_probs_seq[b, t]);      cps.append(col_probs_seq[b, t])
        hps.append(heading_probs_seq[b, t]);  cs.append(context_seq[b, t])
        ars.append(actions[b, t:t+K])
        rt.append(rows[b, t+1:t+1+K]);        ct.append(cols[b, t+1:t+1+K])
        ht.append(headings[b, t+1:t+1+K]);    ot.append(observations[b, t+1:t+1+K])
        colt.append(collisions[b, t:t+K])
        rpt.append(row_probs_seq[b, t+1:t+1+K])
        cpt.append(col_probs_seq[b, t+1:t+1+K])
        hpt.append(heading_probs_seq[b, t+1:t+1+K])
        ctx_t.append(context_seq[b, t+1:t+1+K])

    return {
        "row_probs_start":   torch.stack(rps),
        "col_probs_start":   torch.stack(cps),
        "heading_probs_start": torch.stack(hps),
        "context_start":     torch.stack(cs),
        "action_roll":       torch.stack(ars),
        "row_target":        torch.stack(rt),
        "col_target":        torch.stack(ct),
        "heading_target":    torch.stack(ht),
        "obs_target":        torch.stack(ot),
        "collision_target":  torch.stack(colt),
        "row_probs_target":  torch.stack(rpt),
        "col_probs_target":  torch.stack(cpt),
        "heading_probs_target": torch.stack(hpt),
        "context_target":    torch.stack(ctx_t),
        "start_index":       torch.tensor(si, dtype=torch.long, device=row_probs_seq.device),
    }


# ============================================================
# Batch loss
# ============================================================

def compute_batch_losses(
    model: WorldModelCompressed,
    batch: Dict[str, torch.Tensor],
    cfg:   TrainCompressedConfig,
    curriculum: CurriculumState,
    kl_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    observations = batch["observations"]
    actions      = batch["actions"]
    rows         = batch["rows"].long()
    cols         = batch["cols"].long()
    headings     = batch["headings"].long()
    collisions   = batch["collisions"].long()

    filt = model.forward_filter(observations, actions)

    row_logits_seq    = filt["row_logits_seq"]
    col_logits_seq    = filt["col_logits_seq"]
    heading_logits_seq = filt["heading_logits_seq"]
    row_probs_seq     = filt["row_probs_seq"]
    col_probs_seq     = filt["col_probs_seq"]
    heading_probs_seq = filt["heading_probs_seq"]
    context_seq       = filt["context_seq"]
    recon_seq         = filt["recon_seq"]
    collision_logits_seq = filt["collision_logits_seq"]

    pose_loss      = compute_pose_loss(row_logits_seq, col_logits_seq, heading_logits_seq, rows, cols, headings)
    collision_loss = compute_collision_loss(collision_logits_seq, collisions)
    recon_loss     = compute_recon_loss(recon_seq, observations) if recon_seq is not None else row_logits_seq.new_tensor(0.0)
    kl_context     = model.compute_context_kl(filt["context_mu_seq"], filt["context_logvar_seq"])

    if cfg.use_predictive_prior:
        kl_vfe = compute_vfe_kl_loss(
            row_probs_seq, col_probs_seq, heading_probs_seq,
            filt["transition_prior_row_probs"],
            filt["transition_prior_col_probs"],
            filt["transition_prior_heading_probs"],
        )
    else:
        kl_vfe = row_logits_seq.new_tensor(0.0)

    if recon_seq is not None and cfg.w_logvar_coupling > 0.0:
        recon_err    = F.l1_loss(recon_seq, observations, reduction="none").mean(dim=[2, 3, 4])
        logvar_mean  = filt["context_logvar_seq"].mean(dim=-1)
        x_flat       = recon_err.detach().reshape(-1)
        y_flat       = logvar_mean.reshape(-1)
        x_c, y_c    = x_flat - x_flat.mean(), y_flat - y_flat.mean()
        logvar_coupling_loss = -(x_c * y_c).sum() / (
            (x_c.pow(2).sum().sqrt() * y_c.pow(2).sum().sqrt()).clamp_min(1e-8)
        )
    else:
        logvar_coupling_loss = row_logits_seq.new_tensor(0.0)

    rollout_targets = sample_rollout_targets(
        row_probs_seq, col_probs_seq, heading_probs_seq, context_seq,
        observations, actions, rows, cols, headings, collisions,
        curriculum.rollout_start_t_min, curriculum.rollout_horizon,
    )
    roll = model.rollout_from_filtered_state(
        rollout_targets["row_probs_start"],
        rollout_targets["col_probs_start"],
        rollout_targets["heading_probs_start"],
        rollout_targets["context_start"],
        rollout_targets["action_roll"],
    )

    roll_pose_loss      = compute_pose_loss(
        roll["row_logits_roll"], roll["col_logits_roll"], roll["heading_logits_roll"],
        rollout_targets["row_target"], rollout_targets["col_target"], rollout_targets["heading_target"],
    )
    roll_recon_loss     = compute_recon_loss(roll["recon_roll"], rollout_targets["obs_target"])
    roll_collision_loss = compute_collision_loss(roll["collision_logits_roll"], rollout_targets["collision_target"])

    row_probs_prev_true     = torch.cat([rollout_targets["row_probs_start"].unsqueeze(1),     rollout_targets["row_probs_target"][:, :-1, :]], dim=1)
    col_probs_prev_true     = torch.cat([rollout_targets["col_probs_start"].unsqueeze(1),     rollout_targets["col_probs_target"][:, :-1, :]], dim=1)
    heading_probs_prev_true = torch.cat([rollout_targets["heading_probs_start"].unsqueeze(1), rollout_targets["heading_probs_target"][:, :-1, :]], dim=1)
    context_prev_true       = torch.cat([rollout_targets["context_start"].unsqueeze(1),       rollout_targets["context_target"][:, :-1, :]], dim=1)

    roll_stay_loss, ctx_stability_loss = compute_factor_stay_loss(
        roll["row_probs_roll"], roll["col_probs_roll"], roll["heading_probs_roll"],
        row_probs_prev_true, col_probs_prev_true, heading_probs_prev_true,
        roll["context_roll"], context_prev_true, rollout_targets["collision_target"],
    )

    total_loss = (
        cfg.w_recon                  * recon_loss
        + cfg.w_pose                 * pose_loss
        + cfg.w_collision            * collision_loss
        + curriculum.w_roll_pose     * roll_pose_loss
        + curriculum.w_roll_recon    * roll_recon_loss
        + curriculum.w_roll_collision * roll_collision_loss
        + curriculum.w_roll_stay     * roll_stay_loss
        + curriculum.w_context_stability * ctx_stability_loss
        + kl_weight * cfg.w_kl_context * kl_context
        + kl_weight * cfg.w_vfe_kl     * kl_vfe
        + cfg.w_logvar_coupling        * logvar_coupling_loss
    )

    metrics: Dict[str, float] = {
        "loss_total":             float(total_loss.detach()),
        "loss_recon":             float(recon_loss.detach()),
        "loss_pose":              float(pose_loss.detach()),
        "loss_collision":         float(collision_loss.detach()),
        "loss_kl_context":        float(kl_context.detach()),
        "loss_kl_vfe":            float(kl_vfe.detach()),
        "loss_roll_pose":         float(roll_pose_loss.detach()),
        "loss_roll_recon":        float(roll_recon_loss.detach()),
        "loss_roll_collision":    float(roll_collision_loss.detach()),
        "loss_roll_stay":         float(roll_stay_loss.detach()),
        "loss_context_stability": float(ctx_stability_loss.detach()),
        "loss_logvar_coupling":   float(logvar_coupling_loss.detach()),
        "kl_weight":              float(kl_weight),
    }
    metrics.update(compute_pose_metrics(row_logits_seq, col_logits_seq, heading_logits_seq, rows, cols, headings, "filter"))
    metrics.update(compute_pose_metrics(
        roll["row_logits_roll"], roll["col_logits_roll"], roll["heading_logits_roll"],
        rollout_targets["row_target"], rollout_targets["col_target"], rollout_targets["heading_target"], "roll",
    ))
    metrics.update(compute_entropy_metrics(row_probs_seq, col_probs_seq, heading_probs_seq, "filter"))
    return total_loss, metrics


# ============================================================
# Epoch runner
# ============================================================

def run_epoch(
    model:     WorldModelCompressed,
    loader:    DataLoader,
    optimizer,
    scaler,
    device:    torch.device,
    cfg:       TrainCompressedConfig,
    curriculum: CurriculumState,
    train:     bool,
    kl_weight: float = 1.0,
) -> Dict[str, float]:
    model.train(train)
    all_metrics = []

    with (torch.enable_grad() if train else torch.no_grad()):
        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            if train and cfg.obs_noise_sigma_max > 0.0:
                sigma = random.uniform(0.0, cfg.obs_noise_sigma_max)
                batch["observations"] = add_obs_noise(batch["observations"], sigma)

            if train and cfg.obs_mask_fraction > 0.0:
                batch["observations"] = add_obs_mask(batch["observations"], cfg.obs_mask_fraction)

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
                    f"  [batch {batch_idx+1:4d}] "
                    f"loss={metrics['loss_total']:.4f} | "
                    f"pose={metrics['loss_pose']:.4f} | "
                    f"joint={metrics['filter_joint_acc']:.3f} | "
                    f"roll={metrics['roll_joint_acc']:.3f} | "
                    f"kl_ctx={metrics['loss_kl_context']:.5f}"
                )

    return average_metrics(all_metrics)


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(save_path, model, optimizer, epoch, train_cfg, model_cfg, train_metrics, val_metrics):
    torch.save({
        "epoch":              epoch,
        "model_state_dict":   model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_cfg":          asdict(train_cfg),
        "model_cfg":          asdict(model_cfg),
        "inference_params":   model.inference_param_count,
        "total_params":       model.total_param_count,
        "train_metrics":      train_metrics,
        "val_metrics":        val_metrics,
    }, save_path)


def print_epoch_summary(label, metrics):
    print(
        f"{label:<5} | loss={metrics['loss_total']:.4f} | "
        f"pose={metrics['loss_pose']:.4f} | joint={metrics['filter_joint_acc']:.3f} | "
        f"roll={metrics['roll_joint_acc']:.3f} | "
        f"kl_ctx={metrics['loss_kl_context']:.5f} | kl_vfe={metrics['loss_kl_vfe']:.5f} | "
        f"H=({metrics['filter_row_entropy']:.3f},{metrics['filter_col_entropy']:.3f},{metrics['filter_heading_entropy']:.3f})"
    )


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train compressed AIF model")
    p.add_argument("--config", type=str, default=None,
                   help="Preset config name (e.g. CL1, CL2-DW). See compression_configs.py")
    p.add_argument("--gru",    type=int, default=None, help="GRU hidden dim (overrides config)")
    p.add_argument("--feat",   type=int, default=None, help="Encoder feat dim (overrides config)")
    p.add_argument("--ctx",    type=int, default=None, help="Context dim (overrides config)")
    p.add_argument("--depthwise", action="store_true", help="Use depthwise-separable encoder")
    p.add_argument("--save_dir", type=str, default=None, help="Override save directory")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def build_model_cfg(args, dataset_summary: dict) -> CompressedModelConfig:
    kwargs: dict = {}

    if args.config is not None:
        if args.config not in CONFIGS:
            raise ValueError(f"Unknown config '{args.config}'. Available: {list(CONFIGS.keys())}")
        kwargs = dict(CONFIGS[args.config])

    # Manual overrides
    if args.gru  is not None: kwargs["gru_hidden_dim"]   = args.gru
    if args.feat is not None: kwargs["encoder_feat_dim"] = args.feat
    if args.ctx  is not None: kwargs["context_dim"]      = args.ctx
    if args.depthwise:         kwargs["use_depthwise"]   = True

    # Defaults if nothing specified
    if not kwargs:
        kwargs = dict(CONFIGS["CL0"])

    return CompressedModelConfig(
        obs_channels=dataset_summary["channels"],
        obs_height=dataset_summary["height"],
        obs_width=dataset_summary["width"],
        num_actions=4,
        num_row_classes=dataset_summary["num_row_classes"],
        num_col_classes=dataset_summary["num_col_classes"],
        num_heading_classes=dataset_summary["num_heading_classes"],
        action_emb_dim=16,
        use_predictive_prior=True,
        **kwargs,
    )


def main():
    args = parse_args()
    cfg  = TrainCompressedConfig(seed=args.seed)
    if args.epochs:
        cfg.num_epochs = args.epochs

    seed_everything(cfg.seed, cfg.deterministic)
    device = get_device()
    print(f"Device: {device}")

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

    model_cfg = build_model_cfg(args, summary)

    # Determine save_dir
    if args.save_dir:
        cfg.save_dir = args.save_dir
    elif args.config:
        cfg.save_dir = f"./checkpoints/{args.config}"
    else:
        cfg.save_dir = f"./checkpoints/gru{model_cfg.gru_hidden_dim}_feat{model_cfg.encoder_feat_dim}_ctx{model_cfg.context_dim}"

    model = WorldModelCompressed(model_cfg).to(device)
    print(f"Config          : gru={model_cfg.gru_hidden_dim}, feat={model_cfg.encoder_feat_dim}, "
          f"ctx={model_cfg.context_dim}, depthwise={model_cfg.use_depthwise}")
    print(f"Total params    : {model.total_param_count:,}")
    print(f"Inference params: {model.inference_param_count:,}")
    print(f"Inference size  : {model.model_size_mb:.3f} MB (FP32)")
    print(f"Save dir        : {cfg.save_dir}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler    = torch.amp.GradScaler("cuda") if (cfg.use_amp and device.type == "cuda") else None

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "train_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(save_dir / "model_config.json", "w") as f:
        json.dump(asdict(model_cfg), f, indent=2)

    # Training summary written as we go
    best_val_loss  = math.inf
    best_val_joint = -math.inf
    last_train: Dict[str, float] = {}
    last_val:   Dict[str, float] = {}
    t_start = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        t_ep   = time.time()
        cur    = get_curriculum_state(epoch, cfg)
        kl_w   = min(1.0, epoch / max(1, cfg.kl_warmup_epochs))

        print(f"\nEpoch {epoch:03d}/{cfg.num_epochs}  kl_w={kl_w:.3f}")
        train_m = run_epoch(model, train_loader, optimizer, scaler, device, cfg, cur, train=True,  kl_weight=kl_w)
        val_m   = run_epoch(model, val_loader,   None,      None,   device, cfg, cur, train=False, kl_weight=kl_w)

        print_epoch_summary("Train", train_m)
        print_epoch_summary("Val  ", val_m)
        print(f"  Epoch time: {time.time()-t_ep:.1f}s")

        last_train, last_val = train_m, val_m

        if cfg.save_every_epoch:
            save_checkpoint(save_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, cfg, model_cfg, train_m, val_m)

        if val_m["loss_total"] < best_val_loss:
            best_val_loss = val_m["loss_total"]
            save_checkpoint(save_dir / "best_model.pt", model, optimizer, epoch, cfg, model_cfg, train_m, val_m)
            print(f"  >> Saved best (loss {best_val_loss:.4f})")

        if val_m["filter_joint_acc"] > best_val_joint:
            best_val_joint = val_m["filter_joint_acc"]
            save_checkpoint(save_dir / "best_belief_model.pt", model, optimizer, epoch, cfg, model_cfg, train_m, val_m)
            print(f"  >> Saved best belief (joint {best_val_joint:.4f})")

    total_s = time.time() - t_start
    summary_out = {
        "config":               args.config or "custom",
        "gru_hidden_dim":       model_cfg.gru_hidden_dim,
        "encoder_feat_dim":     model_cfg.encoder_feat_dim,
        "context_dim":          model_cfg.context_dim,
        "use_depthwise":        model_cfg.use_depthwise,
        "total_params":         model.total_param_count,
        "inference_params":     model.inference_param_count,
        "inference_size_mb":    round(model.model_size_mb, 4),
        "best_val_loss":        round(best_val_loss, 6),
        "best_val_joint_acc":   round(best_val_joint, 6),
        "total_training_min":   round(total_s / 60, 2),
        "final_train_metrics":  {k: round(float(v), 6) for k, v in last_train.items()},
        "final_val_metrics":    {k: round(float(v), 6) for k, v in last_val.items()},
    }
    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\nDone. Total: {total_s/60:.1f} min | best_val_loss={best_val_loss:.4f} | best_joint={best_val_joint:.4f}")


if __name__ == "__main__":
    main()

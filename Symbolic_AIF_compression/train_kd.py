"""
train_kd.py — Knowledge Distillation training for WorldModelCompressed.

Teacher: V5f (frozen, 1.68M params, 96% navigation success).
Student: CL1 / CL2 compressed architecture (any config in compression_configs.py).

KD additions on top of the standard hard-label losses from train_compressed.py:
  kd_belief_loss  : KL(student || teacher) for row/col/heading belief distributions
  kd_context_loss : MSE(student_context, linear_proj(teacher_context))  [off by default]

Temperature T > 1 softens the teacher's peaked distributions, giving the student
richer gradient signal from distribution tails ("dark knowledge").

Usage:
    python train_kd.py --config CL1 \\
        --teacher_ckpt ../Symbolic_AIF_v5/checkpoints_v5f/best_model_v5.pt

    python train_kd.py --config CL2 \\
        --teacher_ckpt ../Symbolic_AIF_v5/checkpoints_v5f/best_model_v5.pt \\
        --w_kd_belief 3.0 --kd_temp 3.0 --w_kd_context 0.1

Checkpoints saved to:
    ./checkpoints/CL1-KD/best_model.pt
    ./checkpoints/CL2-KD/best_model.pt
(same format as train_compressed.py — evaluate_compressed.py works unchanged)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from compression_configs import CONFIGS
from dataset_loader import LoaderConfig, build_dataloaders
from model_compressed import CompressedModelConfig, WorldModelCompressed

# V5f teacher model — locate relative to this file
_V5_DIR = Path(__file__).resolve().parent.parent / "Symbolic_AIF_v5"
sys.path.insert(0, str(_V5_DIR))
from model_v5 import ModelV5Config, WorldModelV5  # noqa: E402  (path insert above)

# Reuse every shared helper from train_compressed.py; only override what changes
from train_compressed import (
    TrainCompressedConfig,
    CurriculumState,
    get_curriculum_state,
    seed_everything,
    get_device,
    add_obs_noise,
    add_obs_mask,
    move_batch_to_device,
    average_metrics,
    compute_pose_loss,
    compute_collision_loss,
    compute_recon_loss,
    compute_factor_stay_loss,
    compute_vfe_kl_loss,
    compute_pose_metrics,
    compute_entropy_metrics,
    sample_rollout_targets,
    print_epoch_summary,
    save_checkpoint,
)


# ============================================================
# KD hyperparameters
# ============================================================

@dataclass
class KDConfig:
    teacher_ckpt:     str   = ""
    w_kd_belief:      float = 0.5  # KL(student || teacher) for row/col/heading
    w_kd_context:     float = 0.2  # MSE context alignment (prevents context collapse)
    kd_temperature:   float = 3.0  # soften teacher distributions
    kd_warmup_epochs: int   = 10   # linear ramp from 0 → 1.0 for KD weight


# ============================================================
# Teacher loading
# ============================================================

def load_teacher(ckpt_path: str, device: torch.device) -> WorldModelV5:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw  = ckpt["model_cfg"]
    cfg  = ModelV5Config(**{k: v for k, v in raw.items()
                            if k in ModelV5Config.__dataclass_fields__})
    teacher = WorldModelV5(cfg).to(device)
    teacher.load_state_dict(ckpt["model_state_dict"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    n = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher (V5f): {n:,} params  context_dim={cfg.context_dim}")
    return teacher


# ============================================================
# Optional context projection (training-only, not saved to student ckpt)
# ============================================================

class ContextProjection(nn.Module):
    """Linear map: teacher context dim → student context dim."""
    def __init__(self, teacher_dim: int, student_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ============================================================
# KD loss helpers
# ============================================================

def compute_kd_belief_loss(
    s_row:  torch.Tensor, s_col:  torch.Tensor, s_hdg:  torch.Tensor,
    t_row:  torch.Tensor, t_col:  torch.Tensor, t_hdg:  torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(student || temperature-softened teacher) summed across factors.

    Scaling by T² corrects for the temperature rescaling so the loss magnitude
    stays comparable to the hard cross-entropy terms (Hinton et al., 2015).
    """
    eps = 1e-8
    T   = temperature

    def _soft(probs: torch.Tensor) -> torch.Tensor:
        return F.softmax(torch.log(probs.clamp_min(eps)) / T, dim=-1)

    def _kl(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return F.kl_div(torch.log(s.clamp_min(eps)), _soft(t), reduction="batchmean")

    return (T ** 2) * (_kl(s_row, t_row) + _kl(s_col, t_col) + _kl(s_hdg, t_hdg))


def compute_kd_context_loss(
    student_ctx:  torch.Tensor,
    teacher_ctx:  torch.Tensor,
    ctx_proj:     Optional[ContextProjection],
) -> torch.Tensor:
    if ctx_proj is None:
        return student_ctx.new_tensor(0.0)
    return F.mse_loss(student_ctx, ctx_proj(teacher_ctx.detach()))


# ============================================================
# Batch loss with KD (extends train_compressed.compute_batch_losses)
# ============================================================

def compute_batch_losses_kd(
    student:    WorldModelCompressed,
    teacher:    WorldModelV5,
    ctx_proj:   Optional[ContextProjection],
    batch:      Dict[str, torch.Tensor],
    cfg:        TrainCompressedConfig,
    kd_cfg:     KDConfig,
    curriculum: CurriculumState,
    kl_weight:  float,
    kd_weight:  float,
) -> Tuple[torch.Tensor, Dict[str, float]]:

    observations = batch["observations"]
    actions      = batch["actions"]
    rows         = batch["rows"].long()
    cols         = batch["cols"].long()
    headings     = batch["headings"].long()
    collisions   = batch["collisions"].long()

    # ---- Student forward ----
    filt = student.forward_filter(observations, actions)

    row_logits_seq       = filt["row_logits_seq"]
    col_logits_seq       = filt["col_logits_seq"]
    heading_logits_seq   = filt["heading_logits_seq"]
    row_probs_seq        = filt["row_probs_seq"]
    col_probs_seq        = filt["col_probs_seq"]
    heading_probs_seq    = filt["heading_probs_seq"]
    context_seq          = filt["context_seq"]
    recon_seq            = filt["recon_seq"]
    collision_logits_seq = filt["collision_logits_seq"]

    # ---- Hard-label losses (identical to train_compressed.py) ----
    pose_loss      = compute_pose_loss(row_logits_seq, col_logits_seq, heading_logits_seq, rows, cols, headings)
    collision_loss = compute_collision_loss(collision_logits_seq, collisions)
    recon_loss     = compute_recon_loss(recon_seq, observations) if recon_seq is not None \
                     else row_logits_seq.new_tensor(0.0)
    kl_context     = student.compute_context_kl(filt["context_mu_seq"], filt["context_logvar_seq"])

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
        recon_err   = F.l1_loss(recon_seq, observations, reduction="none").mean(dim=[2, 3, 4])
        logvar_mean = filt["context_logvar_seq"].mean(dim=-1)
        x_c = recon_err.detach().reshape(-1); x_c = x_c - x_c.mean()
        y_c = logvar_mean.reshape(-1);         y_c = y_c - y_c.mean()
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
    roll = student.rollout_from_filtered_state(
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

    rpp = torch.cat([rollout_targets["row_probs_start"].unsqueeze(1),     rollout_targets["row_probs_target"][:, :-1, :]],     dim=1)
    cpp = torch.cat([rollout_targets["col_probs_start"].unsqueeze(1),     rollout_targets["col_probs_target"][:, :-1, :]],     dim=1)
    hpp = torch.cat([rollout_targets["heading_probs_start"].unsqueeze(1), rollout_targets["heading_probs_target"][:, :-1, :]], dim=1)
    ctp = torch.cat([rollout_targets["context_start"].unsqueeze(1),       rollout_targets["context_target"][:, :-1, :]],       dim=1)

    roll_stay_loss, ctx_stability_loss = compute_factor_stay_loss(
        roll["row_probs_roll"], roll["col_probs_roll"], roll["heading_probs_roll"],
        rpp, cpp, hpp, roll["context_roll"], ctp, rollout_targets["collision_target"],
    )

    hard_loss = (
          cfg.w_recon                   * recon_loss
        + cfg.w_pose                    * pose_loss
        + cfg.w_collision               * collision_loss
        + curriculum.w_roll_pose        * roll_pose_loss
        + curriculum.w_roll_recon       * roll_recon_loss
        + curriculum.w_roll_collision   * roll_collision_loss
        + curriculum.w_roll_stay        * roll_stay_loss
        + curriculum.w_context_stability * ctx_stability_loss
        + kl_weight * cfg.w_kl_context  * kl_context
        + kl_weight * cfg.w_vfe_kl      * kl_vfe
        + cfg.w_logvar_coupling          * logvar_coupling_loss
    )

    # ---- Teacher forward (frozen, no grad) ----
    with torch.no_grad():
        t_filt = teacher.forward_filter(observations, actions, skip_recon=True)

    kd_belief_loss = compute_kd_belief_loss(
        row_probs_seq, col_probs_seq, heading_probs_seq,
        t_filt["row_probs_seq"], t_filt["col_probs_seq"], t_filt["heading_probs_seq"],
        kd_cfg.kd_temperature,
    )
    kd_context_loss = (
        compute_kd_context_loss(context_seq, t_filt["context_seq"], ctx_proj)
        if kd_cfg.w_kd_context > 0.0
        else row_logits_seq.new_tensor(0.0)
    )

    total_loss = (
          hard_loss
        + kd_weight * kd_cfg.w_kd_belief  * kd_belief_loss
        + kd_weight * kd_cfg.w_kd_context * kd_context_loss
    )

    metrics: Dict[str, float] = {
        "loss_total":             float(total_loss.detach()),
        "loss_hard":              float(hard_loss.detach()),
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
        "loss_kd_belief":         float(kd_belief_loss.detach()),
        "loss_kd_context":        float(kd_context_loss.detach()),
        "kl_weight":              float(kl_weight),
        "kd_weight":              float(kd_weight),
    }
    metrics.update(compute_pose_metrics(
        row_logits_seq, col_logits_seq, heading_logits_seq, rows, cols, headings, "filter"))
    metrics.update(compute_pose_metrics(
        roll["row_logits_roll"], roll["col_logits_roll"], roll["heading_logits_roll"],
        rollout_targets["row_target"], rollout_targets["col_target"], rollout_targets["heading_target"], "roll",
    ))
    metrics.update(compute_entropy_metrics(row_probs_seq, col_probs_seq, heading_probs_seq, "filter"))
    return total_loss, metrics


# ============================================================
# Epoch runner
# ============================================================

def run_epoch_kd(
    student:    WorldModelCompressed,
    teacher:    WorldModelV5,
    ctx_proj:   Optional[ContextProjection],
    loader:     DataLoader,
    optimizer,
    scaler,
    device:     torch.device,
    cfg:        TrainCompressedConfig,
    kd_cfg:     KDConfig,
    curriculum: CurriculumState,
    train:      bool,
    kl_weight:  float,
    kd_weight:  float,
) -> Dict[str, float]:
    student.train(train)
    if ctx_proj is not None:
        ctx_proj.train(train)
    all_metrics = []

    with (torch.enable_grad() if train else torch.no_grad()):
        for batch_idx, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)

            if train and cfg.obs_noise_sigma_max > 0.0:
                batch["observations"] = add_obs_noise(batch["observations"],
                                                       random.uniform(0.0, cfg.obs_noise_sigma_max))
            if train and cfg.obs_mask_fraction > 0.0:
                batch["observations"] = add_obs_mask(batch["observations"], cfg.obs_mask_fraction)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=(cfg.use_amp and device.type == "cuda")):
                loss, metrics = compute_batch_losses_kd(
                    student, teacher, ctx_proj, batch, cfg, kd_cfg,
                    curriculum, kl_weight, kd_weight,
                )

            if train:
                params = list(student.parameters())
                if ctx_proj is not None:
                    params += list(ctx_proj.parameters())
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip_norm)
                    optimizer.step()

            all_metrics.append(metrics)

            if train and ((batch_idx + 1) % cfg.print_every == 0):
                print(
                    f"  [batch {batch_idx+1:4d}] "
                    f"loss={metrics['loss_total']:.4f} | "
                    f"hard={metrics['loss_hard']:.4f} | "
                    f"kd_bel={metrics['loss_kd_belief']:.4f} | "
                    f"joint={metrics['filter_joint_acc']:.3f} | "
                    f"roll={metrics['roll_joint_acc']:.3f}"
                )

    return average_metrics(all_metrics)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="KD training for compressed AIF world model")
    p.add_argument("--config",       type=str, default="CL1",
                   help="Student config name (e.g. CL1, CL2). See compression_configs.py")
    p.add_argument("--teacher_ckpt", type=str,
                   default="../Symbolic_AIF_v5/checkpoints_v5f/best_model_v5.pt",
                   help="Path to the frozen V5f teacher checkpoint")
    p.add_argument("--save_dir",     type=str, default=None,
                   help="Override checkpoint save directory")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--epochs",       type=int, default=None,
                   help="Override num_epochs (default: 40 from TrainCompressedConfig)")
    # KD knobs (v2 defaults: lower belief weight + context alignment to prevent collapse)
    p.add_argument("--w_kd_belief",  type=float, default=0.5,
                   help="Weight for KL belief distillation loss (v1 was 2.0, caused collapse)")
    p.add_argument("--w_kd_context", type=float, default=0.2,
                   help="Weight for context alignment loss — prevents context std collapse")
    p.add_argument("--kd_temp",      type=float, default=3.0,
                   help="Temperature for softening teacher distributions")
    p.add_argument("--kd_warmup",    type=int,   default=10,
                   help="Epochs over which KD weight ramps from 0 to 1")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = TrainCompressedConfig(seed=args.seed)
    kd_cfg = KDConfig(
        teacher_ckpt=args.teacher_ckpt,
        w_kd_belief=args.w_kd_belief,
        w_kd_context=args.w_kd_context,
        kd_temperature=args.kd_temp,
        kd_warmup_epochs=args.kd_warmup,
    )
    if args.epochs:
        cfg.num_epochs = args.epochs

    seed_everything(cfg.seed, cfg.deterministic)
    device = get_device()
    print(f"Device: {device}")

    # Data loaders
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
    ds_summary = full_dataset.summary()

    # Student model
    if args.config not in CONFIGS:
        raise ValueError(f"Unknown config '{args.config}'. Available: {list(CONFIGS.keys())}")
    student_cfg = CompressedModelConfig(
        obs_channels=ds_summary["channels"],
        obs_height=ds_summary["height"],
        obs_width=ds_summary["width"],
        num_actions=4,
        num_row_classes=ds_summary["num_row_classes"],
        num_col_classes=ds_summary["num_col_classes"],
        num_heading_classes=ds_summary["num_heading_classes"],
        action_emb_dim=16,
        use_predictive_prior=True,
        **CONFIGS[args.config],
    )
    student = WorldModelCompressed(student_cfg).to(device)

    # Teacher model (frozen)
    teacher = load_teacher(kd_cfg.teacher_ckpt, device)

    # Context projection — only instantiated when w_kd_context > 0
    ctx_proj: Optional[ContextProjection] = None
    if kd_cfg.w_kd_context > 0.0:
        ctx_proj = ContextProjection(teacher.cfg.context_dim, student_cfg.context_dim).to(device)

    save_label = f"{args.config}-KD"
    cfg.save_dir = args.save_dir or f"./checkpoints/{save_label}"

    print(f"\nStudent [{save_label}]:")
    print(f"  gru={student_cfg.gru_hidden_dim}, feat={student_cfg.encoder_feat_dim}, "
          f"ctx={student_cfg.context_dim}, depthwise={student_cfg.use_depthwise}")
    print(f"  Total params    : {student.total_param_count:,}")
    print(f"  Inference params: {student.inference_param_count:,}")
    print(f"  Inference size  : {student.model_size_mb:.3f} MB (FP32)")
    print(f"  Save dir        : {cfg.save_dir}")
    print(f"\nKD settings: w_belief={kd_cfg.w_kd_belief}  w_ctx={kd_cfg.w_kd_context}  "
          f"T={kd_cfg.kd_temperature}  warmup={kd_cfg.kd_warmup_epochs}")

    # Optimiser — include ctx_proj if active (it's not saved to student ckpt)
    trainable = list(student.parameters())
    if ctx_proj is not None:
        trainable += list(ctx_proj.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler    = torch.amp.GradScaler("cuda") if (cfg.use_amp and device.type == "cuda") else None

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "train_config.json", "w") as f:
        json.dump({**asdict(cfg), **asdict(kd_cfg)}, f, indent=2)
    with open(save_dir / "model_config.json", "w") as f:
        json.dump(asdict(student_cfg), f, indent=2)

    best_hard_loss = math.inf   # best_model.pt tracks task loss only (not KD-inflated)
    best_val_joint = -math.inf
    last_train: Dict[str, float] = {}
    last_val:   Dict[str, float] = {}
    t_start = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        t_ep  = time.time()
        cur   = get_curriculum_state(epoch, cfg)
        kl_w  = min(1.0, epoch / max(1, cfg.kl_warmup_epochs))
        kd_w  = min(1.0, epoch / max(1, kd_cfg.kd_warmup_epochs))

        print(f"\nEpoch {epoch:03d}/{cfg.num_epochs}  kl_w={kl_w:.3f}  kd_w={kd_w:.3f}")
        train_m = run_epoch_kd(
            student, teacher, ctx_proj, train_loader, optimizer, scaler,
            device, cfg, kd_cfg, cur, train=True, kl_weight=kl_w, kd_weight=kd_w,
        )
        val_m = run_epoch_kd(
            student, teacher, ctx_proj, val_loader, None, None,
            device, cfg, kd_cfg, cur, train=False, kl_weight=kl_w, kd_weight=kd_w,
        )

        print_epoch_summary("Train", train_m)
        print_epoch_summary("Val  ", val_m)
        print(f"  hard={val_m['loss_hard']:.4f} | "
              f"kd_belief={train_m['loss_kd_belief']:.4f} | "
              f"kd_ctx={train_m['loss_kd_context']:.4f} | "
              f"Epoch time: {time.time() - t_ep:.1f}s")

        last_train, last_val = train_m, val_m

        # Save best_model on hard task loss only — KD loss inflates total and
        # causes early-epoch checkpoints to win even when the model is under-trained.
        if val_m["loss_hard"] < best_hard_loss:
            best_hard_loss = val_m["loss_hard"]
            save_checkpoint(save_dir / "best_model.pt", student, optimizer, epoch,
                            cfg, student_cfg, train_m, val_m)
            print(f"  >> Saved best_model (hard_loss {best_hard_loss:.4f})")

        if val_m["filter_joint_acc"] > best_val_joint:
            best_val_joint = val_m["filter_joint_acc"]
            save_checkpoint(save_dir / "best_belief_model.pt", student, optimizer, epoch,
                            cfg, student_cfg, train_m, val_m)
            print(f"  >> Saved best_belief_model (joint {best_val_joint:.4f})")

    total_s = time.time() - t_start
    summary_out = {
        "config":             save_label,
        "base_config":        args.config,
        "gru_hidden_dim":     student_cfg.gru_hidden_dim,
        "encoder_feat_dim":   student_cfg.encoder_feat_dim,
        "context_dim":        student_cfg.context_dim,
        "use_depthwise":      student_cfg.use_depthwise,
        "total_params":       student.total_param_count,
        "inference_params":   student.inference_param_count,
        "inference_size_mb":  round(student.model_size_mb, 4),
        "teacher_ckpt":          str(kd_cfg.teacher_ckpt),
        "w_kd_belief":           kd_cfg.w_kd_belief,
        "w_kd_context":          kd_cfg.w_kd_context,
        "kd_temperature":        kd_cfg.kd_temperature,
        "best_hard_val_loss":    round(best_hard_loss, 6),
        "best_val_joint_acc":    round(best_val_joint, 6),
        "total_training_min": round(total_s / 60, 2),
        "final_train_metrics": {k: round(float(v), 6) for k, v in last_train.items()},
        "final_val_metrics":   {k: round(float(v), 6) for k, v in last_val.items()},
    }
    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    print(f"\nDone. Total: {total_s / 60:.1f} min | "
          f"best_hard_loss={best_hard_loss:.4f} | best_joint={best_val_joint:.4f}")


if __name__ == "__main__":
    main()

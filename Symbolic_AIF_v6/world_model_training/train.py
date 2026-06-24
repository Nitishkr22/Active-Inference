"""
world_model_training/train.py — VFE training loop for the RSSM world model.

Trains WorldModelRSSM on collected episodes to minimise the Variational Free Energy:
  VFE = reconstruction_rgb + reconstruction_depth + occupancy_bce + KL(posterior || prior)

Usage
-----
Step 1 — Collect data (50k steps):
    conda activate AIF
    cd /home/nitish/Documents/AIF_code/Active-Inference
    python Symbolic_AIF_v6/world_model_training/collect_data.py

Step 2 — Train:
    python Symbolic_AIF_v6/world_model_training/train.py

Step 3 — Resume from checkpoint:
    python Symbolic_AIF_v6/world_model_training/train.py --resume checkpoints/epoch_010.pt

Checkpoints are saved every cfg.ckpt_every_epochs epochs in world_model_training/checkpoints/.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_HERE    = os.path.dirname(os.path.abspath(__file__))
_V6_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _V6_ROOT)

from world_model_training.config import TrainConfig
from world_model_training.world_model import WorldModelRSSM
from world_model_training.collect_data import EpisodeDataset


def _make_lr_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    """Linear warmup → cosine annealing."""
    warmup = cfg.lr_warmup_steps
    total  = cfg.num_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(resume_ckpt: Optional[str] = None):
    cfg    = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ── dataset ────────────────────────────────────────────────────────────────
    train_ds = EpisodeDataset(cfg.data_dir, seq_len=cfg.seq_len, split="train",
                               val_split=cfg.val_split)
    val_ds   = EpisodeDataset(cfg.data_dir, seq_len=cfg.seq_len, split="val",
                               val_split=cfg.val_split)

    if len(train_ds) == 0:
        print("[ERROR] No training data found.")
        print(f"  Run collect_data.py first to generate episodes in: {cfg.data_dir}")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
    )

    print(f"Device      : {device}")
    print(f"Train seqs  : {len(train_ds)}  ({len(train_loader)} batches/epoch)")
    print(f"Val seqs    : {len(val_ds)}   ({len(val_loader)} batches/epoch)")

    # ── model ──────────────────────────────────────────────────────────────────
    model     = WorldModelRSSM(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = _make_lr_scheduler(optimizer, cfg, len(train_loader))

    print(f"Parameters  : {model.param_count():,}")

    start_epoch = 0
    global_step = 0

    # ── optional resume ────────────────────────────────────────────────────────
    ckpt_path = resume_ckpt or cfg.resume_ckpt
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from: {ckpt_path}  (epoch {start_epoch})")

    # ── training loop ──────────────────────────────────────────────────────────
    log_path = os.path.join(cfg.ckpt_dir, "train_log.txt")
    log_f    = open(log_path, "a", buffering=1)

    def log(msg: str):
        print(msg)
        log_f.write(msg + "\n")

    log(f"\n{'='*60}")
    log(f"Training started — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*60}")

    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        epoch_losses = {"loss": 0.0, "loss_rgb": 0.0, "loss_depth": 0.0,
                        "loss_occ": 0.0, "loss_kl": 0.0}
        t0 = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            rgb    = batch["rgb"].to(device)           # [B, T, 3, 64, 64]
            depth  = batch["depth"].to(device)         # [B, T, 1, 64, 64]
            action = batch["action_onehot"].to(device) # [B, T, 4]
            occ_gt = batch["occ_gt"].to(device)        # [B, T, G, G]

            losses = model(rgb, depth, action, occ_gt)
            loss   = losses["loss"]

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += float(losses.get(k, 0.0))

            global_step += 1

            if global_step % cfg.log_every_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                log(
                    f"  ep {epoch:3d} step {global_step:6d} | "
                    f"loss={float(loss):.4f}  "
                    f"rgb={float(losses['loss_rgb']):.4f}  "
                    f"dep={float(losses['loss_depth']):.4f}  "
                    f"occ={float(losses['loss_occ']):.4f}  "
                    f"kl={float(losses['loss_kl']):.4f}  "
                    f"lr={lr:.2e}"
                )

        n_batches = len(train_loader)
        elapsed   = time.perf_counter() - t0
        avg = {k: v / n_batches for k, v in epoch_losses.items()}

        # ── validation ─────────────────────────────────────────────────────────
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                rgb    = batch["rgb"].to(device)
                depth  = batch["depth"].to(device)
                action = batch["action_onehot"].to(device)
                occ_gt = batch["occ_gt"].to(device)
                val_losses  = model(rgb, depth, action, occ_gt)
                val_loss_sum += float(val_losses["loss"])
        val_loss_avg = val_loss_sum / max(len(val_loader), 1)

        log(
            f"\n  EPOCH {epoch:3d}/{cfg.num_epochs-1} | "
            f"train={avg['loss']:.4f}  val={val_loss_avg:.4f}  "
            f"[rgb={avg['loss_rgb']:.3f} dep={avg['loss_depth']:.3f} "
            f"occ={avg['loss_occ']:.3f} kl={avg['loss_kl']:.3f}]  "
            f"time={elapsed:.0f}s\n"
        )

        # ── checkpoint ─────────────────────────────────────────────────────────
        if (epoch + 1) % cfg.ckpt_every_epochs == 0 or epoch == cfg.num_epochs - 1:
            ckpt_file = os.path.join(cfg.ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch":       epoch,
                "global_step": global_step,
                "model":       model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "val_loss":    val_loss_avg,
                "cfg":         cfg,
            }, ckpt_file)
            log(f"  Checkpoint → {ckpt_file}")

            # Keep symlink to best checkpoint (lowest val loss)
            best_link = os.path.join(cfg.ckpt_dir, "best.pt")
            if os.path.exists(best_link):
                prev_best = torch.load(best_link, map_location="cpu")
                if val_loss_avg < float(prev_best.get("val_loss", 1e9)):
                    os.replace(ckpt_file, best_link)
                    torch.save({"val_loss": val_loss_avg, "epoch": epoch}, best_link)
            else:
                # Hardcopy as best
                torch.save({
                    "epoch":       epoch,
                    "global_step": global_step,
                    "model":       model.state_dict(),
                    "optimizer":   optimizer.state_dict(),
                    "scheduler":   scheduler.state_dict(),
                    "val_loss":    val_loss_avg,
                    "cfg":         cfg,
                }, best_link)
                log(f"  Best model → {best_link}")

    log(f"\nTraining complete — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(resume_ckpt=args.resume or None)

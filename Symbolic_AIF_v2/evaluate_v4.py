from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset_loader import LoaderConfig, build_dataloaders
from model_v4 import ModelV4Config, WorldModelV4


# ============================================================
# Config
# ============================================================

CHECKPOINT_PATH = "./checkpoints_v4/best_model_v4.pt"
DATASET_PATH    = "../../dataset/train_dataset_v7.npz"
VAL_FRACTION    = 0.1
BATCH_SIZE      = 64
NUM_WORKERS     = 4
PIN_MEMORY      = True
SEED            = 42

ROLLOUT_HORIZONS = [1, 2, 3, 5, 10]
NUM_QUAL_SAMPLES = 4
OUTPUT_DIR       = "./eval_v4_outputs"


# ============================================================
# Helpers
# ============================================================

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def save_filter_sheet(
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    T = gt_frames.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(2.1 * T, 4.8))
    if T == 1:
        axes = np.array(axes).reshape(2, 1)
    for t in range(T):
        axes[0, t].imshow(gt_frames[t], cmap="gray", vmin=0, vmax=1)
        axes[0, t].set_title(f"GT t={t}")
        axes[0, t].axis("off")
        axes[1, t].imshow(pred_frames[t], cmap="gray", vmin=0, vmax=1)
        axes[1, t].set_title(f"Pred t={t}")
        axes[1, t].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_rollout_sheet(
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    row_true: np.ndarray,
    col_true: np.ndarray,
    heading_true: np.ndarray,
    row_pred: np.ndarray,
    col_pred: np.ndarray,
    heading_pred: np.ndarray,
    ctx_std: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    K = gt_frames.shape[0]
    fig, axes = plt.subplots(2, K, figsize=(2.2 * K, 5.2))
    if K == 1:
        axes = np.array(axes).reshape(2, 1)
    for k in range(K):
        axes[0, k].imshow(gt_frames[k], cmap="gray", vmin=0, vmax=1)
        axes[0, k].set_title(
            f"GT k={k+1}\n({row_true[k]},{col_true[k]},{heading_true[k]})"
        )
        axes[0, k].axis("off")
        axes[1, k].imshow(pred_frames[k], cmap="gray", vmin=0, vmax=1)
        axes[1, k].set_title(
            f"Pred k={k+1}\n({row_pred[k]},{col_pred[k]},{heading_pred[k]})\nstd={ctx_std[k]:.2f}"
        )
        axes[1, k].axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_fixed_rollout_targets(
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
    start_t: int,
    horizon: int,
) -> Dict[str, torch.Tensor]:
    return {
        "row_probs_start":     row_probs_seq[:, start_t, :],
        "col_probs_start":     col_probs_seq[:, start_t, :],
        "heading_probs_start": heading_probs_seq[:, start_t, :],
        "context_start":       context_seq[:, start_t, :],
        "action_roll":         actions[:, start_t:start_t + horizon],
        "row_target":          rows[:, start_t + 1:start_t + 1 + horizon],
        "col_target":          cols[:, start_t + 1:start_t + 1 + horizon],
        "heading_target":      headings[:, start_t + 1:start_t + 1 + horizon],
        "obs_target":          observations[:, start_t + 1:start_t + 1 + horizon],
        "collision_target":    collisions[:, start_t:start_t + horizon],
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    # ---- Load model ----
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model_cfg = ModelV4Config(**ckpt["model_cfg"])
    model = WorldModelV4(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint : {CHECKPOINT_PATH}")
    print(f"use_predictive_prior: {model_cfg.use_predictive_prior}")

    # ---- Dataset ----
    loader_cfg = LoaderConfig(
        dataset_path=DATASET_PATH,
        val_fraction=VAL_FRACTION,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle_train=True,
        seed=SEED,
        persistent_workers=True,
        prefetch_factor=2,
    )
    full_dataset, _, val_loader = build_dataloaders(loader_cfg)
    summary = full_dataset.summary()
    print("Dataset summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Accumulators ----
    # Filter
    filt_counts = {k: 0 for k in ["row", "col", "heading", "joint", "collision", "total", "coll_total"]}
    filt_recon_l1 = 0.0

    # V4-specific: encoder vs posterior comparison
    enc_counts = {k: 0 for k in ["row", "col", "heading", "joint"]}

    # V4-specific: uncertainty metrics
    ctx_std_sum = 0.0
    ctx_kl_sum  = 0.0
    vfe_kl_sum  = 0.0
    vfe_kl_n    = 0

    # Rollout per horizon
    rollout_stats: Dict[int, Dict[str, float]] = {
        h: {"row": 0.0, "col": 0.0, "heading": 0.0, "collision": 0.0,
            "joint": 0.0, "total": 0.0, "recon_l1": 0.0, "ctx_std": 0.0}
        for h in ROLLOUT_HORIZONS
    }

    qual_saved = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)
            observations = batch["observations"]
            actions      = batch["actions"]
            rows         = batch["rows"]
            cols         = batch["cols"]
            headings     = batch["headings"]
            collisions   = batch["collisions"].long()

            B, T = rows.shape

            filt = model.forward_filter(observations, actions)

            row_logits   = filt["row_logits_seq"]
            col_logits   = filt["col_logits_seq"]
            head_logits  = filt["heading_logits_seq"]
            row_probs    = filt["row_probs_seq"]
            col_probs    = filt["col_probs_seq"]
            head_probs   = filt["heading_probs_seq"]
            context_seq  = filt["context_seq"]
            recon_seq    = filt["recon_seq"]
            coll_logits  = filt["collision_logits_seq"]

            # ---- Filter accuracy ----
            row_pred  = row_logits.argmax(-1)
            col_pred  = col_logits.argmax(-1)
            head_pred = head_logits.argmax(-1)
            joint_ok  = (row_pred == rows) & (col_pred == cols) & (head_pred == headings)

            filt_counts["row"]      += (row_pred == rows).sum().item()
            filt_counts["col"]      += (col_pred == cols).sum().item()
            filt_counts["heading"]  += (head_pred == headings).sum().item()
            filt_counts["joint"]    += joint_ok.sum().item()
            filt_counts["total"]    += rows.numel()
            filt_recon_l1           += F.l1_loss(recon_seq, observations, reduction="sum").item()

            coll_pred = coll_logits.argmax(-1)
            filt_counts["collision"]  += (coll_pred == collisions).sum().item()
            filt_counts["coll_total"] += collisions.numel()

            # ---- V4: encoder vs posterior accuracy ----
            if model_cfg.use_predictive_prior and "encoder_row_logits_seq" in filt:
                enc_row  = filt["encoder_row_logits_seq"].argmax(-1)
                enc_col  = filt["encoder_col_logits_seq"].argmax(-1)
                enc_head = filt["encoder_heading_logits_seq"].argmax(-1)
                enc_joint = (enc_row == rows) & (enc_col == cols) & (enc_head == headings)
                enc_counts["row"]     += (enc_row == rows).sum().item()
                enc_counts["col"]     += (enc_col == cols).sum().item()
                enc_counts["heading"] += (enc_head == headings).sum().item()
                enc_counts["joint"]   += enc_joint.sum().item()

            # ---- V4: context uncertainty ----
            mu      = filt["context_mu_seq"]       # [B,T,D]
            logvar  = filt["context_logvar_seq"]    # [B,T,D]
            std     = torch.exp(0.5 * logvar)
            ctx_std_sum += std.mean().item() * B
            kl_ctx  = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
            ctx_kl_sum  += kl_ctx.item() * B

            # ---- V4: VFE KL (posterior vs transition prior) ----
            if "transition_prior_row_probs" in filt:
                eps = 1e-8
                post_row  = row_probs[:, 1:, :].detach()
                post_col  = col_probs[:, 1:, :].detach()
                post_head = head_probs[:, 1:, :].detach()
                pr_row    = filt["transition_prior_row_probs"]
                pr_col    = filt["transition_prior_col_probs"]
                pr_head   = filt["transition_prior_heading_probs"]
                kl_vfe = (
                    F.kl_div(torch.log(pr_row.clamp_min(eps)), post_row,  reduction="batchmean") +
                    F.kl_div(torch.log(pr_col.clamp_min(eps)), post_col,  reduction="batchmean") +
                    F.kl_div(torch.log(pr_head.clamp_min(eps)), post_head, reduction="batchmean")
                )
                vfe_kl_sum += kl_vfe.item() * B
                vfe_kl_n   += B

            # ---- Rollout evaluation ----
            max_h   = max(ROLLOUT_HORIZONS)
            start_t = min(10, T - max_h - 1)

            for horizon in ROLLOUT_HORIZONS:
                tgt = build_fixed_rollout_targets(
                    row_probs, col_probs, head_probs, context_seq,
                    observations, actions, rows, cols, headings, collisions,
                    start_t=start_t, horizon=horizon,
                )
                roll = model.rollout_from_filtered_state(
                    row_probs_start=tgt["row_probs_start"],
                    col_probs_start=tgt["col_probs_start"],
                    heading_probs_start=tgt["heading_probs_start"],
                    context_start=tgt["context_start"],
                    action_seq=tgt["action_roll"],
                )

                rp = roll["row_logits_roll"].argmax(-1)
                cp = roll["col_logits_roll"].argmax(-1)
                hp = roll["heading_logits_roll"].argmax(-1)
                clp = roll["collision_logits_roll"].argmax(-1)
                jp = (rp == tgt["row_target"]) & (cp == tgt["col_target"]) & (hp == tgt["heading_target"])

                s = rollout_stats[horizon]
                n = tgt["row_target"].numel()
                s["row"]       += (rp == tgt["row_target"]).sum().item()
                s["col"]       += (cp == tgt["col_target"]).sum().item()
                s["heading"]   += (hp == tgt["heading_target"]).sum().item()
                s["joint"]     += jp.sum().item()
                s["collision"] += (clp == tgt["collision_target"]).sum().item()
                s["total"]     += n
                s["recon_l1"]  += F.l1_loss(roll["recon_roll"], tgt["obs_target"], reduction="sum").item()

                # Context std across rollout steps
                if "context_roll" in roll:
                    # We don't have logvar for rollout (transition model is deterministic there)
                    # Use mean abs value of context as proxy
                    s["ctx_std"] += roll["context_roll"].abs().mean().item() * B

            # ---- Qualitative outputs ----
            if qual_saved < NUM_QUAL_SAMPLES:
                n_save = min(NUM_QUAL_SAMPLES - qual_saved, B)
                for i in range(n_save):
                    idx = qual_saved + i

                    # Filter sheet
                    save_filter_sheet(
                        gt_frames=observations[i, :, 0].cpu().numpy(),
                        pred_frames=recon_seq[i, :, 0].cpu().numpy(),
                        out_path=output_dir / f"sample_{idx:03d}_filter_sheet.png",
                        title=f"V4 filtered reconstruction – sample {idx}",
                    )

                    # Rollout sheet (max horizon)
                    tgt_q = build_fixed_rollout_targets(
                        row_probs[i:i+1], col_probs[i:i+1], head_probs[i:i+1],
                        context_seq[i:i+1], observations[i:i+1], actions[i:i+1],
                        rows[i:i+1], cols[i:i+1], headings[i:i+1], collisions[i:i+1],
                        start_t=start_t, horizon=max_h,
                    )
                    roll_q = model.rollout_from_filtered_state(
                        row_probs_start=tgt_q["row_probs_start"],
                        col_probs_start=tgt_q["col_probs_start"],
                        heading_probs_start=tgt_q["heading_probs_start"],
                        context_start=tgt_q["context_start"],
                        action_seq=tgt_q["action_roll"],
                    )

                    # Context std from rollout (use context mean abs as proxy)
                    ctx_s = roll_q["context_roll"][0].abs().mean(-1).cpu().numpy()

                    save_rollout_sheet(
                        gt_frames=tgt_q["obs_target"][0, :, 0].cpu().numpy(),
                        pred_frames=roll_q["recon_roll"][0, :, 0].cpu().numpy(),
                        row_true=tgt_q["row_target"][0].cpu().numpy(),
                        col_true=tgt_q["col_target"][0].cpu().numpy(),
                        heading_true=tgt_q["heading_target"][0].cpu().numpy(),
                        row_pred=roll_q["row_logits_roll"][0].argmax(-1).cpu().numpy(),
                        col_pred=roll_q["col_logits_roll"][0].argmax(-1).cpu().numpy(),
                        heading_pred=roll_q["heading_logits_roll"][0].argmax(-1).cpu().numpy(),
                        ctx_std=ctx_s,
                        out_path=output_dir / f"sample_{idx:03d}_rollout_sheet.png",
                        title=f"V4 rollout – sample {idx} | start_t={start_t} | horizon={max_h}",
                    )

                qual_saved += n_save

    # ============================================================
    # Print & save results
    # ============================================================
    total = filt_counts["total"]
    px = summary["height"] * summary["width"]
    n_batches = len(val_loader)

    filter_metrics = {
        "filter_row_acc":       filt_counts["row"] / total,
        "filter_col_acc":       filt_counts["col"] / total,
        "filter_heading_acc":   filt_counts["heading"] / total,
        "filter_joint_acc":     filt_counts["joint"] / total,
        "filter_collision_acc": filt_counts["collision"] / filt_counts["coll_total"],
        "filter_recon_l1":      filt_recon_l1 / (total * px),
    }

    v4_metrics: Dict[str, float] = {
        "mean_context_std":  ctx_std_sum / max(total // summary["seq_len"], 1),
        "mean_context_kl":   ctx_kl_sum  / max(total // summary["seq_len"], 1),
        "mean_vfe_kl":       vfe_kl_sum  / max(vfe_kl_n, 1),
    }
    if model_cfg.use_predictive_prior and enc_counts["joint"] > 0:
        v4_metrics["encoder_only_joint_acc"]   = enc_counts["joint"]   / total
        v4_metrics["encoder_only_row_acc"]     = enc_counts["row"]     / total
        v4_metrics["encoder_only_col_acc"]     = enc_counts["col"]     / total
        v4_metrics["encoder_only_heading_acc"] = enc_counts["heading"] / total
        v4_metrics["posterior_gain_joint_acc"] = (
            filter_metrics["filter_joint_acc"] - v4_metrics["encoder_only_joint_acc"]
        )

    rollout_metrics: Dict[str, Dict[str, float]] = {}
    for h in ROLLOUT_HORIZONS:
        s = rollout_stats[h]
        n = s["total"]
        rollout_metrics[h] = {
            "row_acc":       s["row"] / n,
            "col_acc":       s["col"] / n,
            "heading_acc":   s["heading"] / n,
            "joint_acc":     s["joint"] / n,
            "collision_acc": s["collision"] / n,
            "recon_l1":      s["recon_l1"] / (n * px),
        }

    # ---- Console output ----
    print("\n" + "=" * 70)
    print("V4 OFFLINE EVALUATION")
    print("=" * 70)

    print("\nFiltered-state metrics:")
    for k, v in filter_metrics.items():
        print(f"  {k:<30}: {v:.4f}")

    print("\nV4 uncertainty metrics:")
    for k, v in v4_metrics.items():
        print(f"  {k:<35}: {v:.4f}")

    print("\nRollout metrics:")
    print(f"  {'horizon':<6} | {'row':>6} | {'col':>6} | {'head':>6} | {'joint':>6} | {'coll':>6} | {'recon_l1':>9}")
    print("  " + "-" * 60)
    for h in ROLLOUT_HORIZONS:
        m = rollout_metrics[h]
        print(
            f"  {h:<6} | {m['row_acc']:6.4f} | {m['col_acc']:6.4f} | "
            f"{m['heading_acc']:6.4f} | {m['joint_acc']:6.4f} | "
            f"{m['collision_acc']:6.4f} | {m['recon_l1']:9.5f}"
        )

    # ---- Save JSON ----
    results = {
        "checkpoint": CHECKPOINT_PATH,
        "use_predictive_prior": model_cfg.use_predictive_prior,
        "filter": filter_metrics,
        "v4_uncertainty": v4_metrics,
        "rollout": {str(h): rollout_metrics[h] for h in ROLLOUT_HORIZONS},
    }
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results  : {results_path}")
    print(f"Saved images   : {output_dir}/")


if __name__ == "__main__":
    main()

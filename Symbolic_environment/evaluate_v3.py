from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset_loader import LoaderConfig, build_dataloaders
from model_v3 import ModelV3Config, WorldModelV3


# ============================================================
# Config
# ============================================================

CHECKPOINT_PATH = "./checkpoints_v3/best_model.pt"
DATASET_PATH = "../../dataset/train_dataset_v7.npz"
VAL_FRACTION = 0.1
BATCH_SIZE = 64
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2
SEED = 42

ROLLOUT_HORIZONS = [1, 2, 3, 5, 10]
NUM_QUAL_SAMPLES = 4
OUTPUT_DIR = "./eval_v3_outputs"


# ============================================================
# Helpers
# ============================================================

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out



def compute_collision_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == targets.long()).float().mean().item()



def save_contact_sheet(
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
        axes[0, t].imshow(gt_frames[t], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, t].set_title(f"GT t={t}")
        axes[0, t].axis("off")

        axes[1, t].imshow(pred_frames[t], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, t].set_title(f"Pred t={t}")
        axes[1, t].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def save_rollout_contact_sheet(
    gt_frames: np.ndarray,
    pred_frames: np.ndarray,
    row_true: np.ndarray,
    col_true: np.ndarray,
    heading_true: np.ndarray,
    row_pred: np.ndarray,
    col_pred: np.ndarray,
    heading_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    K = gt_frames.shape[0]
    fig, axes = plt.subplots(2, K, figsize=(2.2 * K, 5.0))
    if K == 1:
        axes = np.array(axes).reshape(2, 1)

    for k in range(K):
        axes[0, k].imshow(gt_frames[k], cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, k].set_title(f"GT k={k+1}\n({row_true[k]},{col_true[k]},{heading_true[k]})")
        axes[0, k].axis("off")

        axes[1, k].imshow(pred_frames[k], cmap="gray", vmin=0.0, vmax=1.0)
        axes[1, k].set_title(f"Pred k={k+1}\n({row_pred[k]},{col_pred[k]},{heading_pred[k]})")
        axes[1, k].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Deterministic rollout target helper
# ============================================================

def build_fixed_rollout_targets(
    row_probs_seq: torch.Tensor,      # [B,T,R]
    col_probs_seq: torch.Tensor,      # [B,T,C]
    heading_probs_seq: torch.Tensor,  # [B,T,Hd]
    context_seq: torch.Tensor,        # [B,T,U]
    observations: torch.Tensor,       # [B,T,1,H,W]
    actions: torch.Tensor,            # [B,T-1]
    rows: torch.Tensor,               # [B,T]
    cols: torch.Tensor,               # [B,T]
    headings: torch.Tensor,           # [B,T]
    collisions: torch.Tensor,         # [B,T-1]
    start_t: int,
    horizon: int,
) -> Dict[str, torch.Tensor]:
    B, T, _ = row_probs_seq.shape
    if start_t + horizon >= T:
        raise ValueError(f"Invalid start_t={start_t}, horizon={horizon}, T={T}")

    return {
        "row_probs_start": row_probs_seq[:, start_t, :],
        "col_probs_start": col_probs_seq[:, start_t, :],
        "heading_probs_start": heading_probs_seq[:, start_t, :],
        "context_start": context_seq[:, start_t, :],
        "action_roll": actions[:, start_t:start_t + horizon],
        "row_target": rows[:, start_t + 1:start_t + 1 + horizon],
        "col_target": cols[:, start_t + 1:start_t + 1 + horizon],
        "heading_target": headings[:, start_t + 1:start_t + 1 + horizon],
        "obs_target": observations[:, start_t + 1:start_t + 1 + horizon],
        "collision_target": collisions[:, start_t:start_t + horizon],
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model_cfg = ModelV3Config(**ckpt["model_cfg"])
    model = WorldModelV3(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    loader_cfg = LoaderConfig(
        dataset_path=DATASET_PATH,
        val_fraction=VAL_FRACTION,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle_train=True,
        seed=SEED,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        drop_last_train=False,
        drop_last_val=False,
    )
    full_dataset, _, val_loader = build_dataloaders(loader_cfg)
    summary = full_dataset.summary()

    print("Dataset summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregates
    filtered_row_correct = 0
    filtered_col_correct = 0
    filtered_heading_correct = 0
    filtered_total = 0
    filtered_recon_l1_sum = 0.0
    filtered_collision_correct = 0
    filtered_collision_total = 0

    rollout_stats: Dict[int, Dict[str, float]] = {}
    for h in ROLLOUT_HORIZONS:
        rollout_stats[h] = {
            "row_correct": 0.0,
            "col_correct": 0.0,
            "heading_correct": 0.0,
            "collision_correct": 0.0,
            "total": 0.0,
            "recon_l1_sum": 0.0,
        }

    qual_saved = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = move_batch_to_device(batch, device)

            observations = batch["observations"]
            actions = batch["actions"]
            rows = batch["rows"]
            cols = batch["cols"]
            headings = batch["headings"]
            collisions = batch["collisions"].long()

            B, T = rows.shape

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

            # Filter metrics
            row_pred = row_logits_seq.argmax(dim=-1)
            col_pred = col_logits_seq.argmax(dim=-1)
            heading_pred = heading_logits_seq.argmax(dim=-1)

            filtered_row_correct += (row_pred == rows).sum().item()
            filtered_col_correct += (col_pred == cols).sum().item()
            filtered_heading_correct += (heading_pred == headings).sum().item()
            filtered_total += rows.numel()

            filtered_recon_l1_sum += F.l1_loss(recon_seq, observations, reduction="sum").item()
            filtered_collision_pred = collision_logits_seq.argmax(dim=-1)
            filtered_collision_correct += (filtered_collision_pred == collisions).sum().item()
            filtered_collision_total += collisions.numel()

            # Deterministic rollout evaluation from fixed t
            max_h = max(ROLLOUT_HORIZONS)
            start_t = min(10, T - max_h - 1)
            if start_t < 0:
                raise ValueError("Sequence too short for requested rollout horizons.")

            for horizon in ROLLOUT_HORIZONS:
                targets = build_fixed_rollout_targets(
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
                    start_t=start_t,
                    horizon=horizon,
                )

                roll = model.rollout_from_filtered_state(
                    row_probs_start=targets["row_probs_start"],
                    col_probs_start=targets["col_probs_start"],
                    heading_probs_start=targets["heading_probs_start"],
                    context_start=targets["context_start"],
                    action_seq=targets["action_roll"],
                )

                row_pred_roll = roll["row_logits_roll"].argmax(dim=-1)
                col_pred_roll = roll["col_logits_roll"].argmax(dim=-1)
                heading_pred_roll = roll["heading_logits_roll"].argmax(dim=-1)
                collision_pred_roll = roll["collision_logits_roll"].argmax(dim=-1)

                rollout_stats[horizon]["row_correct"] += (row_pred_roll == targets["row_target"]).sum().item()
                rollout_stats[horizon]["col_correct"] += (col_pred_roll == targets["col_target"]).sum().item()
                rollout_stats[horizon]["heading_correct"] += (heading_pred_roll == targets["heading_target"]).sum().item()
                rollout_stats[horizon]["collision_correct"] += (collision_pred_roll == targets["collision_target"]).sum().item()
                rollout_stats[horizon]["total"] += targets["row_target"].numel()
                rollout_stats[horizon]["recon_l1_sum"] += F.l1_loss(
                    roll["recon_roll"], targets["obs_target"], reduction="sum"
                ).item()

            # Qualitative outputs
            if qual_saved < NUM_QUAL_SAMPLES:
                num_to_save = min(NUM_QUAL_SAMPLES - qual_saved, B)
                for i in range(num_to_save):
                    sample_idx = qual_saved + i

                    gt_f = observations[i, :, 0].detach().cpu().numpy()
                    pr_f = recon_seq[i, :, 0].detach().cpu().numpy()
                    save_contact_sheet(
                        gt_frames=gt_f,
                        pred_frames=pr_f,
                        out_path=output_dir / f"sample_{sample_idx:03d}_filter_sheet.png",
                        title=f"V3 filtered reconstruction sample {sample_idx}",
                    )

                    targets = build_fixed_rollout_targets(
                        row_probs_seq=row_probs_seq[i:i+1],
                        col_probs_seq=col_probs_seq[i:i+1],
                        heading_probs_seq=heading_probs_seq[i:i+1],
                        context_seq=context_seq[i:i+1],
                        observations=observations[i:i+1],
                        actions=actions[i:i+1],
                        rows=rows[i:i+1],
                        cols=cols[i:i+1],
                        headings=headings[i:i+1],
                        collisions=collisions[i:i+1],
                        start_t=start_t,
                        horizon=max_h,
                    )
                    roll = model.rollout_from_filtered_state(
                        row_probs_start=targets["row_probs_start"],
                        col_probs_start=targets["col_probs_start"],
                        heading_probs_start=targets["heading_probs_start"],
                        context_start=targets["context_start"],
                        action_seq=targets["action_roll"],
                    )

                    gt_r = targets["obs_target"][0, :, 0].detach().cpu().numpy()
                    pr_r = roll["recon_roll"][0, :, 0].detach().cpu().numpy()
                    row_true = targets["row_target"][0].detach().cpu().numpy()
                    col_true = targets["col_target"][0].detach().cpu().numpy()
                    heading_true = targets["heading_target"][0].detach().cpu().numpy()
                    row_pred_r = roll["row_logits_roll"][0].argmax(dim=-1).detach().cpu().numpy()
                    col_pred_r = roll["col_logits_roll"][0].argmax(dim=-1).detach().cpu().numpy()
                    heading_pred_r = roll["heading_logits_roll"][0].argmax(dim=-1).detach().cpu().numpy()

                    save_rollout_contact_sheet(
                        gt_frames=gt_r,
                        pred_frames=pr_r,
                        row_true=row_true,
                        col_true=col_true,
                        heading_true=heading_true,
                        row_pred=row_pred_r,
                        col_pred=col_pred_r,
                        heading_pred=heading_pred_r,
                        out_path=output_dir / f"sample_{sample_idx:03d}_rollout_sheet.png",
                        title=f"V3 rollout sample {sample_idx} | start_t={start_t}",
                    )

                qual_saved += num_to_save

    # Final metrics
    filtered_row_acc = filtered_row_correct / filtered_total
    filtered_col_acc = filtered_col_correct / filtered_total
    filtered_heading_acc = filtered_heading_correct / filtered_total
    filtered_recon_l1 = filtered_recon_l1_sum / (filtered_total * summary["height"] * summary["width"])
    filtered_collision_acc = filtered_collision_correct / filtered_collision_total

    print("\nFiltered-state metrics:")
    print(f"  filtered_row_acc:       {filtered_row_acc:.4f}")
    print(f"  filtered_col_acc:       {filtered_col_acc:.4f}")
    print(f"  filtered_heading_acc:   {filtered_heading_acc:.4f}")
    print(f"  filtered_recon_l1:      {filtered_recon_l1:.4f}")
    print(f"  filtered_collision_acc: {filtered_collision_acc:.4f}")

    print("\nRollout metrics:")
    for horizon in ROLLOUT_HORIZONS:
        total = rollout_stats[horizon]["total"]
        row_acc = rollout_stats[horizon]["row_correct"] / total
        col_acc = rollout_stats[horizon]["col_correct"] / total
        heading_acc = rollout_stats[horizon]["heading_correct"] / total
        collision_acc = rollout_stats[horizon]["collision_correct"] / total
        recon_l1 = rollout_stats[horizon]["recon_l1_sum"] / (total * summary["height"] * summary["width"])

        print(f"  rollout_row_acc_h{horizon}:        {row_acc:.4f}")
        print(f"  rollout_col_acc_h{horizon}:        {col_acc:.4f}")
        print(f"  rollout_heading_acc_h{horizon}:    {heading_acc:.4f}")
        print(f"  rollout_collision_acc_h{horizon}:  {collision_acc:.4f}")
        print(f"  rollout_recon_l1_h{horizon}:       {recon_l1:.4f}")

    print(f"\nSaved qualitative outputs to: {output_dir}")


if __name__ == "__main__":
    main()

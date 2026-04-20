# evaluate_v1.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset_loader import LoaderConfig, create_dataloaders
from model_v1 import ModelV1Config, WorldModelV1


# ============================================================
# Config
# ============================================================

@dataclass
class EvalConfig:
    dataset_path: str = "../../dataset/train_dataset__v6.npz"
    checkpoint_path: str = "./checkpoints_v1/best_model.pt"
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    val_fraction: float = 0.1
    seed: int = 42

    max_eval_batches: int = -1          # -1 means evaluate whole validation set
    rollout_horizons: Tuple[int, ...] = (1, 2, 3, 5)

    num_visual_examples: int = 3
    visual_output_dir: str = "./eval_v1_outputs"


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Checkpoint loading
# ============================================================

def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_cfg: ModelV1Config,
) -> WorldModelV1:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = WorldModelV1(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ============================================================
# Accuracy helpers
# ============================================================

def classification_accuracy_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    logits: [B,T,C] or [B,K,C]
    targets: [B,T] or [B,K]
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float().mean().item()
    return float(correct)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# ============================================================
# Filtered-state evaluation
# ============================================================

@torch.no_grad()
def evaluate_filtered_pose_accuracy(
    model: WorldModelV1,
    val_loader,
    device: torch.device,
    max_eval_batches: int = -1,
) -> Dict[str, float]:
    model.eval()

    row_acc_sum = 0.0
    col_acc_sum = 0.0
    heading_acc_sum = 0.0
    recon_l1_sum = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break

        batch = move_batch_to_device(batch, device)

        observations = batch["observations"]   # [B,T,1,H,W]
        actions = batch["actions"]             # [B,T-1]
        rows = batch["rows"]                   # [B,T]
        cols = batch["cols"]                   # [B,T]
        headings = batch["headings"]           # [B,T]

        out = model.forward_filter(observations, actions)

        row_acc = classification_accuracy_from_logits(out["row_logits_seq"], rows)
        col_acc = classification_accuracy_from_logits(out["col_logits_seq"], cols)
        heading_acc = classification_accuracy_from_logits(out["heading_logits_seq"], headings)
        recon_l1 = F.l1_loss(out["recon_seq"], observations).item()

        row_acc_sum += row_acc
        col_acc_sum += col_acc
        heading_acc_sum += heading_acc
        recon_l1_sum += recon_l1
        num_batches += 1

    return {
        "filtered_row_acc": row_acc_sum / max(num_batches, 1),
        "filtered_col_acc": col_acc_sum / max(num_batches, 1),
        "filtered_heading_acc": heading_acc_sum / max(num_batches, 1),
        "filtered_recon_l1": recon_l1_sum / max(num_batches, 1),
    }


# ============================================================
# Rollout evaluation
# ============================================================

def sample_fixed_rollout_targets(
    z_seq: torch.Tensor,
    h_seq: torch.Tensor,
    actions: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    headings: torch.Tensor,
    horizon: int,
    start_t: int,
) -> Dict[str, torch.Tensor]:
    """
    Fixed rollout start for evaluation.

    Inputs:
      z_seq,h_seq : [B,T,*]
      actions     : [B,T-1]
      rows,cols,headings : [B,T]
      horizon     : K
      start_t     : rollout begins at time t, predicting t+1...t+K
    """
    B, T, Z = z_seq.shape
    K = horizon

    if start_t + K > T - 1:
        raise ValueError(
            f"Invalid rollout request: start_t={start_t}, horizon={K}, T={T}"
        )

    z_start = z_seq[:, start_t]                         # [B,Z]
    h_start = h_seq[:, start_t]                         # [B,H]
    action_roll = actions[:, start_t:start_t + K]       # [B,K]

    row_target = rows[:, start_t + 1:start_t + 1 + K]
    col_target = cols[:, start_t + 1:start_t + 1 + K]
    heading_target = headings[:, start_t + 1:start_t + 1 + K]

    return {
        "z_start": z_start,
        "h_start": h_start,
        "action_roll": action_roll,
        "row_target": row_target,
        "col_target": col_target,
        "heading_target": heading_target,
    }


@torch.no_grad()
def evaluate_rollout_pose_accuracy(
    model: WorldModelV1,
    val_loader,
    device: torch.device,
    rollout_horizons: Tuple[int, ...],
    max_eval_batches: int = -1,
) -> Dict[str, float]:
    model.eval()

    results: Dict[str, float] = {}

    # choose a fixed safe rollout start
    # since T=32 and horizons up to 5, start_t=8 is fine
    start_t = 8

    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}

    for horizon in rollout_horizons:
        for key in [f"rollout_row_acc_h{horizon}",
                    f"rollout_col_acc_h{horizon}",
                    f"rollout_heading_acc_h{horizon}"]:
            metric_sums[key] = 0.0
            metric_counts[key] = 0

    for batch_idx, batch in enumerate(val_loader):
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break

        batch = move_batch_to_device(batch, device)

        observations = batch["observations"]
        actions = batch["actions"]
        rows = batch["rows"]
        cols = batch["cols"]
        headings = batch["headings"]

        filt = model.forward_filter(observations, actions)
        z_seq = filt["z_seq"]
        h_seq = filt["h_seq"]

        for horizon in rollout_horizons:
            tgt = sample_fixed_rollout_targets(
                z_seq=z_seq,
                h_seq=h_seq,
                actions=actions,
                rows=rows,
                cols=cols,
                headings=headings,
                horizon=horizon,
                start_t=start_t,
            )

            roll = model.rollout_from_filtered_state(
                z_start=tgt["z_start"],
                h_start=tgt["h_start"],
                action_seq=tgt["action_roll"],
            )

            row_acc = classification_accuracy_from_logits(
                roll["row_logits_roll"], tgt["row_target"]
            )
            col_acc = classification_accuracy_from_logits(
                roll["col_logits_roll"], tgt["col_target"]
            )
            heading_acc = classification_accuracy_from_logits(
                roll["heading_logits_roll"], tgt["heading_target"]
            )

            metric_sums[f"rollout_row_acc_h{horizon}"] += row_acc
            metric_sums[f"rollout_col_acc_h{horizon}"] += col_acc
            metric_sums[f"rollout_heading_acc_h{horizon}"] += heading_acc

            metric_counts[f"rollout_row_acc_h{horizon}"] += 1
            metric_counts[f"rollout_col_acc_h{horizon}"] += 1
            metric_counts[f"rollout_heading_acc_h{horizon}"] += 1

    for key in metric_sums:
        results[key] = metric_sums[key] / max(metric_counts[key], 1)

    return results


# ============================================================
# Visualizations
# ============================================================

def save_image_grid(
    images: List[torch.Tensor],
    titles: List[str],
    save_path: str,
) -> None:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        img_np = img.detach().cpu().squeeze().numpy()
        ax.imshow(img_np, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_visual_examples(
    model: WorldModelV1,
    val_loader,
    device: torch.device,
    output_dir: str,
    num_examples: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    saved = 0
    start_t = 8
    horizon = 5

    for batch in val_loader:
        batch = move_batch_to_device(batch, device)

        observations = batch["observations"]
        actions = batch["actions"]
        rows = batch["rows"]
        cols = batch["cols"]
        headings = batch["headings"]

        filt = model.forward_filter(observations, actions)

        B = observations.shape[0]
        for b in range(B):
            if saved >= num_examples:
                return

            # ------------------------------------------------
            # Reconstruction example
            # ------------------------------------------------
            t_vis = 10
            true_img = observations[b, t_vis]
            recon_img = filt["recon_seq"][b, t_vis]

            save_image_grid(
                images=[true_img, recon_img],
                titles=[
                    f"True obs t={t_vis}",
                    f"Recon obs t={t_vis}",
                ],
                save_path=os.path.join(output_dir, f"example_{saved:03d}_recon.png"),
            )

            # ------------------------------------------------
            # Rollout example
            # ------------------------------------------------
            z_start = filt["z_seq"][b:b+1, start_t]                    # [1,Z]
            h_start = filt["h_seq"][b:b+1, start_t]                    # [1,H]
            action_roll = actions[b:b+1, start_t:start_t + horizon]    # [1,K]

            roll = model.rollout_from_filtered_state(
                z_start=z_start,
                h_start=h_start,
                action_seq=action_roll,
            )

            # Save one sheet comparing true future vs rollout future
            true_future = [observations[b, start_t + 1 + k] for k in range(horizon)]
            pred_future = [roll["recon_roll"][0, k] for k in range(horizon)]

            images = []
            titles = []
            for k in range(horizon):
                images.append(true_future[k])
                titles.append(f"True t+{k+1}")
                images.append(pred_future[k])
                titles.append(f"Pred t+{k+1}")

            save_image_grid(
                images=images,
                titles=titles,
                save_path=os.path.join(output_dir, f"example_{saved:03d}_rollout.png"),
            )

            saved += 1


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = EvalConfig()
    device = get_device()
    print(f"Using device: {device}")

    loader_cfg = LoaderConfig(
        dataset_path=cfg.dataset_path,
        val_fraction=cfg.val_fraction,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
        shuffle_train=False,
        seed=cfg.seed,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=2,
        drop_last_train=False,
        drop_last_val=False,
    )

    full_dataset, train_loader, val_loader = create_dataloaders(loader_cfg)
    summary = full_dataset.summary()

    print("Dataset summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

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

    model = load_model_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        device=device,
        model_cfg=model_cfg,
    )

    print(f"Loaded checkpoint: {cfg.checkpoint_path}")

    filt_metrics = evaluate_filtered_pose_accuracy(
        model=model,
        val_loader=val_loader,
        device=device,
        max_eval_batches=cfg.max_eval_batches,
    )

    roll_metrics = evaluate_rollout_pose_accuracy(
        model=model,
        val_loader=val_loader,
        device=device,
        rollout_horizons=cfg.rollout_horizons,
        max_eval_batches=cfg.max_eval_batches,
    )

    print("\nFiltered-state metrics:")
    for k, v in filt_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\nRollout metrics:")
    for k, v in roll_metrics.items():
        print(f"  {k}: {v:.4f}")

    save_visual_examples(
        model=model,
        val_loader=val_loader,
        device=device,
        output_dir=cfg.visual_output_dir,
        num_examples=cfg.num_visual_examples,
    )

    print(f"\nSaved qualitative outputs to: {cfg.visual_output_dir}")


if __name__ == "__main__":
    main()
# debug_v1_rollout.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch

from dataset_loader import SequenceWorldModelDataset
from model_v1 import ModelV1Config, WorldModelV1


# ============================================================
# Config
# ============================================================

@dataclass
class DebugConfig:
    dataset_path: str = "../../dataset/train_dataset__v6.npz"
    checkpoint_path: str = "./checkpoints_v1/best_model.pt"

    episode_idx: int = 0
    history_t: int = 10          # use observations up to this index for belief state
    rollout_horizon: int = 5

    output_dir: str = "./debug_v1_outputs"


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Loading helpers
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


def load_single_episode(
    dataset_path: str,
    episode_idx: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    dataset = SequenceWorldModelDataset(dataset_path)
    sample = dataset[episode_idx]

    # add batch dimension => [1, ...]
    batch = {k: v.unsqueeze(0).to(device) for k, v in sample.items()}
    return batch, dataset


# ============================================================
# Accuracy / decoding helpers
# ============================================================

HEADING_NAMES = ["N", "E", "S", "W"]


def decode_heading(idx: int) -> str:
    return HEADING_NAMES[idx]


def argmax_seq(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B,T,C] or [B,K,C]
    returns: [B,T] or [B,K]
    """
    return logits.argmax(dim=-1)


# ============================================================
# Printing helpers
# ============================================================

def print_filter_debug(
    rows_true: torch.Tensor,
    cols_true: torch.Tensor,
    headings_true: torch.Tensor,
    row_pred: torch.Tensor,
    col_pred: torch.Tensor,
    heading_pred: torch.Tensor,
) -> None:
    """
    All tensors shape: [T]
    """
    T = rows_true.shape[0]

    print("\n" + "=" * 110)
    print("FILTER / BELIEF DEBUG")
    print("=" * 110)
    print("t | true_pose        | pred_pose        | row_ok col_ok heading_ok")
    print("-" * 110)

    row_correct = 0
    col_correct = 0
    heading_correct = 0

    for t in range(T):
        rt = int(rows_true[t].item())
        ct = int(cols_true[t].item())
        ht = int(headings_true[t].item())

        rp = int(row_pred[t].item())
        cp = int(col_pred[t].item())
        hp = int(heading_pred[t].item())

        row_ok = int(rt == rp)
        col_ok = int(ct == cp)
        heading_ok = int(ht == hp)

        row_correct += row_ok
        col_correct += col_ok
        heading_correct += heading_ok

        print(
            f"{t:2d} | "
            f"({rt:2d},{ct:2d},{decode_heading(ht):>1s})           | "
            f"({rp:2d},{cp:2d},{decode_heading(hp):>1s})           | "
            f"{row_ok:^6d} {col_ok:^6d} {heading_ok:^10d}"
        )

    print("-" * 110)
    print(f"Row accuracy over episode:     {row_correct / T:.4f}")
    print(f"Col accuracy over episode:     {col_correct / T:.4f}")
    print(f"Heading accuracy over episode: {heading_correct / T:.4f}")
    print("=" * 110)


def print_rollout_debug(
    history_t: int,
    rows_true_future: torch.Tensor,
    cols_true_future: torch.Tensor,
    headings_true_future: torch.Tensor,
    row_pred_future: torch.Tensor,
    col_pred_future: torch.Tensor,
    heading_pred_future: torch.Tensor,
    actions_future: torch.Tensor,
    action_names: List[str],
) -> None:
    """
    All tensors shape: [K]
    """
    K = rows_true_future.shape[0]

    print("\n" + "=" * 120)
    print(f"ROLLOUT / TRANSITION DEBUG (starting from filtered state at t={history_t})")
    print("=" * 120)
    print("k | action      | true_future_pose   | pred_future_pose   | row_ok col_ok heading_ok")
    print("-" * 120)

    row_correct = 0
    col_correct = 0
    heading_correct = 0

    for k in range(K):
        a = int(actions_future[k].item())
        a_name = action_names[a]

        rt = int(rows_true_future[k].item())
        ct = int(cols_true_future[k].item())
        ht = int(headings_true_future[k].item())

        rp = int(row_pred_future[k].item())
        cp = int(col_pred_future[k].item())
        hp = int(heading_pred_future[k].item())

        row_ok = int(rt == rp)
        col_ok = int(ct == cp)
        heading_ok = int(ht == hp)

        row_correct += row_ok
        col_correct += col_ok
        heading_correct += heading_ok

        print(
            f"{k+1:2d} | "
            f"{a_name:10s} | "
            f"({rt:2d},{ct:2d},{decode_heading(ht):>1s})            | "
            f"({rp:2d},{cp:2d},{decode_heading(hp):>1s})            | "
            f"{row_ok:^6d} {col_ok:^6d} {heading_ok:^10d}"
        )

    print("-" * 120)
    print(f"Rollout row accuracy:     {row_correct / K:.4f}")
    print(f"Rollout col accuracy:     {col_correct / K:.4f}")
    print(f"Rollout heading accuracy: {heading_correct / K:.4f}")
    print("=" * 120)


# ============================================================
# Visualization helpers
# ============================================================

def save_filter_contact_sheet(
    observations: torch.Tensor,       # [T,1,H,W]
    reconstructions: torch.Tensor,    # [T,1,H,W]
    rows_true: torch.Tensor,          # [T]
    cols_true: torch.Tensor,          # [T]
    headings_true: torch.Tensor,      # [T]
    row_pred: torch.Tensor,           # [T]
    col_pred: torch.Tensor,           # [T]
    heading_pred: torch.Tensor,       # [T]
    save_path: str,
    cols_per_row: int = 4,
) -> None:
    T = observations.shape[0]
    ncols = cols_per_row
    nrows = T

    fig, axes = plt.subplots(nrows, 2, figsize=(8, 3 * nrows))
    if T == 1:
        axes = axes[None, :]

    for t in range(T):
        true_img = observations[t].squeeze().detach().cpu().numpy()
        recon_img = reconstructions[t].squeeze().detach().cpu().numpy()

        rt = int(rows_true[t].item())
        ct = int(cols_true[t].item())
        ht = int(headings_true[t].item())

        rp = int(row_pred[t].item())
        cp = int(col_pred[t].item())
        hp = int(heading_pred[t].item())

        ax0 = axes[t, 0]
        ax0.imshow(true_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax0.set_title(f"True t={t} | ({rt},{ct},{decode_heading(ht)})", fontsize=10)
        ax0.axis("off")

        ax1 = axes[t, 1]
        ax1.imshow(recon_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax1.set_title(f"Recon t={t} | ({rp},{cp},{decode_heading(hp)})", fontsize=10)
        ax1.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_rollout_contact_sheet(
    true_future_obs: List[torch.Tensor],
    pred_future_obs: List[torch.Tensor],
    rows_true_future: torch.Tensor,
    cols_true_future: torch.Tensor,
    headings_true_future: torch.Tensor,
    row_pred_future: torch.Tensor,
    col_pred_future: torch.Tensor,
    heading_pred_future: torch.Tensor,
    actions_future: torch.Tensor,
    action_names: List[str],
    save_path: str,
) -> None:
    K = len(true_future_obs)

    fig, axes = plt.subplots(K, 2, figsize=(8, 3 * K))
    if K == 1:
        axes = axes[None, :]

    for k in range(K):
        true_img = true_future_obs[k].squeeze().detach().cpu().numpy()
        pred_img = pred_future_obs[k].squeeze().detach().cpu().numpy()

        rt = int(rows_true_future[k].item())
        ct = int(cols_true_future[k].item())
        ht = int(headings_true_future[k].item())

        rp = int(row_pred_future[k].item())
        cp = int(col_pred_future[k].item())
        hp = int(heading_pred_future[k].item())

        a = int(actions_future[k].item())
        a_name = action_names[a]

        ax0 = axes[k, 0]
        ax0.imshow(true_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax0.set_title(
            f"True t+{k+1} | action={a_name} | ({rt},{ct},{decode_heading(ht)})",
            fontsize=10,
        )
        ax0.axis("off")

        ax1 = axes[k, 1]
        ax1.imshow(pred_img, cmap="gray", vmin=0.0, vmax=1.0)
        ax1.set_title(
            f"Pred t+{k+1} | action={a_name} | ({rp},{cp},{decode_heading(hp)})",
            fontsize=10,
        )
        ax1.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main debug function
# ============================================================

@torch.no_grad()
def debug_episode(cfg: DebugConfig) -> None:
    device = get_device()
    print(f"Using device: {device}")

    batch, dataset = load_single_episode(
        dataset_path=cfg.dataset_path,
        episode_idx=cfg.episode_idx,
        device=device,
    )

    summary = dataset.summary()

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

    model = load_model_from_checkpoint(cfg.checkpoint_path, device, model_cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)

    observations = batch["observations"]    # [1,T,1,H,W]
    actions = batch["actions"]              # [1,T-1]
    rows = batch["rows"]                    # [1,T]
    cols = batch["cols"]                    # [1,T]
    headings = batch["headings"]            # [1,T]

    T = observations.shape[1]
    if not (0 <= cfg.history_t < T - 1):
        raise ValueError(f"history_t must satisfy 0 <= history_t < T-1, got {cfg.history_t} for T={T}")

    if cfg.history_t + cfg.rollout_horizon >= T:
        raise ValueError(
            f"history_t + rollout_horizon must be < T. "
            f"Got history_t={cfg.history_t}, rollout_horizon={cfg.rollout_horizon}, T={T}"
        )

    # --------------------------------------------------------
    # FILTER / BELIEF DEBUG
    # --------------------------------------------------------
    filt = model.forward_filter(observations, actions)

    row_pred = argmax_seq(filt["row_logits_seq"])[0]           # [T]
    col_pred = argmax_seq(filt["col_logits_seq"])[0]           # [T]
    heading_pred = argmax_seq(filt["heading_logits_seq"])[0]   # [T]

    print_filter_debug(
        rows_true=rows[0],
        cols_true=cols[0],
        headings_true=headings[0],
        row_pred=row_pred,
        col_pred=col_pred,
        heading_pred=heading_pred,
    )

    filter_sheet_path = os.path.join(
        cfg.output_dir, f"episode_{cfg.episode_idx:04d}_filter_debug.png"
    )
    save_filter_contact_sheet(
        observations=observations[0],
        reconstructions=filt["recon_seq"][0],
        rows_true=rows[0],
        cols_true=cols[0],
        headings_true=headings[0],
        row_pred=row_pred,
        col_pred=col_pred,
        heading_pred=heading_pred,
        save_path=filter_sheet_path,
    )
    print(f"Saved filter debug sheet to: {filter_sheet_path}")

    # --------------------------------------------------------
    # ROLLOUT / TRANSITION DEBUG
    # --------------------------------------------------------
    t0 = cfg.history_t
    K = cfg.rollout_horizon

    z_start = filt["z_seq"][:, t0]                         # [1,Z]
    h_start = filt["h_seq"][:, t0]                         # [1,H]
    action_roll = actions[:, t0:t0 + K]                   # [1,K]

    roll = model.rollout_from_filtered_state(
        z_start=z_start,
        h_start=h_start,
        action_seq=action_roll,
    )

    row_pred_future = argmax_seq(roll["row_logits_roll"])[0]           # [K]
    col_pred_future = argmax_seq(roll["col_logits_roll"])[0]           # [K]
    heading_pred_future = argmax_seq(roll["heading_logits_roll"])[0]   # [K]

    rows_true_future = rows[0, t0 + 1:t0 + 1 + K]
    cols_true_future = cols[0, t0 + 1:t0 + 1 + K]
    headings_true_future = headings[0, t0 + 1:t0 + 1 + K]
    actions_future = actions[0, t0:t0 + K]

    action_names = (
        [str(x) for x in dataset.action_names.tolist()]
        if dataset.action_names is not None
        else ["forward", "backward", "turn_left", "turn_right"]
    )

    print_rollout_debug(
        history_t=t0,
        rows_true_future=rows_true_future,
        cols_true_future=cols_true_future,
        headings_true_future=headings_true_future,
        row_pred_future=row_pred_future,
        col_pred_future=col_pred_future,
        heading_pred_future=heading_pred_future,
        actions_future=actions_future,
        action_names=action_names,
    )

    true_future_obs = [observations[0, t0 + 1 + k] for k in range(K)]
    pred_future_obs = [roll["recon_roll"][0, k] for k in range(K)]

    rollout_sheet_path = os.path.join(
        cfg.output_dir, f"episode_{cfg.episode_idx:04d}_rollout_debug_t{t0}_h{K}.png"
    )
    save_rollout_contact_sheet(
        true_future_obs=true_future_obs,
        pred_future_obs=pred_future_obs,
        rows_true_future=rows_true_future,
        cols_true_future=cols_true_future,
        headings_true_future=headings_true_future,
        row_pred_future=row_pred_future,
        col_pred_future=col_pred_future,
        heading_pred_future=heading_pred_future,
        actions_future=actions_future,
        action_names=action_names,
        save_path=rollout_sheet_path,
    )
    print(f"Saved rollout debug sheet to: {rollout_sheet_path}")

    # --------------------------------------------------------
    # Print summary around chosen start
    # --------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"SUMMARY FOR EPISODE {cfg.episode_idx}")
    print(f"History cutoff t0 = {t0}")
    print(f"Rollout horizon K = {K}")
    print(f"Filtered pose at t0 (true): "
          f"({int(rows[0, t0])}, {int(cols[0, t0])}, {decode_heading(int(headings[0, t0]))})")
    print(f"Filtered pose at t0 (pred): "
          f"({int(row_pred[t0])}, {int(col_pred[t0])}, {decode_heading(int(heading_pred[t0]))})")
    print("=" * 100)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    cfg = DebugConfig(
        dataset_path="../../dataset/train_dataset__v6.npz",
        checkpoint_path="./checkpoints_v1/best_model.pt",
        episode_idx=1,
        history_t=10,
        rollout_horizon=10,
        output_dir="./debug_v1_outputs",
    )

    debug_episode(cfg)
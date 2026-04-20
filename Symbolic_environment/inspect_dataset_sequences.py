# inspect_dataset_sequences.py

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt


HEADING_NAMES = ["N", "E", "S", "W"]


def decode_actions(action_ids: np.ndarray, action_names: np.ndarray) -> List[str]:
    return [str(action_names[int(a)]) for a in action_ids]


def decode_headings(heading_ids: np.ndarray) -> List[str]:
    return [HEADING_NAMES[int(h)] for h in heading_ids]


def print_episode_summary(
    episode_idx: int,
    observations: np.ndarray,
    actions: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    headings: np.ndarray,
    collisions: np.ndarray,
    action_names: np.ndarray,
) -> None:
    obs_seq = observations[episode_idx]       # [T, 1, H, W]
    act_seq = actions[episode_idx]            # [T-1]
    row_seq = rows[episode_idx]               # [T]
    col_seq = cols[episode_idx]               # [T]
    head_seq = headings[episode_idx]          # [T]
    coll_seq = collisions[episode_idx]        # [T-1]

    decoded_actions = decode_actions(act_seq, action_names)
    decoded_headings = decode_headings(head_seq)

    print("\n" + "=" * 90)
    print(f"Episode {episode_idx}")
    print(f"Sequence length (observations): {len(obs_seq)}")
    print(f"Number of actions: {len(act_seq)}")
    print("-" * 90)

    print("States:")
    for t in range(len(obs_seq)):
        print(
            f"  t={t:02d} | row={int(row_seq[t]):2d}, col={int(col_seq[t]):2d}, "
            f"heading={decoded_headings[t]}"
        )

    print("-" * 90)
    print("Actions:")
    for t in range(len(act_seq)):
        print(
            f"  t={t:02d} -> t={t+1:02d} | "
            f"action={decoded_actions[t]:>10s} | collision={bool(coll_seq[t])}"
        )

    print("-" * 90)
    print(
        f"Observation stats for this episode: "
        f"min={obs_seq.min():.4f}, max={obs_seq.max():.4f}, mean={obs_seq.mean():.4f}"
    )
    print("=" * 90)


def save_episode_frames(
    episode_idx: int,
    observations: np.ndarray,
    actions: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    headings: np.ndarray,
    collisions: np.ndarray,
    action_names: np.ndarray,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    obs_seq = observations[episode_idx]       # [T, 1, H, W]
    act_seq = actions[episode_idx]            # [T-1]
    row_seq = rows[episode_idx]
    col_seq = cols[episode_idx]
    head_seq = headings[episode_idx]
    coll_seq = collisions[episode_idx]

    decoded_actions = decode_actions(act_seq, action_names)
    decoded_headings = decode_headings(head_seq)

    episode_dir = os.path.join(output_dir, f"episode_{episode_idx:04d}")
    os.makedirs(episode_dir, exist_ok=True)

    T = obs_seq.shape[0]

    for t in range(T):
        img = obs_seq[t, 0]   # [H, W]

        plt.figure(figsize=(5, 5))
        plt.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        plt.axis("off")

        title = (
            f"Episode {episode_idx} | t={t}\n"
            f"pose=({int(row_seq[t])}, {int(col_seq[t])}, {decoded_headings[t]})"
        )

        if t < T - 1:
            title += (
                f"\nnext_action={decoded_actions[t]} | "
                f"collision={bool(coll_seq[t])}"
            )

        plt.title(title)
        plt.tight_layout()

        save_path = os.path.join(episode_dir, f"frame_{t:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Saved frames for episode {episode_idx} to: {episode_dir}")


def save_episode_contact_sheet(
    episode_idx: int,
    observations: np.ndarray,
    actions: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    headings: np.ndarray,
    collisions: np.ndarray,
    action_names: np.ndarray,
    output_dir: str,
    cols_per_row: int = 4,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    obs_seq = observations[episode_idx]
    act_seq = actions[episode_idx]
    row_seq = rows[episode_idx]
    col_seq = cols[episode_idx]
    head_seq = headings[episode_idx]
    coll_seq = collisions[episode_idx]

    decoded_actions = decode_actions(act_seq, action_names)
    decoded_headings = decode_headings(head_seq)

    T = obs_seq.shape[0]
    ncols = cols_per_row
    nrows = int(np.ceil(T / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for t in range(T):
        ax = axes[t]
        ax.imshow(obs_seq[t, 0], cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")

        title = f"t={t}\n({int(row_seq[t])},{int(col_seq[t])},{decoded_headings[t]})"
        if t < T - 1:
            title += f"\n{decoded_actions[t]} | col={bool(coll_seq[t])}"
        ax.set_title(title, fontsize=9)

    for t in range(T, len(axes)):
        axes[t].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"episode_{episode_idx:04d}_sheet.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved contact sheet for episode {episode_idx} to: {save_path}")


def inspect_dataset(npz_path: str, episode_indices: List[int], output_dir: str) -> None:
    data = np.load(npz_path)

    observations = data["observations"]
    actions = data["actions"]
    rows = data["rows"]
    cols = data["cols"]
    headings = data["headings"]
    collisions = data["collisions"]
    action_names = data["action_names"]

    N = observations.shape[0]

    print(f"\nLoaded dataset: {npz_path}")
    print(f"Number of episodes: {N}")
    print(f"Observation shape: {observations.shape}")
    print(f"Actions shape:     {actions.shape}")
    print(f"Rows shape:        {rows.shape}")
    print(f"Cols shape:        {cols.shape}")
    print(f"Headings shape:    {headings.shape}")
    print(f"Collisions shape:  {collisions.shape}")

    for ep in episode_indices:
        if ep < 0 or ep >= N:
            print(f"Skipping invalid episode index: {ep}")
            continue

        print_episode_summary(
            ep,
            observations,
            actions,
            rows,
            cols,
            headings,
            collisions,
            action_names,
        )

        save_episode_frames(
            ep,
            observations,
            actions,
            rows,
            cols,
            headings,
            collisions,
            action_names,
            output_dir=output_dir,
        )

        save_episode_contact_sheet(
            ep,
            observations,
            actions,
            rows,
            cols,
            headings,
            collisions,
            action_names,
            output_dir=output_dir,
            cols_per_row=4,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect saved trajectory sequences from dataset.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the .npz dataset file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Episode indices to inspect",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset_inspection_outputs",
        help="Directory to save per-episode frames and contact sheets",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_dataset(
        npz_path=args.dataset_path,
        episode_indices=args.episodes,
        output_dir=args.output_dir,
    )
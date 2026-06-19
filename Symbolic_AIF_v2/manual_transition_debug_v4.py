from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from simulator import TinyIndoorEnv, Pose, HEADINGS
from model_v4 import ModelV4Config, WorldModelV4

CHECKPOINT_PATH = "./checkpoints_v4/best_model_v4.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./debug_v4_outputs"


# Action indices:
# 0 = forward
# 1 = backward
# 2 = turn_left
# 3 = turn_right


START_POSE = Pose(row=1, col=5, heading="W")

HISTORY_ACTIONS = [
    0, 0, 2, 0, 0, 0, 0, 2, 0
]

ROLLOUT_ACTIONS = [
    0, 0, 2, 0, 0, 3, 0, 0, 2, 0, 0, 2, 2, 0, 3, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 3, 0, 2, 0, 3, 0,
    0, 3, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 2, 0, 0, 0, 0, 2, 0, 2, 0, 1, 2, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0
]


ACTION_NAMES = ["forward", "backward", "turn_left", "turn_right"]


def get_device() -> torch.device:
    return torch.device(DEVICE)


def heading_idx_to_name(idx: int) -> str:
    return HEADINGS[idx]


def pose_tuple_str(row: int, col: int, heading: str) -> str:
    return f"({row:2d}, {col:1d}, {heading})"


def render_ascii_with_pose(env: TinyIndoorEnv, row: int, col: int, heading: str) -> str:
    arrow = {"N": "^", "E": ">", "S": "v", "W": "<"}[heading]
    rows = []
    for r in range(env.rows):
        row_chars = []
        for c in range(env.cols):
            ch = env.grid[r][c]
            if (r, c) == (row, col):
                row_chars.append(arrow)
            else:
                row_chars.append(ch)
        rows.append(" ".join(row_chars))
    return "\n".join(rows)


def build_history_from_actions(
    env: TinyIndoorEnv,
    start_pose: Pose,
    history_actions: List[int],
) -> Dict[str, np.ndarray]:
    obs0, info0 = env.reset(start_pose=start_pose, use_goal=False)

    observations = [obs0]
    rows = [info0["row"]]
    cols = [info0["col"]]
    headings = [info0["heading_idx"]]
    collisions = []

    for a in history_actions:
        result = env.step(a)
        observations.append(result.obs)
        rows.append(result.info["row"])
        cols.append(result.info["col"])
        headings.append(result.info["heading_idx"])
        collisions.append(int(result.info["collision"]))

    return {
        "observations": np.array(observations, dtype=np.float32),
        "rows": np.array(rows, dtype=np.int64),
        "cols": np.array(cols, dtype=np.int64),
        "headings": np.array(headings, dtype=np.int64),
        "collisions": np.array(collisions, dtype=np.int64),
    }


def rollout_ground_truth_from_env(
    env: TinyIndoorEnv,
    rollout_actions: List[int],
) -> Dict[str, np.ndarray]:
    rows, cols, headings, observations, collisions = [], [], [], [], []

    for a in rollout_actions:
        result = env.step(a)
        rows.append(result.info["row"])
        cols.append(result.info["col"])
        headings.append(result.info["heading_idx"])
        observations.append(result.obs)
        collisions.append(int(result.info["collision"]))

    return {
        "rows": np.array(rows, dtype=np.int64),
        "cols": np.array(cols, dtype=np.int64),
        "headings": np.array(headings, dtype=np.int64),
        "observations": np.array(observations, dtype=np.float32),
        "collisions": np.array(collisions, dtype=np.int64),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    device = get_device()
    print(f"Using device: {device}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model_cfg = ModelV4Config(**ckpt["model_cfg"])
    model = WorldModelV4(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    use_prior = model_cfg.use_predictive_prior
    print(f"use_predictive_prior: {use_prior}")

    # --------------------------------------------------------
    # Build true history in simulator
    # --------------------------------------------------------
    env = TinyIndoorEnv(seed=42)
    history = build_history_from_actions(
        env=env,
        start_pose=START_POSE,
        history_actions=HISTORY_ACTIONS,
    )
    gt_roll = rollout_ground_truth_from_env(
        env=env,
        rollout_actions=ROLLOUT_ACTIONS,
    )

    # --------------------------------------------------------
    # Prepare tensors: [1, T, 1, H, W]
    # --------------------------------------------------------
    observations = torch.from_numpy(history["observations"]).unsqueeze(0).unsqueeze(2).to(device)
    actions = torch.tensor(HISTORY_ACTIONS, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        filt = model.forward_filter(observations, actions)

        # Posterior beliefs
        row_pred_seq = filt["row_logits_seq"][0].argmax(dim=-1).cpu().numpy()
        col_pred_seq = filt["col_logits_seq"][0].argmax(dim=-1).cpu().numpy()
        heading_pred_seq = filt["heading_logits_seq"][0].argmax(dim=-1).cpu().numpy()
        row_probs_seq = filt["row_probs_seq"]
        col_probs_seq = filt["col_probs_seq"]
        heading_probs_seq = filt["heading_probs_seq"]
        context_seq = filt["context_seq"]
        context_logvar_seq = filt["context_logvar_seq"]   # [1, T, context_dim]

        # Raw encoder beliefs (pre-Bayesian-update)
        enc_row_pred = filt["encoder_row_logits_seq"][0].argmax(dim=-1).cpu().numpy()
        enc_col_pred = filt["encoder_col_logits_seq"][0].argmax(dim=-1).cpu().numpy()
        enc_heading_pred = filt["encoder_heading_logits_seq"][0].argmax(dim=-1).cpu().numpy()

    # --------------------------------------------------------
    # Filter history debug
    # --------------------------------------------------------
    T = len(history["rows"])
    print("\n" + "=" * 140)
    print("FILTERED HISTORY DEBUG (V4)")
    print("=" * 140)

    if use_prior:
        print("\nt | true_pose       | encoder_pred     | posterior_pred   | row_ok col_ok head_ok | ctx_std")
        print("-" * 130)
    else:
        print("\nt | true_pose       | pred_pose        | row_ok col_ok heading_ok | ctx_std")
        print("-" * 110)

    for t in range(T):
        tr = int(history["rows"][t])
        tc = int(history["cols"][t])
        th = heading_idx_to_name(int(history["headings"][t]))

        pr = int(row_pred_seq[t])
        pc = int(col_pred_seq[t])
        ph = heading_idx_to_name(int(heading_pred_seq[t]))

        row_ok = tr == pr
        col_ok = tc == pc
        heading_ok = th == ph

        # Context uncertainty: mean std at this timestep
        ctx_std = float(torch.exp(0.5 * context_logvar_seq[0, t]).mean().item())

        if use_prior:
            er = int(enc_row_pred[t])
            ec = int(enc_col_pred[t])
            eh = heading_idx_to_name(int(enc_heading_pred[t]))
            print(
                f"{t:2d} | {pose_tuple_str(tr, tc, th):15s} | "
                f"{pose_tuple_str(er, ec, eh):15s} | "
                f"{pose_tuple_str(pr, pc, ph):15s} | "
                f"  {int(row_ok):1d}      {int(col_ok):1d}        {int(heading_ok):1d}    | "
                f"ctx_std={ctx_std:.4f}"
            )
        else:
            print(
                f"{t:2d} | {pose_tuple_str(tr, tc, th):15s} | {pose_tuple_str(pr, pc, ph):15s} | "
                f"  {int(row_ok):1d}      {int(col_ok):1d}        {int(heading_ok):1d} | ctx_std={ctx_std:.4f}"
            )

    t0 = T - 1
    true_t0 = (
        int(history["rows"][t0]),
        int(history["cols"][t0]),
        heading_idx_to_name(int(history["headings"][t0])),
    )
    pred_t0 = (
        int(row_pred_seq[t0]),
        int(col_pred_seq[t0]),
        heading_idx_to_name(int(heading_pred_seq[t0])),
    )

    print("\n" + "=" * 140)
    print("HISTORY STATE USED FOR TRANSITION")
    print("=" * 140)
    print(f"Filtered history final time index t0 = {t0}")
    print(f"TRUE HISTORY FINAL POSE : {pose_tuple_str(*true_t0)}")
    print(f"PRED HISTORY FINAL POSE : {pose_tuple_str(*pred_t0)}")
    ctx_std_t0 = float(torch.exp(0.5 * context_logvar_seq[0, t0]).mean().item())
    print(f"CONTEXT STD at t0       : {ctx_std_t0:.4f}  (higher = more uncertain)")

    env_for_ascii = TinyIndoorEnv(seed=42)
    print("\nTRUE HISTORY FINAL ASCII MAP:")
    print(render_ascii_with_pose(env_for_ascii, *true_t0))
    print("\nPRED HISTORY FINAL ASCII MAP:")
    print(render_ascii_with_pose(env_for_ascii, *pred_t0))

    # --------------------------------------------------------
    # Rollout from filtered structured state
    # --------------------------------------------------------
    row_probs_start = row_probs_seq[:, t0, :]
    col_probs_start = col_probs_seq[:, t0, :]
    heading_probs_start = heading_probs_seq[:, t0, :]
    context_start = context_seq[:, t0, :]
    action_roll = torch.tensor(ROLLOUT_ACTIONS, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        roll = model.rollout_from_filtered_state(
            row_probs_start=row_probs_start,
            col_probs_start=col_probs_start,
            heading_probs_start=heading_probs_start,
            context_start=context_start,
            action_seq=action_roll,
        )

    row_pred_roll = roll["row_logits_roll"][0].argmax(dim=-1).cpu().numpy()
    col_pred_roll = roll["col_logits_roll"][0].argmax(dim=-1).cpu().numpy()
    heading_pred_roll = roll["heading_logits_roll"][0].argmax(dim=-1).cpu().numpy()
    coll_pred_roll = roll["collision_logits_roll"][0].argmax(dim=-1).cpu().numpy()

    # --------------------------------------------------------
    # Rollout debug
    # --------------------------------------------------------
    K = len(ROLLOUT_ACTIONS)
    print("\n" + "=" * 140)
    print("MANUAL TRANSITION DEBUG (V4)")
    print("=" * 140)

    for k in range(K):
        a_idx = ROLLOUT_ACTIONS[k]
        a_name = ACTION_NAMES[a_idx]

        tr = int(gt_roll["rows"][k])
        tc = int(gt_roll["cols"][k])
        th = heading_idx_to_name(int(gt_roll["headings"][k]))

        pr = int(row_pred_roll[k])
        pc = int(col_pred_roll[k])
        ph = heading_idx_to_name(int(heading_pred_roll[k]))

        true_coll = int(gt_roll["collisions"][k])
        pred_coll = int(coll_pred_roll[k])

        row_ok = tr == pr
        col_ok = tc == pc
        heading_ok = th == ph
        coll_ok = true_coll == pred_coll

        # Context std at this rollout step
        ctx_std_k = float(
            torch.exp(0.5 * roll["context_roll"][0, k]).abs().mean().item()
        )

        print("\n" + "-" * 130)
        print(f"rollout step = {k+1} | action = {a_name} (idx={a_idx})")
        print(f"TRUE FUTURE POSE : {pose_tuple_str(tr, tc, th)}")
        print(f"PRED FUTURE POSE : {pose_tuple_str(pr, pc, ph)}")
        print(
            f"ROW OK={row_ok} | COL OK={col_ok} | HEADING OK={heading_ok} | "
            f"TRUE_COLL={true_coll} | PRED_COLL={pred_coll} | COLL_OK={coll_ok} | "
            f"CTX_MEAN_ABS={ctx_std_k:.4f}"
        )

        print("\nTRUE ASCII MAP:")
        print(render_ascii_with_pose(env_for_ascii, tr, tc, th))
        print("\nPRED ASCII MAP:")
        print(render_ascii_with_pose(env_for_ascii, pr, pc, ph))

    print("\n" + "=" * 140)
    print("SUMMARY")
    print("=" * 140)

    row_acc = np.mean(gt_roll["rows"] == row_pred_roll)
    col_acc = np.mean(gt_roll["cols"] == col_pred_roll)
    heading_acc = np.mean(gt_roll["headings"] == heading_pred_roll)
    collision_acc = np.mean(gt_roll["collisions"] == coll_pred_roll)

    print(f"Rollout row accuracy:       {row_acc:.4f}")
    print(f"Rollout col accuracy:       {col_acc:.4f}")
    print(f"Rollout heading accuracy:   {heading_acc:.4f}")
    print(f"Rollout collision accuracy: {collision_acc:.4f}")

    # Context uncertainty stats over the filter history
    ctx_stds = torch.exp(0.5 * context_logvar_seq[0]).mean(dim=-1).cpu().numpy()
    print(f"\nContext std over filter history (mean±std): "
          f"{ctx_stds.mean():.4f} ± {ctx_stds.std():.4f}")
    print(f"  min={ctx_stds.min():.4f}  max={ctx_stds.max():.4f}")


if __name__ == "__main__":
    main()

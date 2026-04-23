# manual_transition_debug_v1.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List

import torch

from model_v1 import ModelV1Config, WorldModelV1
from simulator import TinyIndoorEnv, Pose, ACTION_NAMES


# ============================================================
# Config
# ============================================================

@dataclass
class ManualTransitionDebugConfig:
    checkpoint_path: str = "./checkpoints_v1/best_model.pt"

    start_row: int = 1
    start_col: int = 5
    start_heading: str = "W"

    # First sequence builds the filtered belief state
    history_action_indices: List[int] = None

    # Second sequence tests transition rollout
    rollout_action_indices: List[int] = None

    output_dir: str = "./manual_transition_debug_v1_outputs"


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Helpers
# ============================================================

HEADINGS = ["N", "E", "S", "W"]


def heading_name(idx: int) -> str:
    return HEADINGS[idx]


def action_idx_to_name(idx: int) -> str:
    if not (0 <= idx < len(ACTION_NAMES)):
        raise ValueError(f"Invalid action index {idx}. Valid range: 0 to {len(ACTION_NAMES)-1}")
    return ACTION_NAMES[idx]


def build_model_cfg() -> ModelV1Config:
    return ModelV1Config(
        obs_channels=1,
        obs_height=64,
        obs_width=64,
        num_actions=4,
        action_emb_dim=16,
        encoder_feat_dim=128,
        gru_hidden_dim=256,
        latent_dim=64,
        num_row_classes=9,
        num_col_classes=9,
        num_heading_classes=4,
        decoder_base_channels=128,
    )


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


def render_ascii_for_pose(env: TinyIndoorEnv, pose: Pose) -> str:
    old_pose = env.pose
    env.pose = Pose(pose.row, pose.col, pose.heading)
    txt = env.render_topdown_ascii(show_goal=False)
    env.pose = old_pose
    return txt


# ============================================================
# Manual trajectory generation
# ============================================================

def generate_history_and_rollout_sequences(
    cfg: ManualTransitionDebugConfig,
) -> Dict[str, torch.Tensor]:
    if cfg.history_action_indices is None or len(cfg.history_action_indices) == 0:
        raise ValueError("history_action_indices must be provided and non-empty.")

    if cfg.rollout_action_indices is None or len(cfg.rollout_action_indices) == 0:
        raise ValueError("rollout_action_indices must be provided and non-empty.")

    env = TinyIndoorEnv(obs_height=64, obs_width=64, seed=42)

    start_pose = Pose(
        row=cfg.start_row,
        col=cfg.start_col,
        heading=cfg.start_heading,
    )

    obs0, info0 = env.reset(start_pose=start_pose, use_goal=False)

    observations = []
    rows = []
    cols = []
    headings = []
    history_actions = []

    observations.append(obs0)
    rows.append(int(info0["row"]))
    cols.append(int(info0["col"]))
    headings.append(int(info0["heading_idx"]))

    # Apply history actions first
    for a_idx in cfg.history_action_indices:
        action_name = action_idx_to_name(a_idx)
        result = env.step(action_name)

        observations.append(result.obs)
        rows.append(int(result.info["row"]))
        cols.append(int(result.info["col"]))
        headings.append(int(result.info["heading_idx"]))
        history_actions.append(a_idx)

    # Save true future rollout states by continuing simulator
    rollout_rows = []
    rollout_cols = []
    rollout_headings = []
    rollout_observations = []

    for a_idx in cfg.rollout_action_indices:
        action_name = action_idx_to_name(a_idx)
        result = env.step(action_name)

        rollout_observations.append(result.obs)
        rollout_rows.append(int(result.info["row"]))
        rollout_cols.append(int(result.info["col"]))
        rollout_headings.append(int(result.info["heading_idx"]))

    observations = torch.tensor(observations, dtype=torch.float32).unsqueeze(1)      # [T_hist,1,H,W]
    history_actions = torch.tensor(history_actions, dtype=torch.long)                 # [T_hist-1]
    rows = torch.tensor(rows, dtype=torch.long)
    cols = torch.tensor(cols, dtype=torch.long)
    headings = torch.tensor(headings, dtype=torch.long)

    rollout_actions = torch.tensor(cfg.rollout_action_indices, dtype=torch.long)      # [K]
    rollout_observations = torch.tensor(rollout_observations, dtype=torch.float32).unsqueeze(1)  # [K,1,H,W]
    rollout_rows = torch.tensor(rollout_rows, dtype=torch.long)
    rollout_cols = torch.tensor(rollout_cols, dtype=torch.long)
    rollout_headings = torch.tensor(rollout_headings, dtype=torch.long)

    return {
        "history_observations": observations.unsqueeze(0),         # [1,T_hist,1,H,W]
        "history_actions": history_actions.unsqueeze(0),           # [1,T_hist-1]
        "history_rows": rows.unsqueeze(0),                        # [1,T_hist]
        "history_cols": cols.unsqueeze(0),                        # [1,T_hist]
        "history_headings": headings.unsqueeze(0),                # [1,T_hist]

        "rollout_actions": rollout_actions.unsqueeze(0),          # [1,K]
        "rollout_observations_true": rollout_observations.unsqueeze(0),  # [1,K,1,H,W]
        "rollout_rows_true": rollout_rows.unsqueeze(0),           # [1,K]
        "rollout_cols_true": rollout_cols.unsqueeze(0),           # [1,K]
        "rollout_headings_true": rollout_headings.unsqueeze(0),   # [1,K]

        "env": env,
    }


# ============================================================
# Transition debug
# ============================================================

@torch.no_grad()
def debug_transition(
    model: WorldModelV1,
    batch: Dict[str, torch.Tensor],
    env: TinyIndoorEnv,
    device: torch.device,
) -> None:
    hist_obs = batch["history_observations"].to(device)
    hist_actions = batch["history_actions"].to(device)

    hist_rows = batch["history_rows"][0].cpu()
    hist_cols = batch["history_cols"][0].cpu()
    hist_headings = batch["history_headings"][0].cpu()

    rollout_actions = batch["rollout_actions"].to(device)
    rollout_rows_true = batch["rollout_rows_true"][0].cpu()
    rollout_cols_true = batch["rollout_cols_true"][0].cpu()
    rollout_headings_true = batch["rollout_headings_true"][0].cpu()

    filt = model.forward_filter(hist_obs, hist_actions)

    row_pred_hist = filt["row_logits_seq"].argmax(dim=-1)[0].cpu()
    col_pred_hist = filt["col_logits_seq"].argmax(dim=-1)[0].cpu()
    heading_pred_hist = filt["heading_logits_seq"].argmax(dim=-1)[0].cpu()

    t0 = hist_obs.shape[1] - 1

    print("\n" + "=" * 130)
    print("HISTORY STATE USED FOR TRANSITION")
    print("=" * 130)
    print(f"Filtered history final time index t0 = {t0}")
    print(f"TRUE HISTORY FINAL POSE : ({int(hist_rows[t0])}, {int(hist_cols[t0])}, {heading_name(int(hist_headings[t0]))})")
    print(f"PRED HISTORY FINAL POSE : ({int(row_pred_hist[t0])}, {int(col_pred_hist[t0])}, {heading_name(int(heading_pred_hist[t0]))})")

    true_hist_pose = Pose(int(hist_rows[t0]), int(hist_cols[t0]), heading_name(int(hist_headings[t0])))
    pred_hist_pose = Pose(int(row_pred_hist[t0]), int(col_pred_hist[t0]), heading_name(int(heading_pred_hist[t0])))

    print("\nTRUE HISTORY FINAL ASCII MAP:")
    print(render_ascii_for_pose(env, true_hist_pose))

    print("\nPRED HISTORY FINAL ASCII MAP:")
    print(render_ascii_for_pose(env, pred_hist_pose))

    z_start = filt["z_seq"][:, -1]   # [1,Z]
    h_start = filt["h_seq"][:, -1]   # [1,H]

    roll = model.rollout_from_filtered_state(
        z_start=z_start,
        h_start=h_start,
        action_seq=rollout_actions,
    )

    row_pred_roll = roll["row_logits_roll"].argmax(dim=-1)[0].cpu()
    col_pred_roll = roll["col_logits_roll"].argmax(dim=-1)[0].cpu()
    heading_pred_roll = roll["heading_logits_roll"].argmax(dim=-1)[0].cpu()

    print("\n" + "=" * 130)
    print("MANUAL TRANSITION DEBUG")
    print("=" * 130)

    for k in range(rollout_actions.shape[1]):
        a_idx = int(rollout_actions[0, k].item())
        a_name = action_idx_to_name(a_idx)

        rt = int(rollout_rows_true[k])
        ct = int(rollout_cols_true[k])
        ht = int(rollout_headings_true[k])

        rp = int(row_pred_roll[k])
        cp = int(col_pred_roll[k])
        hp = int(heading_pred_roll[k])

        true_pose = Pose(rt, ct, heading_name(ht))
        pred_pose = Pose(rp, cp, heading_name(hp))

        print("\n" + "-" * 130)
        print(f"rollout step = {k+1} | action = {a_name} (idx={a_idx})")
        print(f"TRUE FUTURE POSE : ({rt}, {ct}, {heading_name(ht)})")
        print(f"PRED FUTURE POSE : ({rp}, {cp}, {heading_name(hp)})")
        print(f"ROW OK={rt==rp} | COL OK={ct==cp} | HEADING OK={ht==hp}")

        print("\nTRUE ASCII MAP:")
        print(render_ascii_for_pose(env, true_pose))

        print("\nPRED ASCII MAP:")
        print(render_ascii_for_pose(env, pred_pose))


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = ManualTransitionDebugConfig(
        checkpoint_path="./checkpoints_v1/best_model.pt",
        start_row=1,
        start_col=5,
        start_heading="W",
        # Actions: forward:0, backward:1, turn_right:2, turn_left:3 #
        history_action_indices=[
            1, 1, 2, 0, 3, 0, 0, 0, 0, 
        ],

        rollout_action_indices=[
            0,0,2,0,0,0,0,2,0,0
            # 0, 3, 0, 0, 0, 0, 3, 0, 3, 0
        ],

        output_dir="./manual_transition_debug_v1_outputs",
    )

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    batch = generate_history_and_rollout_sequences(cfg)
    env = batch.pop("env")

    model_cfg = build_model_cfg()
    model = load_model_from_checkpoint(cfg.checkpoint_path, device, model_cfg)

    debug_transition(
        model=model,
        batch=batch,
        env=env,
        device=device,
    )


if __name__ == "__main__":
    main()
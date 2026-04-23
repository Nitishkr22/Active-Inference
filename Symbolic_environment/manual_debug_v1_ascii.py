# manual_debug_v1_ascii.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch

from dataset_loader import SequenceWorldModelDataset
from model_v1 import ModelV1Config, WorldModelV1
from simulator import TinyIndoorEnv, Pose, ACTION_NAMES


# ============================================================
# Config
# ============================================================

@dataclass
class ManualDebugConfig:
    checkpoint_path: str = "./checkpoints_v1/best_model.pt"

    # --------------------------------------------------------
    # Manual sequence definition
    # --------------------------------------------------------
    start_row: int = 1
    start_col: int = 5
    start_heading: str = "W"

    # Full manual action sequence used to generate true observations
    # from the simulator.
    manual_actions: List[str] = None

    # Choose where belief history ends and transition rollout starts
    history_t: int = 10

    # Output
    save_firstperson_debug: bool = False
    output_dir: str = "./manual_debug_v1_outputs"


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Loading model
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
# Helpers
# ============================================================

HEADINGS = ["N", "E", "S", "W"]


def heading_name(idx: int) -> str:
    return HEADINGS[idx]


def action_to_idx_map() -> Dict[str, int]:
    return {name: i for i, name in enumerate(ACTION_NAMES)}


def build_model_from_dataset_signature(device: torch.device) -> WorldModelV1:
    """
    We use dataset signature only to recover the class counts and image size.
    Assumes the same environment/data format as training.
    """
    # These match the current setup.
    model_cfg = ModelV1Config(
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
    return model_cfg


def render_ascii_for_pose(env: TinyIndoorEnv, pose: Pose) -> str:
    """
    Temporarily render ASCII map with the given pose.
    """
    old_pose = env.pose
    env.pose = Pose(pose.row, pose.col, pose.heading)
    txt = env.render_topdown_ascii(show_goal=False)
    env.pose = old_pose
    return txt


def pose_from_predictions(row_idx: int, col_idx: int, heading_idx: int) -> Pose:
    return Pose(row=row_idx, col=col_idx, heading=HEADINGS[heading_idx])


# ============================================================
# Generate manual trajectory from simulator
# ============================================================

def generate_manual_trajectory(
    cfg: ManualDebugConfig,
) -> Dict[str, torch.Tensor]:
    """
    Uses the simulator to generate the true observations and poses
    from the manually specified start pose and action sequence.

    Returns tensors with batch dimension 1:
      observations : [1,T,1,64,64]
      actions      : [1,T-1]
      rows         : [1,T]
      cols         : [1,T]
      headings     : [1,T]
    """
    if cfg.manual_actions is None or len(cfg.manual_actions) == 0:
        raise ValueError("manual_actions must be provided and non-empty.")

    env = TinyIndoorEnv(obs_height=64, obs_width=64, seed=42)

    start_pose = Pose(
        row=cfg.start_row,
        col=cfg.start_col,
        heading=cfg.start_heading,
    )

    obs0, info0 = env.reset(start_pose=start_pose, use_goal=False)

    T = len(cfg.manual_actions) + 1
    observations = []
    rows = []
    cols = []
    headings = []
    action_indices = []

    action_to_idx = action_to_idx_map()

    # initial state
    observations.append(obs0)
    rows.append(int(info0["row"]))
    cols.append(int(info0["col"]))
    headings.append(int(info0["heading_idx"]))

    for action_name in cfg.manual_actions:
        if action_name not in action_to_idx:
            raise ValueError(f"Unknown action '{action_name}'. Valid actions: {ACTION_NAMES}")

        result = env.step(action_name)

        observations.append(result.obs)
        rows.append(int(result.info["row"]))
        cols.append(int(result.info["col"]))
        headings.append(int(result.info["heading_idx"]))
        action_indices.append(action_to_idx[action_name])

    observations = torch.tensor(observations, dtype=torch.float32).unsqueeze(1)   # [T,1,H,W]
    actions = torch.tensor(action_indices, dtype=torch.long)                       # [T-1]
    rows = torch.tensor(rows, dtype=torch.long)                                    # [T]
    cols = torch.tensor(cols, dtype=torch.long)                                    # [T]
    headings = torch.tensor(headings, dtype=torch.long)                            # [T]

    return {
        "observations": observations.unsqueeze(0),   # [1,T,1,H,W]
        "actions": actions.unsqueeze(0),             # [1,T-1]
        "rows": rows.unsqueeze(0),                   # [1,T]
        "cols": cols.unsqueeze(0),                   # [1,T]
        "headings": headings.unsqueeze(0),           # [1,T]
        "env": env,
    }


# ============================================================
# Belief debug
# ============================================================

@torch.no_grad()
def debug_belief(
    model: WorldModelV1,
    batch: Dict[str, torch.Tensor],
    env: TinyIndoorEnv,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    observations = batch["observations"].to(device)
    actions = batch["actions"].to(device)

    out = model.forward_filter(observations, actions)

    row_pred = out["row_logits_seq"].argmax(dim=-1)[0].cpu()        # [T]
    col_pred = out["col_logits_seq"].argmax(dim=-1)[0].cpu()        # [T]
    heading_pred = out["heading_logits_seq"].argmax(dim=-1)[0].cpu()# [T]

    rows_true = batch["rows"][0].cpu()
    cols_true = batch["cols"][0].cpu()
    headings_true = batch["headings"][0].cpu()

    T = rows_true.shape[0]

    print("\n" + "=" * 130)
    print("MANUAL BELIEF / FILTER DEBUG")
    print("=" * 130)

    for t in range(T):
        rt, ct, ht = int(rows_true[t]), int(cols_true[t]), int(headings_true[t])
        rp, cp, hp = int(row_pred[t]), int(col_pred[t]), int(heading_pred[t])

        true_pose = Pose(rt, ct, heading_name(ht))
        pred_pose = Pose(rp, cp, heading_name(hp))

        print("\n" + "-" * 130)
        print(f"t = {t}")
        print(f"TRUE POSE : ({rt}, {ct}, {heading_name(ht)})")
        print(f"PRED POSE : ({rp}, {cp}, {heading_name(hp)})")
        print(f"ROW OK={rt==rp} | COL OK={ct==cp} | HEADING OK={ht==hp}")

        print("\nTRUE ASCII MAP:")
        print(render_ascii_for_pose(env, true_pose))

        print("\nPRED ASCII MAP:")
        print(render_ascii_for_pose(env, pred_pose))

    return {
        "filter_out": out,
        "row_pred": row_pred,
        "col_pred": col_pred,
        "heading_pred": heading_pred,
    }


# ============================================================
# Transition debug
# ============================================================

@torch.no_grad()
def debug_transition(
    model: WorldModelV1,
    batch: Dict[str, torch.Tensor],
    env: TinyIndoorEnv,
    filter_out: Dict[str, torch.Tensor],
    cfg: ManualDebugConfig,
    device: torch.device,
) -> None:
    observations = batch["observations"].to(device)
    actions = batch["actions"].to(device)
    rows_true = batch["rows"][0].cpu()
    cols_true = batch["cols"][0].cpu()
    headings_true = batch["headings"][0].cpu()

    T = observations.shape[1]
    K = len(cfg.manual_actions) - cfg.history_t
    if K <= 0:
        raise ValueError(
            f"history_t={cfg.history_t} leaves no future rollout steps. "
            f"Need history_t < len(manual_actions)."
        )

    z_start = filter_out["z_seq"][:, cfg.history_t]     # [1,Z]
    h_start = filter_out["h_seq"][:, cfg.history_t]     # [1,H]
    action_roll = actions[:, cfg.history_t:cfg.history_t + K]  # [1,K]

    roll = model.rollout_from_filtered_state(
        z_start=z_start,
        h_start=h_start,
        action_seq=action_roll,
    )

    row_pred_roll = roll["row_logits_roll"].argmax(dim=-1)[0].cpu()         # [K]
    col_pred_roll = roll["col_logits_roll"].argmax(dim=-1)[0].cpu()         # [K]
    heading_pred_roll = roll["heading_logits_roll"].argmax(dim=-1)[0].cpu() # [K]

    action_names = cfg.manual_actions[cfg.history_t:cfg.history_t + K]

    print("\n" + "=" * 130)
    print(f"MANUAL TRANSITION / ROLLOUT DEBUG (start from filtered state at t={cfg.history_t})")
    print("=" * 130)

    for k in range(K):
        t_future = cfg.history_t + 1 + k

        rt = int(rows_true[t_future])
        ct = int(cols_true[t_future])
        ht = int(headings_true[t_future])

        rp = int(row_pred_roll[k])
        cp = int(col_pred_roll[k])
        hp = int(heading_pred_roll[k])

        true_pose = Pose(rt, ct, heading_name(ht))
        pred_pose = Pose(rp, cp, heading_name(hp))

        print("\n" + "-" * 130)
        print(f"rollout step = {k+1} | action = {action_names[k]}")
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
    cfg = ManualDebugConfig(
        checkpoint_path="./checkpoints_v1/best_model.pt",
        start_row=1,
        start_col=5,
        start_heading="W",
        manual_actions=[
            "backward",
            "backward",
            "turn_left",
            "forward",
            "turn_right",
            "forward",
            "forward",
            "forward",
            "forward",
            "turn_right",
            "forward",
            "turn_right",
            "forward",
            "forward",
            "forward",
            "forward",
            "turn_right",
            "forward",
            "turn_right",
            "forward",
        ],
        history_t=10,
        save_firstperson_debug=False,
        output_dir="./manual_debug_v1_outputs",
    )

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    batch = generate_manual_trajectory(cfg)
    env = batch.pop("env")

    model_cfg = build_model_from_dataset_signature(device)
    model = load_model_from_checkpoint(cfg.checkpoint_path, device, model_cfg)

    belief_out = debug_belief(
        model=model,
        batch=batch,
        env=env,
        device=device,
    )

    debug_transition(
        model=model,
        batch=batch,
        env=env,
        filter_out=belief_out["filter_out"],
        cfg=cfg,
        device=device,
    )


if __name__ == "__main__":
    main()
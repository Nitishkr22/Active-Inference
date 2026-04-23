# manual_belief_debug_v1.py

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
class ManualBeliefDebugConfig:
    checkpoint_path: str = "./checkpoints_v1/best_model.pt"

    start_row: int = 1
    start_col: int = 5
    start_heading: str = "W"

    # Actions given as indices:
    # 0=forward, 1=backward, 2=turn_left, 3=turn_right
    manual_action_indices: List[int] = None

    output_dir: str = "./manual_belief_debug_v1_outputs"


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
# Trajectory generation
# ============================================================

def generate_manual_belief_sequence(
    cfg: ManualBeliefDebugConfig,
) -> Dict[str, torch.Tensor]:
    if cfg.manual_action_indices is None or len(cfg.manual_action_indices) == 0:
        raise ValueError("manual_action_indices must be provided and non-empty.")

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
    actions = []

    observations.append(obs0)
    rows.append(int(info0["row"]))
    cols.append(int(info0["col"]))
    headings.append(int(info0["heading_idx"]))

    for a_idx in cfg.manual_action_indices:
        action_name = action_idx_to_name(a_idx)
        result = env.step(action_name)

        observations.append(result.obs)
        rows.append(int(result.info["row"]))
        cols.append(int(result.info["col"]))
        headings.append(int(result.info["heading_idx"]))
        actions.append(a_idx)

    observations = torch.tensor(observations, dtype=torch.float32).unsqueeze(1)   # [T,1,H,W]
    actions = torch.tensor(actions, dtype=torch.long)                               # [T-1]
    rows = torch.tensor(rows, dtype=torch.long)
    cols = torch.tensor(cols, dtype=torch.long)
    headings = torch.tensor(headings, dtype=torch.long)

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
) -> None:
    observations = batch["observations"].to(device)
    actions = batch["actions"].to(device)

    rows_true = batch["rows"][0].cpu()
    cols_true = batch["cols"][0].cpu()
    headings_true = batch["headings"][0].cpu()

    out = model.forward_filter(observations, actions)

    row_pred = out["row_logits_seq"].argmax(dim=-1)[0].cpu()
    col_pred = out["col_logits_seq"].argmax(dim=-1)[0].cpu()
    heading_pred = out["heading_logits_seq"].argmax(dim=-1)[0].cpu()

    T = rows_true.shape[0]

    print("\n" + "=" * 130)
    print("MANUAL BELIEF DEBUG")
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


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = ManualBeliefDebugConfig(
        checkpoint_path="./checkpoints_v1/best_model.pt",
        start_row=6,
        start_col=3,
        start_heading="S",
        # Actions: forward:0, backward:1, turn_right:2, turn_left:3 #
        manual_action_indices=[0,1,2,0,2,2,0,0,3,3,3
            # 2,0,0,2,0,0,0,3,0,0,0,0,0,3,0,0,3,0,2,0,3,0,0,
            #                    3,0,0,0,2,0,0,2,0,0,0,0,0,2,0,0,0,1,2,0,2,2,0,0,3,3,3
            # 1, 1, 2, 0, 3, 0, 0, 0, 0, 3,
            # 0, 3, 0, 0, 0, 0
        ],
        output_dir="./manual_belief_debug_v1_outputs",
    )

    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    batch = generate_manual_belief_sequence(cfg)
    env = batch.pop("env")

    model_cfg = build_model_cfg()
    model = load_model_from_checkpoint(cfg.checkpoint_path, device, model_cfg)

    debug_belief(
        model=model,
        batch=batch,
        env=env,
        device=device,
    )


if __name__ == "__main__":
    main()
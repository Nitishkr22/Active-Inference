# evaluate_latent_predictiveness_v3.py

from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F

from model_v3 import WorldModelV3, ModelV3Config
from simulator import TinyIndoorEnv, Pose, StepResult


ACTION_NAMES = {
    0: "forward",
    1: "backward",
    2: "turn_left",
    3: "turn_right",
}

IDX_TO_HEADING = {0: "N", 1: "E", 2: "S", 3: "W"}


@dataclass
class EvalConfig:
    # checkpoint_path: str = "./checkpoints_v3/best_model.pt"
    checkpoint_path: str = "./checkpoints_v3_predictive_latent/best_predictive_latent.pt"
    model_config_json: str = "./checkpoints_v32/model_config.json"

    num_episodes: int = 200
    seq_len: int = 32
    rollout_horizon: int = 8
    seed: int = 123

    output_csv: str = "./latent_predictiveness_v3_results.csv"

    # random policy action probabilities
    p_forward: float = 0.55
    p_backward: float = 0.10
    p_left: float = 0.175
    p_right: float = 0.175


# ============================================================
# Model loading
# ============================================================

def load_model_config_from_json(path: str) -> ModelV3Config:
    with open(path, "r") as f:
        return ModelV3Config(**json.load(f))


def build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    model_config_json: Optional[str] = None,
) -> WorldModelV3:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model_config" in ckpt:
        model_cfg = ModelV3Config(**ckpt["model_config"])
    else:
        if model_config_json is None or not Path(model_config_json).exists():
            raise ValueError("Checkpoint has no model_config and model_config_json was not found.")
        print(f"Loaded model_config from JSON: {model_config_json}")
        model_cfg = load_model_config_from_json(model_config_json)

    model = WorldModelV3(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    return model


# ============================================================
# Environment helpers
# ============================================================

def step_env(env: TinyIndoorEnv, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    result: StepResult = env.step(action)
    return result.obs, float(result.reward), bool(result.done), result.info


def sample_action(rng: np.random.Generator, cfg: EvalConfig) -> int:
    probs = np.array(
        [cfg.p_forward, cfg.p_backward, cfg.p_left, cfg.p_right],
        dtype=np.float64,
    )
    probs = probs / probs.sum()
    return int(rng.choice(np.arange(4), p=probs))


def collect_episode(
    env: TinyIndoorEnv,
    cfg: EvalConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    obs, info = env.reset(use_goal=False)

    observations: List[np.ndarray] = [np.asarray(obs, dtype=np.float32)]
    actions: List[int] = []
    rows: List[int] = [int(info["row"])]
    cols: List[int] = [int(info["col"])]
    headings: List[int] = [int(info["heading_idx"])]
    collisions: List[int] = []

    for _ in range(cfg.seq_len - 1):
        a = sample_action(rng, cfg)
        obs, reward, done, info = step_env(env, a)

        observations.append(np.asarray(obs, dtype=np.float32))
        actions.append(a)
        rows.append(int(info["row"]))
        cols.append(int(info["col"]))
        headings.append(int(info["heading_idx"]))
        collisions.append(int(info["collision"]))

    return {
        "observations": np.stack(observations, axis=0),  # [T,H,W] or [T,1,H,W]
        "actions": np.asarray(actions, dtype=np.int64),  # [T-1]
        "rows": np.asarray(rows, dtype=np.int64),
        "cols": np.asarray(cols, dtype=np.int64),
        "headings": np.asarray(headings, dtype=np.int64),
        "collisions": np.asarray(collisions, dtype=np.int64),
    }


def ensure_obs_shape(obs_np: np.ndarray) -> np.ndarray:
    """
    Converts episode observations to [T,1,H,W].
    """
    if obs_np.ndim == 3:
        return obs_np[:, None, :, :]
    if obs_np.ndim == 4:
        return obs_np
    raise ValueError(f"Unexpected obs shape: {obs_np.shape}")


# ============================================================
# Model helpers
# ============================================================

@torch.no_grad()
def run_filter(
    model: WorldModelV3,
    obs_np: np.ndarray,
    actions_np: np.ndarray,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Full teacher-forced filtering over the whole observed sequence.

    Returns sequences:
      row_probs_seq      [1,T,R]
      col_probs_seq      [1,T,C]
      heading_probs_seq  [1,T,Hd]
      context_seq        [1,T,U]
    """
    obs_np = ensure_obs_shape(obs_np).astype(np.float32)
    obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)  # [1,T,1,H,W]
    act_t = torch.from_numpy(actions_np.astype(np.int64)).unsqueeze(0).to(device)  # [1,T-1]

    if hasattr(model, "forward_filter"):
        out = model.forward_filter(obs_t, act_t)
    elif hasattr(model, "filter_sequence"):
        out = model.filter_sequence(obs_t, act_t)
    else:
        raise AttributeError("Model must provide forward_filter() or filter_sequence().")

    return out


@torch.no_grad()
def rollout_from_time(
    model: WorldModelV3,
    filter_out: Dict[str, torch.Tensor],
    actions_np: np.ndarray,
    t0: int,
    horizon: int,
) -> Dict[str, torch.Tensor]:
    """
    Starts from teacher-forced filtered belief at t0, then free-rolls actions
    a[t0], ..., a[t0+horizon-1].
    """
    device = filter_out["row_probs_seq"].device

    row0 = filter_out["row_probs_seq"][:, t0, :]
    col0 = filter_out["col_probs_seq"][:, t0, :]
    hdg0 = filter_out["heading_probs_seq"][:, t0, :]

    ctx0 = None
    if "context_seq" in filter_out:
        ctx0 = filter_out["context_seq"][:, t0, :]

    seq_np = actions_np[t0 : t0 + horizon].astype(np.int64)
    seq_t = torch.from_numpy(seq_np).unsqueeze(0).to(device)

    if hasattr(model, "rollout_from_filtered_state"):
        roll = model.rollout_from_filtered_state(
            row_probs_start=row0,
            col_probs_start=col0,
            heading_probs_start=hdg0,
            context_start=ctx0,
            action_seq=seq_t,
        )
    elif hasattr(model, "rollout_from_belief"):
        inputs = {
            "row_probs_start": row0,
            "col_probs_start": col0,
            "heading_probs_start": hdg0,
            "action_seq": seq_t,
        }
        if ctx0 is not None:
            inputs["context_start"] = ctx0
        roll = model.rollout_from_belief(**inputs)
    else:
        raise AttributeError(
            "Model must provide rollout_from_filtered_state() or rollout_from_belief()."
        )

    return roll


def get_context_rollout_if_available(roll: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Handles possible key names for predicted latent/context rollout.
    Returns [1,H,U] or None.
    """
    for key in ["context_roll", "context_rollout", "context_probs_roll", "context_seq_roll"]:
        if key in roll:
            return roll[key]
    return None


# ============================================================
# Metrics
# ============================================================

def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def argmax_pose_from_probs(
    row_probs: torch.Tensor,
    col_probs: torch.Tensor,
    heading_probs: torch.Tensor,
) -> Tuple[int, int, int]:
    r = int(torch.argmax(row_probs).item())
    c = int(torch.argmax(col_probs).item())
    h = int(torch.argmax(heading_probs).item())
    return r, c, h


def safe_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    a,b: [U]
    """
    a = a.detach().float()
    b = b.detach().float()
    return float((1.0 - F.cosine_similarity(a[None, :], b[None, :], dim=-1)).item())


def safe_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.mse_loss(a.detach().float(), b.detach().float()).item())


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_episode(
    model: WorldModelV3,
    episode: Dict[str, np.ndarray],
    cfg: EvalConfig,
    device: torch.device,
) -> List[Dict[str, Any]]:

    obs_np = episode["observations"]
    actions_np = episode["actions"]
    rows_np = episode["rows"]
    cols_np = episode["cols"]
    headings_np = episode["headings"]

    T = len(rows_np)
    Hmax = cfg.rollout_horizon

    filter_out = run_filter(model, obs_np, actions_np, device)

    has_context = "context_seq" in filter_out
    results: List[Dict[str, Any]] = []

    # valid t0: need at least horizon future actions
    for t0 in range(1, T - Hmax):
        roll = rollout_from_time(
            model=model,
            filter_out=filter_out,
            actions_np=actions_np,
            t0=t0,
            horizon=Hmax,
        )

        row_roll = roll["row_probs_roll"][0]          # [H,R]
        col_roll = roll["col_probs_roll"][0]          # [H,C]
        hdg_roll = roll["heading_probs_roll"][0]      # [H,Hd]

        ctx_roll = get_context_rollout_if_available(roll)

        for h in range(1, Hmax + 1):
            true_t = t0 + h

            pred_r, pred_c, pred_h = argmax_pose_from_probs(
                row_roll[h - 1],
                col_roll[h - 1],
                hdg_roll[h - 1],
            )

            true_r = int(rows_np[true_t])
            true_c = int(cols_np[true_t])
            true_h = int(headings_np[true_t])

            row_acc = int(pred_r == true_r)
            col_acc = int(pred_c == true_c)
            heading_acc = int(pred_h == true_h)
            full_pose_acc = int(row_acc and col_acc and heading_acc)

            row_ent = float(entropy(row_roll[h - 1]).item())
            col_ent = float(entropy(col_roll[h - 1]).item())
            heading_ent = float(entropy(hdg_roll[h - 1]).item())

            context_mse = np.nan
            context_cosdist = np.nan

            if has_context and ctx_roll is not None:
                # teacher-forced target context at future true time
                ctx_target = filter_out["context_seq"][0, true_t, :]

                # predicted context for horizon h
                # ctx_roll may be [1,H,U]
                ctx_pred = ctx_roll[0, h - 1, :]

                context_mse = safe_mse(ctx_pred, ctx_target)
                context_cosdist = safe_cosine_distance(ctx_pred, ctx_target)

            results.append(
                {
                    "t0": t0,
                    "horizon": h,
                    "row_acc": row_acc,
                    "col_acc": col_acc,
                    "heading_acc": heading_acc,
                    "full_pose_acc": full_pose_acc,
                    "row_entropy": row_ent,
                    "col_entropy": col_ent,
                    "heading_entropy": heading_ent,
                    "state_entropy": row_ent + col_ent + heading_ent,
                    "context_mse": context_mse,
                    "context_cosdist": context_cosdist,
                    "pred_row": pred_r,
                    "pred_col": pred_c,
                    "pred_heading": pred_h,
                    "true_row": true_r,
                    "true_col": true_c,
                    "true_heading": true_h,
                }
            )

    return results


def summarize(all_rows: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 100)
    print("LATENT / CONTEXT PREDICTIVENESS SUMMARY")
    print("=" * 100)

    if not all_rows:
        print("No results.")
        return

    horizons = sorted(set(int(r["horizon"]) for r in all_rows))

    print()
    print("Pose rollout accuracy by horizon:")
    print(
        f"{'H':>3} | {'row':>8} | {'col':>8} | {'heading':>8} | "
        f"{'full_pose':>10} | {'entropy':>8} | {'ctx_mse':>10} | {'ctx_cos':>10}"
    )
    print("-" * 86)

    for h in horizons:
        rows = [r for r in all_rows if int(r["horizon"]) == h]

        row_acc = np.mean([r["row_acc"] for r in rows])
        col_acc = np.mean([r["col_acc"] for r in rows])
        heading_acc = np.mean([r["heading_acc"] for r in rows])
        full_acc = np.mean([r["full_pose_acc"] for r in rows])
        ent = np.mean([r["state_entropy"] for r in rows])

        ctx_mse_vals = np.array([r["context_mse"] for r in rows], dtype=np.float64)
        ctx_cos_vals = np.array([r["context_cosdist"] for r in rows], dtype=np.float64)

        ctx_mse = np.nanmean(ctx_mse_vals) if not np.all(np.isnan(ctx_mse_vals)) else np.nan
        ctx_cos = np.nanmean(ctx_cos_vals) if not np.all(np.isnan(ctx_cos_vals)) else np.nan

        print(
            f"{h:3d} | "
            f"{row_acc:8.4f} | "
            f"{col_acc:8.4f} | "
            f"{heading_acc:8.4f} | "
            f"{full_acc:10.4f} | "
            f"{ent:8.4f} | "
            f"{ctx_mse:10.6f} | "
            f"{ctx_cos:10.6f}"
        )

    print()
    print("Interpretation guide:")
    print("  - If full_pose_acc stays high as horizon increases, the rollout is predictive.")
    print("  - If context_mse/cosdist stays low, the latent/context is dynamically stable.")
    print("  - If pose accuracy is good but context error explodes, current latent is not ideal for latent planning.")
    print("  - If both pose and context degrade quickly, train a better predictive latent state before latent EFE.")


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return

    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved detailed results to: {path}")


# ============================================================
# Main
# ============================================================

def main():
    cfg = EvalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Num episodes: {cfg.num_episodes}")
    print(f"Seq len:      {cfg.seq_len}")
    print(f"Horizon:      {cfg.rollout_horizon}")

    model = build_model_from_checkpoint(
        checkpoint_path=cfg.checkpoint_path,
        device=device,
        model_config_json=cfg.model_config_json,
    )

    rng = np.random.default_rng(cfg.seed)

    all_rows: List[Dict[str, Any]] = []

    for ep in range(cfg.num_episodes):
        env = TinyIndoorEnv(seed=cfg.seed + ep)
        episode = collect_episode(env, cfg, rng)

        ep_rows = evaluate_episode(
            model=model,
            episode=episode,
            cfg=cfg,
            device=device,
        )

        for r in ep_rows:
            r["episode"] = ep

        all_rows.extend(ep_rows)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[{ep + 1:04d}/{cfg.num_episodes:04d}] collected {len(ep_rows)} rollout points")

    summarize(all_rows)
    save_csv(all_rows, cfg.output_csv)


if __name__ == "__main__":
    main()
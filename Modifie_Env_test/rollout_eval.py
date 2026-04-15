## rollout evaluation script ##
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split

from sequence_dataset import SequenceDataset
from gru_world_model import GRUWorldModel
from train_hidden_transition_multistep_pose import HiddenTransitionModel_multistep_pose


def decode_pose_from_hidden(wm, h):
    """
    Decode row / col / heading from a hidden state.

    Args:
        wm: trained GRU world model (used only for pose_head)
        h:  [B, H] hidden state

    Returns:
        row_pred:      [B]
        col_pred:      [B]
        heading_pred:  [B]
        row_conf:      [B]
        col_conf:      [B]
        heading_conf:  [B]
    """
    row_logits, col_logits, heading_logits = wm.pose_head(h)

    row_prob = torch.softmax(row_logits, dim=-1)
    col_prob = torch.softmax(col_logits, dim=-1)
    heading_prob = torch.softmax(heading_logits, dim=-1)

    row_pred = torch.argmax(row_prob, dim=-1)
    col_pred = torch.argmax(col_prob, dim=-1)
    heading_pred = torch.argmax(heading_prob, dim=-1)

    row_conf = row_prob.gather(1, row_pred.unsqueeze(1)).squeeze(1)
    col_conf = col_prob.gather(1, col_pred.unsqueeze(1)).squeeze(1)
    heading_conf = heading_prob.gather(1, heading_pred.unsqueeze(1)).squeeze(1)

    return row_pred, col_pred, heading_pred, row_conf, col_conf, heading_conf


def evaluate_rollout_horizon(
    dataset_path="tiny_nav_sequence_dataset.npz",
    gru_ckpt_path="checkpoints/best_gru_world_model.pt",
    transition_ckpt_path="checkpoints/best_hidden_transition_multistep_pose.pt",
    batch_size=64,
    max_horizon=15,
    split="val",
):
    """
    Evaluate how transition-model rollout accuracy changes with rollout horizon.

    Method:
      1. Run GRU on real observation-action sequence to get true hidden states h_seq.
      2. Choose every possible rollout start time t.
      3. Start from true hidden state h_seq[:, t].
      4. Roll out transition model using future actions.
      5. Decode pose from predicted hidden state.
      6. Compare with ground-truth pose/heading at each rollout step.

    Prints:
      For each horizon H:
        - row accuracy
        - col accuracy
        - heading accuracy
        - full pose accuracy
        - mean absolute row error
        - mean absolute col error
        - mean confidence values
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # Load dataset
    # -----------------------------
    dataset = SequenceDataset(dataset_path)

    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    if split == "train":
        eval_ds = train_ds
    else:
        eval_ds = val_ds

    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    # -----------------------------
    # Load trained GRU world model
    # -----------------------------
    wm = GRUWorldModel().to(device)
    gru_ckpt = torch.load(gru_ckpt_path, map_location=device)
    wm.load_state_dict(gru_ckpt["model_state_dict"])
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    # -----------------------------
    # Load trained hidden transition model
    # -----------------------------
    transition_model = HiddenTransitionModel_multistep_pose(hidden_dim=wm.hidden_dim).to(device)
    trans_ckpt = torch.load(transition_ckpt_path, map_location=device)
    transition_model.load_state_dict(trans_ckpt["model_state_dict"])
    transition_model.eval()

    # -----------------------------
    # Metric containers per horizon
    # -----------------------------
    metrics = {}
    for horizon in range(1, max_horizon + 1):
        metrics[horizon] = {
            "row_correct": 0,
            "col_correct": 0,
            "heading_correct": 0,
            "full_pose_correct": 0,
            "row_abs_err_sum": 0.0,
            "col_abs_err_sum": 0.0,
            "heading_conf_sum": 0.0,
            "row_conf_sum": 0.0,
            "col_conf_sum": 0.0,
            "count": 0,
        }

    # -----------------------------
    # Evaluation loop
    # -----------------------------
    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(device)           # [B, T, 1, 64, 64]
            actions = batch["actions"].to(device)   # [B, T-1]
            pos = batch["pos"].to(device)           # [B, T, 2]
            heading = batch["heading"].to(device)   # [B, T]

            # Run GRU on real sequence to get hidden states
            out = wm.forward_sequence(obs, actions)
            h_seq = out["hidden_states"]            # [B, T, H]

            B, T, H = h_seq.shape

            # Possible rollout start times:
            # start_t must leave enough future actions and states
            for start_t in range(T - 1):
                h0 = h_seq[:, start_t, :]          # [B, H]
                h_pred = h0.clone()

                max_possible_horizon = min(max_horizon, T - 1 - start_t)

                for k in range(1, max_possible_horizon + 1):
                    action_t = actions[:, start_t + k - 1]   # action leading to next state
                    h_pred = transition_model(h_pred, action_t)

                    # Decode imagined pose
                    row_pred, col_pred, heading_pred, row_conf, col_conf, heading_conf = decode_pose_from_hidden(wm, h_pred)

                    # Ground truth at future step
                    true_row = pos[:, start_t + k, 0]
                    true_col = pos[:, start_t + k, 1]
                    true_heading = heading[:, start_t + k]

                    row_correct = (row_pred == true_row)
                    col_correct = (col_pred == true_col)
                    heading_correct = (heading_pred == true_heading)
                    full_pose_correct = row_correct & col_correct & heading_correct

                    row_abs_err = torch.abs(row_pred - true_row)
                    col_abs_err = torch.abs(col_pred - true_col)

                    metrics[k]["row_correct"] += row_correct.sum().item()
                    metrics[k]["col_correct"] += col_correct.sum().item()
                    metrics[k]["heading_correct"] += heading_correct.sum().item()
                    metrics[k]["full_pose_correct"] += full_pose_correct.sum().item()
                    metrics[k]["row_abs_err_sum"] += row_abs_err.sum().item()
                    metrics[k]["col_abs_err_sum"] += col_abs_err.sum().item()
                    metrics[k]["row_conf_sum"] += row_conf.sum().item()
                    metrics[k]["col_conf_sum"] += col_conf.sum().item()
                    metrics[k]["heading_conf_sum"] += heading_conf.sum().item()
                    metrics[k]["count"] += B

    # -----------------------------
    # Print results
    # -----------------------------
    print("\n" + "=" * 90)
    print("ROLLOUT-HORIZON EVALUATION")
    print("=" * 90)
    print(
        f"{'H':>3} | {'row_acc':>8} | {'col_acc':>8} | {'head_acc':>8} | "
        f"{'full_pose':>9} | {'mean_row_err':>12} | {'mean_col_err':>12} | "
        f"{'row_conf':>8} | {'col_conf':>8} | {'head_conf':>9}"
    )
    print("-" * 90)

    for horizon in range(1, max_horizon + 1):
        count = metrics[horizon]["count"]
        if count == 0:
            continue

        row_acc = metrics[horizon]["row_correct"] / count
        col_acc = metrics[horizon]["col_correct"] / count
        heading_acc = metrics[horizon]["heading_correct"] / count
        full_pose_acc = metrics[horizon]["full_pose_correct"] / count
        mean_row_err = metrics[horizon]["row_abs_err_sum"] / count
        mean_col_err = metrics[horizon]["col_abs_err_sum"] / count
        mean_row_conf = metrics[horizon]["row_conf_sum"] / count
        mean_col_conf = metrics[horizon]["col_conf_sum"] / count
        mean_heading_conf = metrics[horizon]["heading_conf_sum"] / count

        print(
            f"{horizon:>3} | "
            f"{row_acc:>8.3f} | "
            f"{col_acc:>8.3f} | "
            f"{heading_acc:>8.3f} | "
            f"{full_pose_acc:>9.3f} | "
            f"{mean_row_err:>12.3f} | "
            f"{mean_col_err:>12.3f} | "
            f"{mean_row_conf:>8.3f} | "
            f"{mean_col_conf:>8.3f} | "
            f"{mean_heading_conf:>9.3f}"
        )


if __name__ == "__main__":
    evaluate_rollout_horizon(
        dataset_path="tiny_nav_sequence_dataset.npz",
        gru_ckpt_path="checkpoints/best_gru_world_model.pt",
        transition_ckpt_path="checkpoints/best_hidden_transition_multistep_pose.pt",
        batch_size=64,
        max_horizon=15,
        split="val",
    )
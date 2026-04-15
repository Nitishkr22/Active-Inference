# evaluate_teacher_forced_vs_free_rollout.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# from sequence_dataset import SequenceDataset
# from gru_world_model import GRUWorldModel


# ==========================================================
# Transition model classes
# ==========================================================

class HiddenTransitionModelMultiStepPose(nn.Module):
    """
    Plain MLP transition:
        h_next = f(h, action)
    """
    def __init__(self, hidden_dim=128, num_actions=4, action_emb_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.action_emb_dim = action_emb_dim

        self.action_embed = nn.Embedding(num_actions, action_emb_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + action_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

    def forward(self, h, action):
        """
        h      : [B, H]
        action : [B]
        """
        a_emb = self.action_embed(action)          # [B, action_emb_dim]
        x = torch.cat([h, a_emb], dim=-1)          # [B, H + action_emb_dim]
        h_next = self.net(x)                       # [B, H]
        return h_next


class ResidualHiddenTransitionModelMultiStepPose(nn.Module):
    """
    Residual transition:
        h_next = h + delta(h, action)
    """
    def __init__(self, hidden_dim=128, num_actions=4, action_emb_dim=16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.action_emb_dim = action_emb_dim

        self.action_embed = nn.Embedding(num_actions, action_emb_dim)
        self.delta_net = nn.Sequential(
            nn.Linear(hidden_dim + action_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

    def forward(self, h, action):
        """
        h      : [B, H]
        action : [B]
        """
        a_emb = self.action_embed(action)          # [B, action_emb_dim]
        x = torch.cat([h, a_emb], dim=-1)          # [B, H + action_emb_dim]
        delta = self.delta_net(x)                  # [B, H]
        h_next = h + delta                         # [B, H]
        return h_next


class ScaledResidualHiddenTransitionModelMultiStepPose(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_actions=4,
        action_emb_dim=16,
        alpha_init=0.1,
        learnable_alpha=False,
    ):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, action_emb_dim)
        self.delta_net = nn.Sequential(
            nn.Linear(hidden_dim + action_emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha_init)))

    def forward(self, h, action):
        a_emb = self.action_embed(action)
        x = torch.cat([h, a_emb], dim=-1)
        delta = self.delta_net(x)
        h_next = h + self.alpha * delta
        return h_next

# ==========================================================
# Metric accumulator helpers
# ==========================================================

def make_empty_stats(rollout_len):
    """
    Create one accumulator dict per horizon.
    """
    stats = []
    for _ in range(rollout_len):
        stats.append({
            "latent_mse_sum": 0.0,
            "row_correct": 0,
            "col_correct": 0,
            "heading_correct": 0,
            "full_pose_correct": 0,
            "count": 0,
        })
    return stats


def update_stats_for_one_prediction(
    stats_k,
    h_pred,
    h_target,
    row_logits,
    col_logits,
    heading_logits,
    row_target,
    col_target,
    heading_target,
):
    """
    Update stats for one prediction step.
    """
    B, H = h_pred.shape

    # Sum over all latent dimensions and samples
    stats_k["latent_mse_sum"] += F.mse_loss(h_pred, h_target, reduction="sum").item()

    # Predicted classes
    row_pred = row_logits.argmax(dim=1)
    col_pred = col_logits.argmax(dim=1)
    heading_pred = heading_logits.argmax(dim=1)

    # Correct masks
    row_ok = (row_pred == row_target)
    col_ok = (col_pred == col_target)
    heading_ok = (heading_pred == heading_target)

    stats_k["row_correct"] += row_ok.sum().item()
    stats_k["col_correct"] += col_ok.sum().item()
    stats_k["heading_correct"] += heading_ok.sum().item()
    stats_k["full_pose_correct"] += (row_ok & col_ok & heading_ok).sum().item()
    stats_k["count"] += B


def finalize_stats(stats, hidden_dim):
    """
    Convert raw accumulators to readable metrics.
    """
    results = []

    for k, s in enumerate(stats):
        count = s["count"]
        results.append({
            "horizon": k + 1,
            "latent_mse": s["latent_mse_sum"] / max(count * hidden_dim, 1),
            "row_acc": s["row_correct"] / max(count, 1),
            "col_acc": s["col_correct"] / max(count, 1),
            "heading_acc": s["heading_correct"] / max(count, 1),
            "full_pose_acc": s["full_pose_correct"] / max(count, 1),
            "count": count,
        })

    return results


# ==========================================================
# Core batch evaluation
# ==========================================================

def evaluate_batch_teacher_forced_vs_free(
    h_seq,
    actions,
    pos,
    heading,
    transition_model,
    wm,
    rollout_len,
    tf_stats,
    fr_stats,
):
    """
    Evaluate one batch for:
      1) teacher-forced rollout
      2) free rollout

    h_seq   : [B, T, H]
    actions : [B, T-1]
    pos     : [B, T, 2]
    heading : [B, T]
    """
    B, T, H = h_seq.shape

    max_start = T - rollout_len - 1
    if max_start < 0:
        return

    for start_t in range(max_start + 1):
        # Free rollout starts once from the true hidden state
        h_free = h_seq[:, start_t, :]   # [B, H]

        for k in range(rollout_len):
            a_t = actions[:, start_t + k]                  # [B]
            h_target = h_seq[:, start_t + k + 1, :]       # [B, H]

            row_target = pos[:, start_t + k + 1, 0]       # [B]
            col_target = pos[:, start_t + k + 1, 1]       # [B]
            heading_target = heading[:, start_t + k + 1]  # [B]

            # --------------------------------------------------
            # 1) Teacher-forced rollout
            # each step uses TRUE hidden state as input
            # --------------------------------------------------
            h_tf_in = h_seq[:, start_t + k, :]            # [B, H]
            h_tf_pred = transition_model(h_tf_in, a_t)    # [B, H]

            tf_row_logits, tf_col_logits, tf_heading_logits = wm.pose_head(h_tf_pred)

            update_stats_for_one_prediction(
                tf_stats[k],
                h_tf_pred,
                h_target,
                tf_row_logits,
                tf_col_logits,
                tf_heading_logits,
                row_target,
                col_target,
                heading_target,
            )

            # --------------------------------------------------
            # 2) Free rollout
            # first step uses true hidden state, later uses own prediction
            # --------------------------------------------------
            h_free = transition_model(h_free, a_t)        # [B, H]

            fr_row_logits, fr_col_logits, fr_heading_logits = wm.pose_head(h_free)

            update_stats_for_one_prediction(
                fr_stats[k],
                h_free,
                h_target,
                fr_row_logits,
                fr_col_logits,
                fr_heading_logits,
                row_target,
                col_target,
                heading_target,
            )


# ==========================================================
# Printing helpers
# ==========================================================

def print_results_table(title, results):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(
        f"{'H':>3} | {'latent_mse':>12} | {'row_acc':>8} | {'col_acc':>8} | "
        f"{'heading_acc':>11} | {'full_pose':>10} | {'count':>8}"
    )
    print("-" * 90)

    for r in results:
        print(
            f"{r['horizon']:>3d} | "
            f"{r['latent_mse']:>12.6f} | "
            f"{r['row_acc']:>8.3f} | "
            f"{r['col_acc']:>8.3f} | "
            f"{r['heading_acc']:>11.3f} | "
            f"{r['full_pose_acc']:>10.3f} | "
            f"{r['count']:>8d}"
        )


def print_gap_table(tf_results, fr_results):
    print("\n" + "=" * 90)
    print("FREE ROLLOUT vs TEACHER-FORCED GAP")
    print("=" * 90)
    print(
        f"{'H':>3} | {'mse_gap':>10} | {'row_gap':>8} | {'col_gap':>8} | "
        f"{'head_gap':>9} | {'full_gap':>9}"
    )
    print("-" * 90)

    for tf_r, fr_r in zip(tf_results, fr_results):
        mse_gap = fr_r["latent_mse"] - tf_r["latent_mse"]
        row_gap = fr_r["row_acc"] - tf_r["row_acc"]
        col_gap = fr_r["col_acc"] - tf_r["col_acc"]
        head_gap = fr_r["heading_acc"] - tf_r["heading_acc"]
        full_gap = fr_r["full_pose_acc"] - tf_r["full_pose_acc"]

        print(
            f"{tf_r['horizon']:>3d} | "
            f"{mse_gap:>10.6f} | "
            f"{row_gap:>8.3f} | "
            f"{col_gap:>8.3f} | "
            f"{head_gap:>9.3f} | "
            f"{full_gap:>9.3f}"
        )


def summarize_overall(results):
    """
    Weighted average across horizons.
    """
    total_count = sum(r["count"] for r in results)
    if total_count == 0:
        return {
            "latent_mse": 0.0,
            "row_acc": 0.0,
            "col_acc": 0.0,
            "heading_acc": 0.0,
            "full_pose_acc": 0.0,
            "count": 0,
        }

    return {
        "latent_mse": sum(r["latent_mse"] * r["count"] for r in results) / total_count,
        "row_acc": sum(r["row_acc"] * r["count"] for r in results) / total_count,
        "col_acc": sum(r["col_acc"] * r["count"] for r in results) / total_count,
        "heading_acc": sum(r["heading_acc"] * r["count"] for r in results) / total_count,
        "full_pose_acc": sum(r["full_pose_acc"] * r["count"] for r in results) / total_count,
        "count": total_count,
    }


def print_overall_summary(tf_summary, fr_summary):
    print("\n" + "=" * 90)
    print("OVERALL SUMMARY")
    print("=" * 90)

    print("Teacher-forced:")
    print(
        f"  latent_mse   = {tf_summary['latent_mse']:.6f}\n"
        f"  row_acc      = {tf_summary['row_acc']:.3f}\n"
        f"  col_acc      = {tf_summary['col_acc']:.3f}\n"
        f"  heading_acc  = {tf_summary['heading_acc']:.3f}\n"
        f"  full_pose    = {tf_summary['full_pose_acc']:.3f}\n"
        f"  count        = {tf_summary['count']}"
    )

    print("\nFree rollout:")
    print(
        f"  latent_mse   = {fr_summary['latent_mse']:.6f}\n"
        f"  row_acc      = {fr_summary['row_acc']:.3f}\n"
        f"  col_acc      = {fr_summary['col_acc']:.3f}\n"
        f"  heading_acc  = {fr_summary['heading_acc']:.3f}\n"
        f"  full_pose    = {fr_summary['full_pose_acc']:.3f}\n"
        f"  count        = {fr_summary['count']}"
    )

    print("\nDifference (free - teacher-forced):")
    print(
        f"  latent_mse_gap   = {fr_summary['latent_mse'] - tf_summary['latent_mse']:.6f}\n"
        f"  row_acc_gap      = {fr_summary['row_acc'] - tf_summary['row_acc']:.3f}\n"
        f"  col_acc_gap      = {fr_summary['col_acc'] - tf_summary['col_acc']:.3f}\n"
        f"  heading_acc_gap  = {fr_summary['heading_acc'] - tf_summary['heading_acc']:.3f}\n"
        f"  full_pose_gap    = {fr_summary['full_pose_acc'] - tf_summary['full_pose_acc']:.3f}"
    )


# ==========================================================
# Main evaluation
# ==========================================================

def evaluate_teacher_forced_vs_free_rollout(
    dataset_path="tiny_nav_sequence_dataset.npz",
    gru_ckpt_path="checkpoints/best_gru_world_model.pt",
    transition_ckpt_path="checkpoints/best_hidden_transition_multistep_pose.pt",
    batch_size=32,
    rollout_len=None,
    split="val",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    dataset = SequenceDataset(dataset_path)

    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train

    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    if split == "train":
        eval_ds = train_ds
    elif split == "val":
        eval_ds = val_ds
    elif split == "all":
        eval_ds = dataset
    else:
        raise ValueError("split must be one of: 'train', 'val', 'all'")

    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset split: {split}")
    print(f"Number of sequences in evaluation set: {len(eval_ds)}")

    # --------------------------------------------------
    # Load frozen GRU world model
    # --------------------------------------------------
    wm = GRUWorldModel().to(device)
    wm_ckpt = torch.load(gru_ckpt_path, map_location=device)
    wm.load_state_dict(wm_ckpt["model_state_dict"])
    wm.eval()

    for p in wm.parameters():
        p.requires_grad = False

    # --------------------------------------------------
    # Load transition checkpoint
    # --------------------------------------------------
    trans_ckpt = torch.load(transition_ckpt_path, map_location=device)

    hidden_dim = trans_ckpt.get("hidden_dim", wm.hidden_dim)

    if rollout_len is None:
        rollout_len = trans_ckpt.get("rollout_len", 3)

    model_type = trans_ckpt.get("model_type", "plain_mlp_transition")

    # Choose correct model class automatically
    if model_type in ["residual_mlp_transition", "residual_mlp_transition_scheduled_sampling"]:
        transition_model = ResidualHiddenTransitionModelMultiStepPose(
            hidden_dim=hidden_dim,
            num_actions=4,
            action_emb_dim=16,
        ).to(device)
    elif model_type == "scaled_residual_mlp_transition":
        transition_model = ScaledResidualHiddenTransitionModelMultiStepPose(
            hidden_dim=hidden_dim,
            num_actions=4,
            action_emb_dim=16,
            alpha_init=trans_ckpt.get("alpha_init", 0.1),
            learnable_alpha=trans_ckpt.get("learnable_alpha", False),
        ).to(device)
    else:
        transition_model = HiddenTransitionModelMultiStepPose(
            hidden_dim=hidden_dim,
            num_actions=4,
            action_emb_dim=16,
        ).to(device)

    transition_model.load_state_dict(trans_ckpt["model_state_dict"])
    transition_model.eval()

    print(f"Loaded transition model from: {transition_ckpt_path}")
    print(f"Detected model_type = {model_type}")
    print(f"Using rollout_len = {rollout_len}")

    # --------------------------------------------------
    # Stats accumulators
    # --------------------------------------------------
    tf_stats = make_empty_stats(rollout_len)
    fr_stats = make_empty_stats(rollout_len)

    # --------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            obs = batch["obs"].to(device)           # [B, T, 1, 64, 64]
            actions = batch["actions"].to(device)   # [B, T-1]
            pos = batch["pos"].to(device)           # [B, T, 2]
            heading = batch["heading"].to(device)   # [B, T]

            out = wm.forward_sequence(obs, actions)
            h_seq = out["hidden_states"]            # [B, T, H]

            evaluate_batch_teacher_forced_vs_free(
                h_seq=h_seq,
                actions=actions,
                pos=pos,
                heading=heading,
                transition_model=transition_model,
                wm=wm,
                rollout_len=rollout_len,
                tf_stats=tf_stats,
                fr_stats=fr_stats,
            )

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(eval_loader):
                print(f"Processed batch {batch_idx + 1}/{len(eval_loader)}")

    # --------------------------------------------------
    # Finalize and print
    # --------------------------------------------------
    tf_results = finalize_stats(tf_stats, hidden_dim)
    fr_results = finalize_stats(fr_stats, hidden_dim)

    print_results_table("TEACHER-FORCED ROLLOUT RESULTS", tf_results)
    print_results_table("FREE ROLLOUT RESULTS", fr_results)
    print_gap_table(tf_results, fr_results)

    tf_summary = summarize_overall(tf_results)
    fr_summary = summarize_overall(fr_results)
    print_overall_summary(tf_summary, fr_summary)


if __name__ == "__main__":
    evaluate_teacher_forced_vs_free_rollout(
        dataset_path="tiny_nav_sequence_dataset.npz",
        gru_ckpt_path="checkpoints/best_gru_world_model.pt",
        transition_ckpt_path="checkpoints/best_hidden_transition_scaled_residual_multistep_pose.pt",
        batch_size=32,
        rollout_len=None,   # None -> read from checkpoint
        split="val",        # "train", "val", or "all"
    )
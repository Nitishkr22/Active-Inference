## script B: debug_hidden_transition_model.py ##
import torch
import torch.nn as nn
import numpy as np

from simulator import TinyIndoorEnv
from gru_world_model import GRUWorldModel


# ---------------------------------------------------------
# Same transition model definition used during training
# ---------------------------------------------------------
class HiddenTransitionModel_multistep_pose(nn.Module):
    def __init__(self, hidden_dim=128, num_actions=4):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, 16)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
        )

    def forward(self, h, action):
        a_emb = self.action_embed(action)          # [B, 16]
        x = torch.cat([h, a_emb], dim=-1)          # [B, hidden_dim+16]
        return self.net(x)                         # [B, hidden_dim]


# ---------------------------------------------------------
# Helper: decode pose logits to discrete pose
# ---------------------------------------------------------
def decode_pose(logits_row, logits_col, logits_head):
    prob_row = torch.softmax(logits_row, dim=-1)
    prob_col = torch.softmax(logits_col, dim=-1)
    prob_head = torch.softmax(logits_head, dim=-1)

    row = torch.argmax(prob_row, dim=-1).item()
    col = torch.argmax(prob_col, dim=-1).item()
    head = torch.argmax(prob_head, dim=-1).item()

    conf_row = prob_row[0, row].item()
    conf_col = prob_col[0, col].item()
    conf_head = prob_head[0, head].item()

    return row, col, head, conf_row, conf_col, conf_head


ACTION_NAMES = ["forward", "turn_left", "turn_right", "stay"]


# ---------------------------------------------------------
# You can change these as you want
# History actions: used to reach a current state first
# Future actions: used only for imagination rollout
# ---------------------------------------------------------
def get_history_actions():
    return [
        0, 0, 0,      # move east
        2,            # turn south
        0, 0,         # move south
    ]


def get_future_actions():
    return [
        0,            # forward
        1,            # turn left
        0, 0,         # forward, forward
        2,            # turn right
        0,            # forward
        0,0,0,0,2,0,1,1,0,0
    ]


# ---------------------------------------------------------
# Build GRU hidden state from real history
# ---------------------------------------------------------
def build_hidden_from_history(env, wm, history_actions, device):
    """
    Runs the simulator using real actions and feeds the observation-action
    history into the GRU. Returns:

    - current hidden state h_t
    - current simulator result
    - obs history list
    - action history list
    """
    # result = env.reset()
    # obs = result.observation

    # obs_seq = []
    # action_seq = []

    # for action in history_actions:
    #     obs_seq.append(obs)
    #     action_seq.append(action)

    #     result = env.step(action)
    #     obs = result.observation
    result = env.reset()
    obs_seq = [result.observation]
    action_seq = []

    for action in history_actions:
      result = env.step(action)
      action_seq.append(action)
      obs_seq.append(result.observation)


    # Build tensors for GRU
    obs_array = np.stack(obs_seq, axis=0)                     # [T, 64, 64]
    obs_tensor = (
        torch.from_numpy(obs_array)
        .float()
        .unsqueeze(0)     # [1, T, 64, 64]
        .unsqueeze(2)     # [1, T, 1, 64, 64]
        .to(device)
    )

    act_array = np.array(action_seq, dtype=np.int64)         # [T]
    act_tensor = (
        torch.from_numpy(act_array)
        .long()
        .unsqueeze(0)     # [1, T]
        .to(device)
    )

    with torch.no_grad():
        out = wm.forward_sequence(obs_tensor, act_tensor)
        h_seq = out["hidden_states"]                         # [1, T, H]
        h_t = h_seq[:, -1, :]                                # [1, H]

        logits_row = out["row_logits"][:, -1]
        logits_col = out["col_logits"][:, -1]
        logits_head = out["heading_logits"][:, -1]

    pred_row, pred_col, pred_head, conf_r, conf_c, conf_h = decode_pose(
        logits_row, logits_col, logits_head
    )

    return h_t, result, pred_row, pred_col, pred_head, conf_r, conf_c, conf_h


# ---------------------------------------------------------
# Main Script B
# ---------------------------------------------------------
def run_debug_hidden_transition(
    gru_ckpt_path="checkpoints/best_gru_world_model.pt",
    transition_ckpt_path="checkpoints/best_hidden_transition_multistep_pose.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------- ENV ----------------
    env = TinyIndoorEnv()
    history_actions = get_history_actions()
    future_actions = get_future_actions()

    print("\nInitial environment state:")
    env.reset().info
    env.render()

    # ---------------- LOAD GRU ----------------
    wm = GRUWorldModel().to(device)
    gru_ckpt = torch.load(gru_ckpt_path, map_location=device)
    wm.load_state_dict(gru_ckpt["model_state_dict"])
    wm.eval()

    # ---------------- LOAD TRANSITION MODEL ----------------
    transition_model = HiddenTransitionModel_multistep_pose(hidden_dim=wm.hidden_dim).to(device)
    trans_ckpt = torch.load(transition_ckpt_path, map_location=device)
    transition_model.load_state_dict(trans_ckpt["model_state_dict"])
    transition_model.eval()

    # -----------------------------------------------------
    # Step 1: Use real history to reach a current state
    # -----------------------------------------------------
    h_t, result, pred_row, pred_col, pred_head, conf_r, conf_c, conf_h = build_hidden_from_history(
        env, wm, history_actions, device
    )

    print("\n" + "=" * 70)
    print("STATE AFTER REAL HISTORY")
    print("History actions:", [ACTION_NAMES[a] for a in history_actions])
    print("True current pose:", result.info["pos"], "heading:", result.info["heading_name"])
    print(f"GRU current pose prediction: ({pred_row}, {pred_col}), heading: {pred_head}")
    print(f"Confidence | row={conf_r:.3f}, col={conf_c:.3f}, heading={conf_h:.3f}")
    env.render()

    # -----------------------------------------------------
    # Step 2: Start imagination rollout from h_t
    # -----------------------------------------------------
    print("\n" + "=" * 70)
    print("IMAGINED FUTURE ROLLOUT")
    print("Future actions:", [ACTION_NAMES[a] for a in future_actions])

    imagined_h = h_t.clone()

    for k, action in enumerate(future_actions):
        print("\n" + "-" * 60)
        print(f"Imagined step {k}")
        print("Action:", ACTION_NAMES[action])

        # ---- transition model prediction ----
        a_tensor = torch.tensor([action], dtype=torch.long, device=device)

        with torch.no_grad():
            imagined_h = transition_model(imagined_h, a_tensor)

            row_logits, col_logits, heading_logits = wm.pose_head(imagined_h)

        pred_row, pred_col, pred_head, conf_r, conf_c, conf_h = decode_pose(
            row_logits, col_logits, heading_logits
        )

        # ---- real environment step for comparison ----
        result = env.step(action)
        true_pos = result.info["pos"]
        true_heading = result.info["heading"]
        true_heading_name = result.info["heading_name"]

        row_err = abs(pred_row - true_pos[0])
        col_err = abs(pred_col - true_pos[1])
        heading_correct = (pred_head == true_heading)

        print(f"True pose: {true_pos}, heading: {true_heading_name}")
        print(f"Imagined pose: ({pred_row}, {pred_col}), heading: {pred_head}")
        print(f"Confidence | row={conf_r:.3f}, col={conf_c:.3f}, heading={conf_h:.3f}")
        print(f"Error | row_err={row_err}, col_err={col_err}, heading_correct={heading_correct}")

        env.render()

        if result.done:
            print("\nReached terminal state.")
            break


if __name__ == "__main__":
    run_debug_hidden_transition()
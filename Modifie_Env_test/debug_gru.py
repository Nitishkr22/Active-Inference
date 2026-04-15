## debug gru.py ##
import torch
import numpy as np

from simulator import TinyIndoorEnv
from gru_world_model import GRUWorldModel


# ------------------------------
# Decode pose
# ------------------------------
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


def get_manual_actions():
    return [
        0, 0, 0,      # forward
        2,            # turn_right
        0, 0, 0,      # forward
        1,            # turn_left
        0, 0,
        2,0,0,0,0,0,1,0,0,0
    ]


# ------------------------------
# Main debug
# ------------------------------
def run_debug_gru(
    gru_ckpt_path="checkpoints/best_gru_world_model.pt",
    max_steps=20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- ENV --------
    env = TinyIndoorEnv()

    result = env.reset()
    obs = result.observation

    print("\nStarting state:", result.info)
    env.render()

    # -------- MODEL --------
    wm = GRUWorldModel().to(device)
    ckpt = torch.load(gru_ckpt_path, map_location=device)
    wm.load_state_dict(ckpt["model_state_dict"])
    wm.eval()

    # -------- HISTORY --------
    obs_seq = []
    action_seq = []

    actions = get_manual_actions()

    for t in range(min(len(actions), max_steps)):
        print("\n" + "=" * 60)
        print(f"Time step {t}")

        # -------- STEP ENV --------
        action = actions[t]

        # result = env.step(action)
        # obs_next = result.observation
        # reward = result.reward
        done = result.done
        info = result.info

        print(f"Action taken: {ACTION_NAMES[action]}")

        # -------- STORE HISTORY --------
        obs_seq.append(obs)         # IMPORTANT: current obs
        action_seq.append(action)

        # -------- BUILD TENSORS (CORRECT SHAPES) --------
        # obs_seq: [T, 64, 64] → [1, T, 1, 64, 64]
        obs_array = np.stack(obs_seq, axis=0)
        obs_tensor = torch.from_numpy(obs_array).float().unsqueeze(0).unsqueeze(2).to(device)

        # action_seq: [T] → [1, T]
        act_array = np.array(action_seq)
        act_tensor = torch.from_numpy(act_array).long().unsqueeze(0).to(device)

        # -------- FORWARD GRU --------
        with torch.no_grad():
            out = wm.forward_sequence(obs_tensor, act_tensor)

            logits_row = out["row_logits"][:, -1]
            logits_col = out["col_logits"][:, -1]
            logits_head = out["heading_logits"][:, -1]

        # -------- DECODE --------
        pred_row, pred_col, pred_head, conf_r, conf_c, conf_h = decode_pose(
            logits_row, logits_col, logits_head
        )

        # -------- TRUE STATE --------
        true_pos = info["pos"]
        true_head = info["heading_name"]

        print(f"True pose: {true_pos}, heading: {true_head}")
        print(f"Pred pose: ({pred_row}, {pred_col}), heading: {pred_head}")

        print(
            f"Confidence | row={conf_r:.3f}, col={conf_c:.3f}, heading={conf_h:.3f}"
        )

        # -------- ERROR --------
        row_err = abs(pred_row - true_pos[0])
        col_err = abs(pred_col - true_pos[1])
        head_correct = (pred_head == info["heading"])

        print(
            f"Error | row_err={row_err}, col_err={col_err}, heading_correct={head_correct}"
        )

        result = env.step(action)
        obs_next = result.observation
        reward = result.reward
        env.render()

        # -------- NEXT STEP --------
        obs = obs_next

        if done:
            print("\nReached terminal state.")
            break


if __name__ == "__main__":
    run_debug_gru()
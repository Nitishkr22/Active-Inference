## collect_sequence_dataset.py ##
import numpy as np
from simulator import TinyIndoorEnv


def collect_sequence_dataset(
    num_episodes=1000,
    seq_len=8,
    max_steps_per_episode=30,
    save_path="../../dataset/tiny_nav_sequence_dataset.npz",
    seed=0,
    random_start=True,
    random_goal=True,
):
    """
    Collect sequence dataset for GRU world model training.

    Important:
    - Observations are goal-independent now.
    - We still store goal_pos for later planner/debug use.
    - Random starts help robustness.
    - Random goals are fine because goal is NOT rendered in observation anymore.
    """
    rng = np.random.default_rng(seed)
    env = TinyIndoorEnv(max_steps=max_steps_per_episode)
    free_cells = env.get_free_cells()

    obs_seqs = []
    action_seqs = []
    pos_seqs = []
    heading_seqs = []
    done_seqs = []
    goal_seqs = []

    for ep in range(num_episodes):
        # choose goal for this episode
        if random_goal:
            goal_pos = free_cells[rng.integers(len(free_cells))]
        else:
            goal_pos = env.goal_pos

        # reset with optional random start
        result = env.reset(
            goal_pos=goal_pos,
            random_start=random_start,
            rng=rng,
        )

        # make sure start != goal when random_start=True
        # if reset accidentally put us on goal, reset again
        tries = 0
        while result.info["pos"] == result.info["goal_pos"] and tries < 20:
            result = env.reset(
                goal_pos=goal_pos,
                random_start=random_start,
                rng=rng,
            )
            tries += 1

        obs_buffer = []
        action_buffer = []
        pos_buffer = []
        heading_buffer = []
        done_buffer = []
        goal_buffer = []

        obs_buffer.append(result.observation.astype(np.float32))
        pos_buffer.append(result.info["pos"])
        heading_buffer.append(result.info["heading"])
        done_buffer.append(False)
        goal_buffer.append(result.info["goal_pos"])

        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            action = int(rng.integers(0, 4))
            next_result = env.step(action)

            action_buffer.append(action)
            obs_buffer.append(next_result.observation.astype(np.float32))
            pos_buffer.append(next_result.info["pos"])
            heading_buffer.append(next_result.info["heading"])
            done_buffer.append(next_result.done)
            goal_buffer.append(next_result.info["goal_pos"])

            done = next_result.done
            steps += 1

        # Build sliding windows
        if len(obs_buffer) >= seq_len:
            for start in range(0, len(obs_buffer) - seq_len + 1):
                end = start + seq_len

                obs_seq = obs_buffer[start:end]             # [T, 64, 64]
                act_seq = action_buffer[start:end - 1]      # [T-1]
                pos_seq = pos_buffer[start:end]             # [T, 2]
                heading_seq = heading_buffer[start:end]     # [T]
                done_seq = done_buffer[start:end]           # [T]
                goal_seq = goal_buffer[start:end]           # [T, 2]

                obs_seqs.append(obs_seq)
                action_seqs.append(act_seq)
                pos_seqs.append(pos_seq)
                heading_seqs.append(heading_seq)
                done_seqs.append(done_seq)
                goal_seqs.append(goal_seq)

        if (ep + 1) % 100 == 0:
            print(f"Collected episode {ep + 1}/{num_episodes}")

    dataset = {
        "obs": np.array(obs_seqs, dtype=np.float32),            # [N, T, 64, 64]
        "actions": np.array(action_seqs, dtype=np.int64),       # [N, T-1]
        "pos": np.array(pos_seqs, dtype=np.int64),              # [N, T, 2]
        "heading": np.array(heading_seqs, dtype=np.int64),      # [N, T]
        "done": np.array(done_seqs, dtype=np.bool_),            # [N, T]
        "goal_pos": np.array(goal_seqs, dtype=np.int64),        # [N, T, 2]
    }

    np.savez_compressed(save_path, **dataset)

    print(f"\nSaved sequence dataset to: {save_path}")
    print("obs shape:", dataset["obs"].shape)
    print("actions shape:", dataset["actions"].shape)
    print("pos shape:", dataset["pos"].shape)
    print("heading shape:", dataset["heading"].shape)
    print("done shape:", dataset["done"].shape)
    print("goal_pos shape:", dataset["goal_pos"].shape)


if __name__ == "__main__":
    collect_sequence_dataset()
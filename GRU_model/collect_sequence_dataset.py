import numpy as np
from tiny_env import TinyIndoorEnv   # change if your filename differs

def collect_sequence_dataset(
    num_episodes=1000, # random rollouts
    seq_len=8,  # length of observation sequences (T)
    max_steps_per_episode=30, #max steps per episode to avoid infinite loops
    save_path="tiny_nav_sequence_dataset.npz",
    seed=0,
):
    rng = np.random.default_rng(seed)
    env = TinyIndoorEnv(max_steps=max_steps_per_episode)

    obs_seqs = []
    action_seqs = []
    pos_seqs = []
    heading_seqs = []
    done_seqs = []

    for ep in range(num_episodes):
        result = env.reset()

        # buffers to store full trajectory of one episode
        obs_buffer = []
        action_buffer = []
        pos_buffer = []
        heading_buffer = []
        done_buffer = []

        # set initial state
        obs_buffer.append(result.observation.astype(np.float32))
        pos_buffer.append(result.info["pos"])
        heading_buffer.append(result.info["heading"])
        done_buffer.append(False)

        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            action = rng.integers(0, 4)
            next_result = env.step(action)

            action_buffer.append(action)
            obs_buffer.append(next_result.observation.astype(np.float32))
            pos_buffer.append(next_result.info["pos"])
            heading_buffer.append(next_result.info["heading"])
            done_buffer.append(next_result.done)

            done = next_result.done
            steps += 1

        # We need sequences of length seq_len observations
        # and seq_len-1 actions between them
        if len(obs_buffer) >= seq_len:
            for start in range(0, len(obs_buffer) - seq_len + 1): # 24 sequence in one episode
                end = start + seq_len
                obs_seq = obs_buffer[start:end]                 # length seq_len
                act_seq = action_buffer[start:end - 1]         # length seq_len-1
                pos_seq = pos_buffer[start:end]
                heading_seq = heading_buffer[start:end]
                done_seq = done_buffer[start:end]

                obs_seqs.append(obs_seq)
                action_seqs.append(act_seq)
                pos_seqs.append(pos_seq)
                heading_seqs.append(heading_seq)
                done_seqs.append(done_seq)

        if (ep + 1) % 100 == 0:
            print(f"Collected episode {ep+1}/{num_episodes}")

    dataset = {
        "obs": np.array(obs_seqs, dtype=np.float32),            # [N, T, 64, 64] 24 sequences per episode for 1000 episodes = 24000 sequences
        "actions": np.array(action_seqs, dtype=np.int64),       # [N, T-1]
        "pos": np.array(pos_seqs, dtype=np.int64),              # [N, T, 2]
        "heading": np.array(heading_seqs, dtype=np.int64),      # [N, T]
        "done": np.array(done_seqs, dtype=np.bool_),            # [N, T]
    }

    np.savez_compressed(save_path, **dataset)

    print(f"\nSaved sequence dataset to: {save_path}")
    print("obs shape:", dataset["obs"].shape)
    print("actions shape:", dataset["actions"].shape)
    print("pos shape:", dataset["pos"].shape)
    print("heading shape:", dataset["heading"].shape)
    print("done shape:", dataset["done"].shape)


if __name__ == "__main__":
    collect_sequence_dataset()
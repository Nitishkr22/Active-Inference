import os
import numpy as np
from tiny_env import TinyIndoorEnv   # change this to your actual file name


def collect_random_dataset(
    num_episodes: int = 500,
    max_steps_per_episode: int = 30,
    save_path: str = "tiny_nav_dataset.npz",
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    env = TinyIndoorEnv(max_steps=max_steps_per_episode)

    obs_list = []
    act_list = []
    next_obs_list = []
    done_list = []

    # Optional debugging information
    pos_list = []
    heading_list = []
    next_pos_list = []
    next_heading_list = []

    for ep in range(num_episodes):
        result = env.reset()
        obs = result.observation
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            pos = result.info["pos"]
            heading = result.info["heading"]

            action = rng.integers(0, 4)  # 4 actions: forward, left, right, stay

            next_result = env.step(action)
            next_obs = next_result.observation
            done = next_result.done

            next_pos = next_result.info["pos"]
            next_heading = next_result.info["heading"]

            obs_list.append(obs.astype(np.float32))
            act_list.append(action)
            next_obs_list.append(next_obs.astype(np.float32))
            done_list.append(done)

            pos_list.append(pos)
            heading_list.append(heading)
            next_pos_list.append(next_pos)
            next_heading_list.append(next_heading)

            obs = next_obs
            result = next_result
            steps += 1

        if (ep + 1) % 50 == 0:
            print(f"Collected episode {ep+1}/{num_episodes}")

    dataset = {
        "obs": np.stack(obs_list),                 # [N, 64, 64]
        "actions": np.array(act_list, dtype=np.int64),   # [N]
        "next_obs": np.stack(next_obs_list),       # [N, 64, 64]
        "done": np.array(done_list, dtype=np.bool_),     # [N]
        "pos": np.array(pos_list, dtype=np.int64),       # [N, 2]
        "heading": np.array(heading_list, dtype=np.int64),       # [N]
        "next_pos": np.array(next_pos_list, dtype=np.int64),     # [N, 2]
        "next_heading": np.array(next_heading_list, dtype=np.int64),  # [N]
    }

    np.savez_compressed(save_path, **dataset)
    print(f"\nSaved dataset to: {save_path}")
    print(f"Number of transitions: {len(act_list)}")
    print(f"obs shape: {dataset['obs'].shape}")
    print(f"next_obs shape: {dataset['next_obs'].shape}")
    print(f"actions shape: {dataset['actions'].shape}")


if __name__ == "__main__":
    collect_random_dataset(
        num_episodes=500,
        max_steps_per_episode=30,
        save_path="tiny_nav_dataset.npz",
        seed=0,
    )
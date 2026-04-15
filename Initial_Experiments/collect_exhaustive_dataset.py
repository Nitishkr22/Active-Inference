import numpy as np
from tiny_env import TinyIndoorEnv   # change file name if needed


def inverse_action(action):
    """
    Approximate inverse action for generating a previous frame.
    This is only used to build a meaningful obs_prev for stacked input.

    forward -> stay (can't always invert safely)
    left    -> right
    right   -> left
    stay    -> stay
    """
    if action == TinyIndoorEnv.ACTION_LEFT:
        return TinyIndoorEnv.ACTION_RIGHT
    elif action == TinyIndoorEnv.ACTION_RIGHT:
        return TinyIndoorEnv.ACTION_LEFT
    elif action == TinyIndoorEnv.ACTION_FORWARD:
        return TinyIndoorEnv.ACTION_STAY
    else:
        return TinyIndoorEnv.ACTION_STAY


def apply_action_from_state(env, pos, heading, action):
    """
    Apply one action from a given state without relying on episode rollout.
    Returns:
      next_pos, next_heading
    """
    env.set_state(pos, heading)
    result = env.step(action)
    return result.info["pos"], result.info["heading"], result.observation, result.done


def collect_exhaustive_dataset(
    save_path="tiny_nav_exhaustive_dataset.npz",
):
    env = TinyIndoorEnv()
    free_cells = env.get_free_cells()

    obs_list = []
    act_list = []
    next_obs_list = []
    done_list = []

    pos_list = []
    heading_list = []
    next_pos_list = []
    next_heading_list = []

    obs_prev_list = []

    headings = [env.NORTH, env.EAST, env.SOUTH, env.WEST]
    actions = [
        env.ACTION_FORWARD,
        env.ACTION_LEFT,
        env.ACTION_RIGHT,
        env.ACTION_STAY,
    ]

    total = 0

    for pos in free_cells:
        for heading in headings:
            # current observation
            env.set_state(pos, heading)
            obs = env._render_first_person()

            for action in actions:
                # build an approximate previous observation
                inv_a = inverse_action(action)

                env.set_state(pos, heading)
                if inv_a == env.ACTION_LEFT:
                    prev_heading = (heading - 1) % 4
                    prev_pos = pos
                elif inv_a == env.ACTION_RIGHT:
                    prev_heading = (heading + 1) % 4
                    prev_pos = pos
                else:
                    prev_heading = heading
                    prev_pos = pos

                env.set_state(prev_pos, prev_heading)
                obs_prev = env._render_first_person()

                # true current state again
                env.set_state(pos, heading)
                obs = env._render_first_person()

                # next state after action
                next_pos, next_heading, next_obs, done = apply_action_from_state(env, pos, heading, action)

                obs_prev_list.append(obs_prev.astype(np.float32))
                obs_list.append(obs.astype(np.float32))
                act_list.append(action)
                next_obs_list.append(next_obs.astype(np.float32))
                done_list.append(done)

                pos_list.append(pos)
                heading_list.append(heading)
                next_pos_list.append(next_pos)
                next_heading_list.append(next_heading)

                total += 1

    dataset = {
        "obs_prev": np.stack(obs_prev_list),               # [N, 64, 64]
        "obs": np.stack(obs_list),                         # [N, 64, 64]
        "actions": np.array(act_list, dtype=np.int64),     # [N]
        "next_obs": np.stack(next_obs_list),               # [N, 64, 64]
        "done": np.array(done_list, dtype=np.bool_),       # [N]
        "pos": np.array(pos_list, dtype=np.int64),         # [N, 2]
        "heading": np.array(heading_list, dtype=np.int64), # [N]
        "next_pos": np.array(next_pos_list, dtype=np.int64),       # [N, 2]
        "next_heading": np.array(next_heading_list, dtype=np.int64), # [N]
    }

    np.savez_compressed(save_path, **dataset)

    print(f"Saved exhaustive dataset to: {save_path}")
    print(f"Number of transitions: {total}")
    print("obs_prev:", dataset["obs_prev"].shape)
    print("obs:", dataset["obs"].shape)
    print("actions:", dataset["actions"].shape)
    print("next_obs:", dataset["next_obs"].shape)
    print("pos:", dataset["pos"].shape)
    print("heading:", dataset["heading"].shape)


if __name__ == "__main__":
    collect_exhaustive_dataset()
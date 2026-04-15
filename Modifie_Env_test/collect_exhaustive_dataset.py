## collect_exhaustive_dataset.py ##
import numpy as np
from simulator import TinyIndoorEnv


def inverse_action(action):
    """
    Approximate inverse action for generating a previous frame.

    Used only to construct an approximate previous observation
    for stacked-input experiments.

    forward -> stay   (cannot reliably invert position)
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
    Apply one action from a given state.

    Returns:
        next_pos, next_heading, next_obs, done
    """
    env.set_state(pos, heading)
    result = env.step(action)
    return (
        result.info["pos"],
        result.info["heading"],
        result.observation,
        result.done,
    )


def collect_exhaustive_dataset(
    save_path="../../dataset/tiny_nav_exhaustive_dataset.npz",
    include_all_goals=False,
):
    """
    Exhaustively enumerate all free states and actions.

    If include_all_goals=False:
        - goal is fixed to env.goal_pos
        - observations are still goal-independent
        - goal is stored only as metadata/debug info

    If include_all_goals=True:
        - duplicate each transition for every possible free goal cell
        - useful only if later planner training directly conditions on goal
        - not needed for pure perception/world-model training
    """
    env = TinyIndoorEnv()
    free_cells = env.get_free_cells()

    obs_prev_list = []
    obs_list = []
    act_list = []
    next_obs_list = []
    done_list = []

    pos_list = []
    heading_list = []
    next_pos_list = []
    next_heading_list = []

    goal_list = []

    headings = [env.NORTH, env.EAST, env.SOUTH, env.WEST]
    actions = [
        env.ACTION_FORWARD,
        env.ACTION_LEFT,
        env.ACTION_RIGHT,
        env.ACTION_STAY,
    ]

    if include_all_goals:
        goal_candidates = free_cells
    else:
        goal_candidates = [env.goal_pos]

    total = 0

    for goal_pos in goal_candidates:
        env.set_goal(goal_pos)

        for pos in free_cells:
            for heading in headings:
                # current observation at (pos, heading)
                env.set_state(pos, heading)
                obs = env._render_first_person()

                for action in actions:
                    # approximate previous state only for stacked-frame dataset
                    inv_a = inverse_action(action)

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

                    # restore true current state
                    env.set_state(pos, heading)
                    obs = env._render_first_person()

                    # one-step transition
                    next_pos, next_heading, next_obs, done = apply_action_from_state(
                        env, pos, heading, action
                    )

                    obs_prev_list.append(obs_prev.astype(np.float32))
                    obs_list.append(obs.astype(np.float32))
                    act_list.append(action)
                    next_obs_list.append(next_obs.astype(np.float32))
                    done_list.append(done)

                    pos_list.append(pos)
                    heading_list.append(heading)
                    next_pos_list.append(next_pos)
                    next_heading_list.append(next_heading)
                    goal_list.append(goal_pos)

                    total += 1

    dataset = {
        "obs_prev": np.stack(obs_prev_list),                       # [N, 64, 64]
        "obs": np.stack(obs_list),                                 # [N, 64, 64]
        "actions": np.array(act_list, dtype=np.int64),             # [N]
        "next_obs": np.stack(next_obs_list),                       # [N, 64, 64]
        "done": np.array(done_list, dtype=np.bool_),               # [N]
        "pos": np.array(pos_list, dtype=np.int64),                 # [N, 2]
        "heading": np.array(heading_list, dtype=np.int64),         # [N]
        "next_pos": np.array(next_pos_list, dtype=np.int64),       # [N, 2]
        "next_heading": np.array(next_heading_list, dtype=np.int64),  # [N]
        "goal_pos": np.array(goal_list, dtype=np.int64),           # [N, 2]
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
    print("goal_pos:", dataset["goal_pos"].shape)


if __name__ == "__main__":
    collect_exhaustive_dataset()
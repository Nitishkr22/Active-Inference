# visual_simulator.py

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from simulator import TinyIndoorEnv, Pose, ACTION_NAMES


def print_help() -> None:
    print("\nControls:")
    print("  w : forward")
    print("  a : turn_left")
    print("  d : turn_right")
    print("  s : stay")
    print("  r : reset with random start/goal")
    print("  p : reset with user-defined start/goal")
    print("  g : toggle showing goal in ASCII map")
    print("  h : print help")
    print("  q : quit")


def action_from_key(key: str):
    mapping = {
        "w": "forward",
        "a": "turn_left",
        "d": "turn_right",
        "s": "stay",
    }
    return mapping.get(key, None)


def ask_int(prompt: str, low: int, high: int) -> int:
    while True:
        try:
            val = int(input(prompt).strip())
            if low <= val <= high:
                return val
            print(f"Please enter a value in [{low}, {high}]")
        except ValueError:
            print("Please enter a valid integer.")


def ask_heading() -> str:
    valid = {"N", "E", "S", "W"}
    while True:
        h = input("Enter heading (N/E/S/W): ").strip().upper()
        if h in valid:
            return h
        print("Please enter one of N, E, S, W.")


def find_valid_pose(env: TinyIndoorEnv) -> Pose:
    while True:
        r = ask_int(f"Start row [0, {env.rows - 1}]: ", 0, env.rows - 1)
        c = ask_int(f"Start col [0, {env.cols - 1}]: ", 0, env.cols - 1)
        h = ask_heading()
        try:
            pose = Pose(r, c, h)
            env._validate_pose(pose)   # okay for debugging script
            return pose
        except Exception as e:
            print(f"Invalid pose: {e}")


def find_valid_goal(env: TinyIndoorEnv, start_pose: Pose) -> tuple[int, int]:
    while True:
        r = ask_int(f"Goal row [0, {env.rows - 1}]: ", 0, env.rows - 1)
        c = ask_int(f"Goal col [0, {env.cols - 1}]: ", 0, env.rows - 1)  # fixed below
        # correction:
        if c > env.cols - 1:
            print(f"Please enter a value in [0, {env.cols - 1}]")
            continue

        goal = (r, c)
        try:
            env._validate_goal(goal)   # okay for debugging script
            if not env.is_reachable((start_pose.row, start_pose.col), goal):
                print("Goal is not reachable from the chosen start pose.")
                continue
            return goal
        except Exception as e:
            print(f"Invalid goal: {e}")


def draw(
    env: TinyIndoorEnv,
    obs: np.ndarray,
    info: dict,
    show_goal_in_ascii: bool,
    fig,
    ax_img,
) -> None:
    ax_img.clear()
    ax_img.imshow(obs, cmap="gray", vmin=0.0, vmax=1.0)
    ax_img.set_title("First-Person Observation")
    ax_img.axis("off")

    fig.canvas.draw_idle()

    print("\n" + "=" * 70)
    print("Top-down ASCII map:")
    print(env.render_topdown_ascii(show_goal=show_goal_in_ascii))
    print("-" * 70)
    print(
        f"Pose: row={info['row']}, col={info['col']}, "
        f"heading={info['heading']} ({info['heading_idx']})"
    )
    print(f"Goal: row={info['goal_row']}, col={info['goal_col']}")
    print(
        f"Collision: {info['collision']} | "
        f"Reached goal: {info['reached_goal']} | "
        f"Steps: {info['steps']} | "
        f"Last action: {info['last_action']}"
    )
    print(
        f"Obs stats: shape={obs.shape}, "
        f"min={obs.min():.4f}, max={obs.max():.4f}, mean={obs.mean():.4f}"
    )
    print("=" * 70)


def main() -> None:
    env = TinyIndoorEnv(seed=42)

    show_goal_in_ascii = True

    obs, info = env.reset()

    plt.ion()
    fig, ax_img = plt.subplots(figsize=(5, 5))
    draw(env, obs, info, show_goal_in_ascii, fig, ax_img)

    print_help()

    while True:
        key = input("\nEnter command: ").strip().lower()

        if key == "q":
            print("Exiting.")
            break

        elif key == "h":
            print_help()

        elif key == "g":
            show_goal_in_ascii = not show_goal_in_ascii
            print(f"show_goal_in_ascii = {show_goal_in_ascii}")
            draw(env, obs, info, show_goal_in_ascii, fig, ax_img)

        elif key == "r":
            obs, info = env.reset()
            draw(env, obs, info, show_goal_in_ascii, fig, ax_img)

        elif key == "p":
            print("\nChoose custom start pose:")
            start_pose = find_valid_pose(env)

            while True:
                try:
                    gr = ask_int(f"Goal row [0, {env.rows - 1}]: ", 0, env.rows - 1)
                    gc = ask_int(f"Goal col [0, {env.cols - 1}]: ", 0, env.cols - 1)
                    goal = (gr, gc)
                    env._validate_goal(goal)
                    if not env.is_reachable((start_pose.row, start_pose.col), goal):
                        print("Goal is not reachable from the chosen start pose.")
                        continue
                    break
                except Exception as e:
                    print(f"Invalid goal: {e}")

            obs, info = env.reset(start_pose=start_pose, goal_pos=goal)
            draw(env, obs, info, show_goal_in_ascii, fig, ax_img)

        else:
            action = action_from_key(key)
            if action is None:
                print("Unknown command. Press h for help.")
                continue

            result = env.step(action)
            obs = result.obs
            info = result.info

            draw(env, obs, info, show_goal_in_ascii, fig, ax_img)

            if result.done:
                print("\nGoal reached. Resetting automatically with a new episode.")
                obs, info = env.reset()
                draw(env, obs, info, show_goal_in_ascii, fig, ax_img)

    plt.ioff()
    plt.close(fig)


if __name__ == "__main__":
    main()
# visual_simulator.py

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from simulator import TinyIndoorEnv, Pose


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


def save_frame(obs: np.ndarray, filename: str = "current_view.png") -> None:
    plt.figure(figsize=(5, 5))
    plt.imshow(obs, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title("First-Person Observation")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def draw(env: TinyIndoorEnv, obs: np.ndarray, info: dict, show_goal_in_ascii: bool) -> None:
    save_frame(obs, "current_view.png")

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
    print("Saved image: current_view.png")
    print("=" * 70)


def main() -> None:
    env = TinyIndoorEnv(seed=42)
    show_goal_in_ascii = True

    obs, info = env.reset()
    draw(env, obs, info, show_goal_in_ascii)
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
            draw(env, obs, info, show_goal_in_ascii)

        elif key == "r":
            obs, info = env.reset()
            draw(env, obs, info, show_goal_in_ascii)

        elif key == "p":
            while True:
                try:
                    sr = ask_int(f"Start row [0, {env.rows - 1}]: ", 0, env.rows - 1)
                    sc = ask_int(f"Start col [0, {env.cols - 1}]: ", 0, env.cols - 1)
                    sh = ask_heading()
                    start_pose = Pose(sr, sc, sh)
                    env._validate_pose(start_pose)
                    break
                except Exception as e:
                    print(f"Invalid start pose: {e}")

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
            draw(env, obs, info, show_goal_in_ascii)

        else:
            action = action_from_key(key)
            if action is None:
                print("Unknown command. Press h for help.")
                continue

            result = env.step(action)
            obs = result.obs
            info = result.info
            draw(env, obs, info, show_goal_in_ascii)

            if result.done:
                print("\nGoal reached. Resetting automatically with a new episode.")
                obs, info = env.reset()
                draw(env, obs, info, show_goal_in_ascii)


if __name__ == "__main__":
    main()
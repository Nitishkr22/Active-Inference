"""
verify_habitat.py — Sanity check for Habitat-Sim RGBD setup.

Run with:
  conda activate habitat
  cd /home/nitish/Documents/AIF_code/Active-Inference/Symbolic_AIF_v6/habitat_adapter
  python verify_habitat.py

Checks:
  1. Scene loads without error
  2. RGB observation shape and dtype
  3. Depth observation shape, dtype, and value range (should be metres)
  4. Agent navigates 10 random steps
  5. Saves rgb_sample.png and depth_sample.png for visual inspection
  6. Prints agent pose at each step (position + rotation)
"""

import random
import sys
import numpy as np

# ---- Habitat import ----
try:
    import habitat_sim
except ImportError:
    print("ERROR: habitat_sim not found. Make sure you are in the 'habitat' conda env.")
    sys.exit(1)

SCENE_PATH = "/home/nitish/Documents/habitat_test/data/scene_datasets/habitat-test-scenes/apartment_1.glb"

IMG_H = 256
IMG_W = 256


def build_sim():
    """Configure and return a Habitat-Sim simulator with RGB + Depth sensors."""

    # ---- Scene ----
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE_PATH
    sim_cfg.enable_physics = False   # kinematic only — no physics needed for navigation

    # ---- RGB sensor ----
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid         = "rgb"
    rgb_spec.sensor_type  = habitat_sim.SensorType.COLOR
    rgb_spec.resolution   = [IMG_H, IMG_W]
    rgb_spec.position     = [0.0, 1.5, 0.0]   # 1.5m above agent base
    rgb_spec.hfov         = 90.0              # horizontal FOV degrees

    # ---- Depth sensor ----
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid         = "depth"
    depth_spec.sensor_type  = habitat_sim.SensorType.DEPTH
    depth_spec.resolution   = [IMG_H, IMG_W]
    depth_spec.position     = [0.0, 1.5, 0.0]
    depth_spec.hfov         = 90.0

    # ---- Agent ----
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=0.25)   # 0.25 m per step
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=30.0)   # 30 degrees
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "stop": habitat_sim.agent.ActionSpec("stop"),
    }

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


def get_agent_pose(agent):
    """Return agent position (x, y, z) and heading angle (degrees)."""
    state    = agent.get_state()
    pos      = state.position          # [x, y, z] in Habitat world frame (Y=up)
    rotation = state.rotation          # quaternion
    # Convert quaternion to yaw angle (rotation around Y axis)
    import math
    q = rotation
    # yaw from quaternion (assuming flat-floor navigation, Y is up)
    siny = 2.0 * (q.w * q.y - q.z * q.x)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.x * q.x)
    yaw_deg = math.degrees(math.atan2(siny, cosy))
    return pos, yaw_deg


def main():
    print("=" * 60)
    print("Habitat-Sim RGBD Verification")
    print("=" * 60)
    print(f"Scene  : {SCENE_PATH}")
    print(f"Image  : {IMG_H} x {IMG_W}")
    print(f"HFOV   : 90°")
    print()

    print("Building simulator...", end=" ", flush=True)
    sim = build_sim()
    print("OK")

    agent = sim.get_agent(0)

    # Place agent at a random valid position
    random.seed(42)
    start = sim.pathfinder.get_random_navigable_point()
    state = agent.get_state()
    state.position = start
    agent.set_state(state)
    print(f"Agent placed at: ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})")

    ACTIONS = ["move_forward", "turn_left", "turn_right"]
    ACTION_MAP = {0: "move_forward", 1: "turn_left", 2: "turn_right"}

    print()
    print(f"{'Step':>4} | {'Action':>12} | {'Pos (x, y, z)':>28} | {'Yaw':>8}")
    print("-" * 65)

    rgb_saved = depth_saved = False

    for step in range(10):
        action_name = ACTION_MAP[random.choice([0, 1, 2])]
        obs = sim.step(action_name)

        rgb_obs   = obs["rgb"]      # [H, W, 4]  uint8 RGBA
        depth_obs = obs["depth"]    # [H, W]     float32 metres

        pos, yaw = get_agent_pose(agent)

        if step == 0:
            print()
            print("  First observation:")
            print(f"    RGB   shape = {rgb_obs.shape},  dtype = {rgb_obs.dtype}")
            print(f"    Depth shape = {depth_obs.shape},  dtype = {depth_obs.dtype}")
            print(f"    Depth range = [{depth_obs.min():.2f}, {depth_obs.max():.2f}] metres")
            print()

        print(f"{step+1:>4} | {action_name:>12} | "
              f"({pos[0]:6.2f}, {pos[1]:5.2f}, {pos[2]:6.2f}) | {yaw:7.1f}°")

        # Save first frame for visual inspection
        if not rgb_saved:
            from PIL import Image
            rgb_img = rgb_obs[:, :, :3]   # drop alpha channel
            Image.fromarray(rgb_img).save("rgb_sample.png")
            print(f"    >> Saved rgb_sample.png")
            rgb_saved = True

        if not depth_saved:
            # Normalise depth to 0-255 for visualisation
            d_norm = (depth_obs / depth_obs.max() * 255).astype(np.uint8)
            from PIL import Image
            Image.fromarray(d_norm).save("depth_sample.png")
            print(f"    >> Saved depth_sample.png")
            depth_saved = True

    sim.close()

    print()
    print("=" * 60)
    print("VERIFICATION PASSED")
    print("  rgb_sample.png   — what the agent sees (RGB)")
    print("  depth_sample.png — depth map (brighter = farther)")
    print("=" * 60)


if __name__ == "__main__":
    main()

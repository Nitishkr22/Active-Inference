"""
env_wrapper.py — Wraps Habitat-Sim into the V6 model interface.

What this file does:
  HabitatEnv is the bridge between Habitat-Sim and WorldModelV6.
  On each step it:
    1. Runs the chosen action in Habitat (moves the virtual agent)
    2. Returns the RGBD observation in the format WorldModelV6 expects
    3. Computes the cam_to_v6 transform matrix (camera frame → V6 world frame)
    4. Computes odometry (measured position change, passed to PoseEstimator)

Coordinate system mapping (IMPORTANT):
  Habitat world:  X=right, Y=UP, Z=backward  (Y-up, right-handed)
  V6 world:       x=forward, y=left, z=UP    (Z-up, matches robot convention)

  Transform (Habitat → V6):
    x_v6 = -Z_hab    (V6 forward = Habitat backward flipped)
    y_v6 = -X_hab    (V6 left    = Habitat right flipped)
    z_v6 =  Y_hab    (V6 up      = Habitat up)

  Yaw angle:  θ_v6 = θ_hab (same sign convention — turn left = positive)

V6 action indices:
  0 = move_forward    (0.25 m)
  1 = turn_left       (30°)
  2 = turn_right      (30°)
  3 = stop
"""

from __future__ import annotations

import math
import sys
import os
from typing import Optional, Tuple

import numpy as np
import torch

import habitat_sim

SCENE_PATH = "/home/nitish/Documents/habitat_test/data/scene_datasets/habitat-test-scenes/apartment_1.glb"

# Habitat action name for each V6 action index
_ACTION_NAMES = {
    0: "move_forward",
    1: "turn_left",
    2: "turn_right",
    3: "stop",
}

# 4×4 matrix: Habitat world frame → V6 world frame
#   x_v6 = -Z_hab  →  row: [0,  0, -1, 0]
#   y_v6 = -X_hab  →  row: [-1, 0,  0, 0]
#   z_v6 =  Y_hab  →  row: [0,  1,  0, 0]
_M_HAB_TO_V6 = np.array([
    [0,  0, -1, 0],
    [-1, 0,  0, 0],
    [0,  1,  0, 0],
    [0,  0,  0, 1],
], dtype=np.float32)


class HabitatEnv:
    """
    Thin wrapper around Habitat-Sim that speaks the WorldModelV6 interface.

    Usage:
        env = HabitatEnv()
        rgb, depth, cam_to_v6, pose_v6, info = env.reset()
        goal_v6 = env.sample_goal()

        while not done:
            action = planner.select_action(belief)["best_action"]
            rgb, depth, cam_to_v6, odom, pose_v6, done, info = env.step(action)
    """

    IMG_H   = 256
    IMG_W   = 256
    HFOV    = 90.0
    FWD_M   = 0.25    # metres per forward step
    TURN_DEG = 30.0   # degrees per turn

    def __init__(self, scene_path: str = SCENE_PATH):
        self.scene_path   = scene_path
        self.sim          = self._build_sim()
        self.agent        = self.sim.get_agent(0)
        self._prev_pose   = None            # (x_v6, y_v6, θ_rad) after last step
        self._goal_hab3d: Optional[np.ndarray] = None  # Habitat 3-D goal (preserves Y/height)

    # ------------------------------------------------------------------ #
    # Episode control
    # ------------------------------------------------------------------ #

    def reset(
        self,
        start_pos_hab: Optional[np.ndarray] = None,
        start_yaw_deg: float = 0.0,
    ) -> Tuple:
        """
        Reset to a navigable position.

        start_pos_hab: [3] position in Habitat world (X, Y, Z).
                       If None, a random navigable point is used.

        Returns (rgb, depth, cam_to_v6, pose_v6, info)
          rgb        : torch [3, H, W] float32 0–1
          depth      : torch [1, H, W] float32 metres
          cam_to_v6  : torch [4, 4]
          pose_v6    : torch [3]  (x_fwd, y_left, θ_rad) in V6 frame
          info       : dict with habitat position and yaw
        """
        if start_pos_hab is None:
            start_pos_hab = self.sim.pathfinder.get_random_navigable_point()

        state            = self.agent.get_state()
        state.position   = start_pos_hab
        state.rotation   = _yaw_deg_to_quat(start_yaw_deg)
        self.agent.set_state(state, reset_sensors=True)

        self._prev_pose  = self._get_pose_v6()
        obs              = self.sim.get_sensor_observations()

        rgb, depth, cam_to_v6 = self._process_obs(obs)
        pose_v6                = torch.from_numpy(self._prev_pose)

        info = {
            "hab_position": np.array(start_pos_hab),
            "yaw_deg":      start_yaw_deg,
        }
        return rgb, depth, cam_to_v6, pose_v6, info

    def step(self, action: int) -> Tuple:
        """
        Execute one action in Habitat and return the new observation.

        Returns (rgb, depth, cam_to_v6, odom, pose_v6, done, info)
          odom    : torch [3]  measured world-frame displacement (dx, dy, dθ)
          done    : True if action==3 (stop)
          info    : dict with collision flag, step metrics
        """
        prev_pose = self._prev_pose.copy()

        action_name = _ACTION_NAMES[action]
        obs = self.sim.step(action_name)

        curr_pose        = self._get_pose_v6()
        self._prev_pose  = curr_pose

        # Odometry: world-frame position + heading difference
        dx     = curr_pose[0] - prev_pose[0]
        dy     = curr_pose[1] - prev_pose[1]
        dtheta = curr_pose[2] - prev_pose[2]
        # Wrap heading difference to [-π, π]
        dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))
        odom   = torch.tensor([dx, dy, dtheta], dtype=torch.float32)

        rgb, depth, cam_to_v6 = self._process_obs(obs)
        pose_v6                = torch.from_numpy(curr_pose)
        done                   = (action == 3)

        info = {"collided": self._check_collision(prev_pose, curr_pose, action)}
        return rgb, depth, cam_to_v6, odom, pose_v6, done, info

    # ------------------------------------------------------------------ #
    # Goal sampling
    # ------------------------------------------------------------------ #

    def sample_goal(
        self,
        min_dist: float = 3.0,
        max_dist: float = 8.0,
        max_tries: int  = 200,
    ) -> torch.Tensor:
        """
        Sample a random navigable goal at geodesic distance [min_dist, max_dist]
        from the current agent position.

        Internally stores the 3-D Habitat position (including floor Y) so that
        goal_dist_habitat() can pass the correct Y back to the pathfinder.

        Returns goal_v6: torch [2]  (x_fwd, y_left) in V6 frame.
        """
        current_hab_pos = np.array(self.agent.get_state().position)

        for _ in range(max_tries):
            candidate = self.sim.pathfinder.get_random_navigable_point()
            try:
                d = self.sim.pathfinder.geodesic_distance(current_hab_pos, candidate)
            except Exception:
                continue
            if math.isfinite(d) and min_dist <= d <= max_dist:
                self._goal_hab3d = np.array(candidate, dtype=np.float32)
                return torch.tensor(
                    [-float(candidate[2]), -float(candidate[0])],
                    dtype=torch.float32,
                )

        # Fallback: find any navigable point with a finite geodesic distance
        for _ in range(max_tries):
            candidate = self.sim.pathfinder.get_random_navigable_point()
            try:
                d = self.sim.pathfinder.geodesic_distance(current_hab_pos, candidate)
            except Exception:
                continue
            if math.isfinite(d) and d > 0.5:
                self._goal_hab3d = np.array(candidate, dtype=np.float32)
                return torch.tensor(
                    [-float(candidate[2]), -float(candidate[0])],
                    dtype=torch.float32,
                )

        # Last resort: raw navigable point (may be disconnected)
        candidate = self.sim.pathfinder.get_random_navigable_point()
        self._goal_hab3d = np.array(candidate, dtype=np.float32)
        return torch.tensor(
            [-float(candidate[2]), -float(candidate[0])],
            dtype=torch.float32,
        )

    def goal_dist_habitat(self, goal_v6: torch.Tensor) -> float:
        """
        Geodesic distance from current agent to goal.

        Uses the stored Habitat 3-D position from sample_goal() when available
        (preserves the correct floor height).  Falls back to reconstructing from
        V6 coordinates using the current agent's Y for the goal floor.
        """
        current = np.array(self.agent.get_state().position)
        if self._goal_hab3d is not None:
            goal_hab = self._goal_hab3d
        else:
            goal_hab = np.array([
                -goal_v6[1].item(),  # X_hab = -y_v6
                current[1],          # Y_hab (guessed from agent floor)
                -goal_v6[0].item(),  # Z_hab = -x_v6
            ])
        try:
            return float(self.sim.pathfinder.geodesic_distance(current, goal_hab))
        except Exception:
            return float("inf")

    # ------------------------------------------------------------------ #
    # Observation processing
    # ------------------------------------------------------------------ #

    def _process_obs(self, obs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert raw Habitat observations to V6 tensors."""
        # RGB: [H, W, 4] uint8 RGBA → [3, H, W] float32 0–1
        rgb_np = obs["rgb"][:, :, :3].astype(np.float32) / 255.0
        rgb    = torch.from_numpy(rgb_np).permute(2, 0, 1)   # [3, H, W]

        # Depth: [H, W] float32 metres → [1, H, W]
        depth_np = obs["depth"].astype(np.float32)
        depth    = torch.from_numpy(depth_np).unsqueeze(0)    # [1, H, W]

        cam_to_v6 = self._get_cam_to_v6()
        return rgb, depth, cam_to_v6

    # ------------------------------------------------------------------ #
    # Pose and transform helpers
    # ------------------------------------------------------------------ #

    def _get_pose_v6(self) -> np.ndarray:
        """
        Current agent pose in V6 standard frame: (x_fwd, y_left, θ_rad).

        Conversion:
          x_v6 = -Z_hab,  y_v6 = -X_hab,  θ_v6 = θ_hab_rad
        """
        state  = self.agent.get_state()
        pos    = state.position
        yaw_rad = math.radians(_quat_to_yaw_deg(state.rotation))
        return np.array([-pos[2], -pos[0], yaw_rad], dtype=np.float32)

    def _get_cam_to_v6(self) -> torch.Tensor:
        """
        Build the 4×4 camera-to-V6-world transform matrix.

        Steps:
          1. Get the RGB sensor's world position and orientation from Habitat.
          2. Build cam_to_hab:  standard pinhole camera (x right, y down, z fwd)
                               → Habitat world (x right, Y up, -z fwd).
             The Habitat sensor frame has Y up and Z backward, so we flip Y and Z
             to go from our camera convention to the Habitat sensor frame.
          3. Multiply by M_HAB_TO_V6 to get cam_to_v6.
        """
        sensor_state = self.agent.get_state().sensor_states["rgb"]
        pos_hab      = np.array(sensor_state.position, dtype=np.float32)  # [3]
        R_hab        = _quat_to_rotation_matrix(sensor_state.rotation)    # [3, 3]

        # Standard pinhole (x right, y down, z fwd) → Habitat sensor (x right, y up, z bwd)
        M_flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)

        cam_to_hab         = np.eye(4, dtype=np.float32)
        cam_to_hab[:3, :3] = R_hab @ M_flip
        cam_to_hab[:3,  3] = pos_hab

        cam_to_v6 = _M_HAB_TO_V6 @ cam_to_hab
        return torch.from_numpy(cam_to_v6)

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    def _check_collision(
        self,
        prev_pose: np.ndarray,
        curr_pose: np.ndarray,
        action: int,
    ) -> bool:
        """Heuristic: if a forward action produced very little movement, it was blocked."""
        if action != 0:
            return False
        moved = math.sqrt((curr_pose[0] - prev_pose[0])**2 +
                          (curr_pose[1] - prev_pose[1])**2)
        return moved < 0.05   # expected 0.25 m; blocked if <5 cm

    def _build_sim(self) -> habitat_sim.Simulator:
        sim_cfg           = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_id  = self.scene_path
        sim_cfg.enable_physics = False

        rgb_spec             = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid        = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution  = [self.IMG_H, self.IMG_W]
        rgb_spec.position    = [0.0, 1.5, 0.0]
        rgb_spec.hfov        = self.HFOV

        depth_spec             = habitat_sim.CameraSensorSpec()
        depth_spec.uuid        = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution  = [self.IMG_H, self.IMG_W]
        depth_spec.position    = [0.0, 1.5, 0.0]
        depth_spec.hfov        = self.HFOV

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=self.FWD_M)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=self.TURN_DEG)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=self.TURN_DEG)
            ),
            "stop": habitat_sim.agent.ActionSpec("stop"),
        }

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        return habitat_sim.Simulator(cfg)

    def close(self):
        self.sim.close()


# ------------------------------------------------------------------ #
# Quaternion utilities
# ------------------------------------------------------------------ #

def _quat_to_yaw_deg(q) -> float:
    """Extract yaw (rotation around Y-up axis) from a Habitat quaternion."""
    siny = 2.0 * (q.w * q.y - q.z * q.x)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.x * q.x)
    return math.degrees(math.atan2(siny, cosy))


def _quat_to_rotation_matrix(q) -> np.ndarray:
    """Quaternion with .w .x .y .z attributes → 3×3 rotation matrix (float32)."""
    w, x, y, z = q.w, q.x, q.y, q.z
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)     ],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),  2*(y*z - w*x)     ],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def _yaw_deg_to_quat(yaw_deg: float):
    """
    Create a numpy.quaternion for a yaw rotation around the Y-up axis.
    Uses habitat_sim.utils.common.quat_from_angle_axis which returns
    numpy.quaternion — the type Habitat's set_state() expects.
    """
    from habitat_sim.utils.common import quat_from_angle_axis
    return quat_from_angle_axis(
        math.radians(yaw_deg),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )

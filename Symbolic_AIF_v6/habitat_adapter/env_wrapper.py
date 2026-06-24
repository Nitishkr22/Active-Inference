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
        max_tries: int  = 500,
    ) -> torch.Tensor:
        """
        Sample a random navigable goal at geodesic distance [min_dist, max_dist]
        from the current agent position.

        Strategy 1 (primary): Radial sampling around the current position.
          Draw a random distance + angle, offset in V6 frame, convert to Habitat 3D,
          snap to the navmesh. Snapped points stay on the same floor and are far
          more likely to be in the same connected navmesh component than purely
          random navigable points scattered across the scene.

        Strategy 2 (fallback): Pure random navigable points (the original approach).

        Internally stores the 3-D Habitat position (including floor Y) so that
        goal_dist_habitat() can pass the correct Y back to the pathfinder.

        Returns goal_v6: torch [2]  (x_fwd, y_left) in V6 frame.
        """
        current_hab_pos = np.array(self.agent.get_state().position)
        # Snap start to navmesh — state.position can be a few mm off the mesh,
        # which causes geodesic_distance to return inf for ALL queries.
        current_snapped = self.sim.pathfinder.snap_point(current_hab_pos)
        if np.any(np.isnan(current_snapped)):
            current_snapped = current_hab_pos
        # V6 y of the start — used to reject goals too far south of the corridor.
        start_y_v6 = -float(current_hab_pos[0])

        def _to_v6(pt_hab: np.ndarray) -> torch.Tensor:
            return torch.tensor(
                [-float(pt_hab[2]), -float(pt_hab[0])],
                dtype=torch.float32,
            )

        def _euclidean_v6(a: np.ndarray, b: np.ndarray) -> float:
            """Euclidean distance between two Habitat 3D points in V6 XY plane."""
            return math.sqrt((-a[2] - -b[2])**2 + (-a[0] - -b[0])**2)

        def _try_geodesic(cand: np.ndarray) -> Optional[torch.Tensor]:
            """Accept candidate if geodesic distance is in [min_dist, max_dist].
            Uses the navmesh-snapped start so the pathfinder can locate it."""
            if np.any(np.isnan(cand)):
                return None
            try:
                d = self.sim.pathfinder.geodesic_distance(current_snapped, cand)
            except Exception:
                return None
            if math.isfinite(d) and min_dist <= d <= max_dist:
                self._goal_hab3d = np.array(cand, dtype=np.float32)
                return _to_v6(cand)
            return None

        def _try_euclidean(cand: np.ndarray) -> Optional[torch.Tensor]:
            """Accept candidate if Euclidean distance is in [min_dist, max_dist],
            the point is navigable, and it is not too far south of the start
            (avoids rooms behind walls in the south of the apartment)."""
            if np.any(np.isnan(cand)):
                return None
            try:
                if not self.sim.pathfinder.is_navigable(cand):
                    return None
            except Exception:
                return None
            # Reject goals more than 0.5 m south (in V6 y) of the start.
            # Near the start position the south wall is ~0.5 m away; allowing
            # goals further south leads to unreachable rooms behind that wall.
            goal_y_v6 = -float(cand[0])
            if goal_y_v6 < start_y_v6 - 0.5:
                return None
            d = _euclidean_v6(current_hab_pos, cand)
            if min_dist <= d <= max_dist:
                self._goal_hab3d = np.array(cand, dtype=np.float32)
                return _to_v6(cand)
            return None

        def _make_radial(dist: float, angle: float) -> np.ndarray:
            """Radial offset in V6 frame → Habitat 3D candidate (same floor Y)."""
            dx_v6 = dist * math.cos(angle)
            dy_v6 = dist * math.sin(angle)
            return np.array([
                current_hab_pos[0] - dy_v6,
                current_hab_pos[1],
                current_hab_pos[2] - dx_v6,
            ], dtype=np.float32)

        # Strategy 1: radial snap + geodesic check.
        # Radial offsets from current position convert correctly:
        #   x_v6 = -Z_hab  →  Z_hab = Z_start - dx_v6
        #   y_v6 = -X_hab  →  X_hab = X_start - dy_v6
        for _ in range(max_tries):
            raw = _make_radial(np.random.uniform(min_dist, max_dist),
                               np.random.uniform(0.0, 2.0 * math.pi))
            result = _try_geodesic(self.sim.pathfinder.snap_point(raw))
            if result is not None:
                return result

        # Strategy 2: random navigable points + geodesic check.
        for _ in range(max_tries):
            result = _try_geodesic(self.sim.pathfinder.get_random_navigable_point())
            if result is not None:
                return result

        # Strategy 3: radial snap + Euclidean check (geodesic unavailable on this scene).
        # Goal may be in a different room, but at least the right Euclidean distance.
        for _ in range(max_tries):
            raw = _make_radial(np.random.uniform(min_dist, max_dist),
                               np.random.uniform(0.0, 2.0 * math.pi))
            result = _try_euclidean(self.sim.pathfinder.snap_point(raw))
            if result is not None:
                return result

        # Last resort: any reachable navigable point
        for _ in range(max_tries):
            cand = self.sim.pathfinder.get_random_navigable_point()
            try:
                d = self.sim.pathfinder.geodesic_distance(current_hab_pos, cand)
            except Exception:
                continue
            if math.isfinite(d) and d > 0.5:
                self._goal_hab3d = np.array(cand, dtype=np.float32)
                return _to_v6(cand)

        # Absolute last resort: raw random navigable point (may be disconnected)
        cand = self.sim.pathfinder.get_random_navigable_point()
        self._goal_hab3d = np.array(cand, dtype=np.float32)
        return _to_v6(cand)

    def get_path_waypoints(
        self,
        goal_v6: torch.Tensor,
        waypoint_spacing: float = 1.0,
    ) -> Optional[List[torch.Tensor]]:
        """
        Use the Habitat pathfinder to find the navigable path from the current
        agent position to goal_v6, and return it as a list of V6 [x, y] tensors.

        Waypoints are spaced ~waypoint_spacing metres apart along the path.
        Returns None if no path exists (different navmesh components).
        The first element is always close to the current position; the last is
        the goal.
        """
        try:
            from habitat_sim.nav import ShortestPath
        except ImportError:
            return None

        start = np.array(self.agent.get_state().position)
        # snap_point() returns a Magnum Vector3, not a numpy array.
        # Convert element-by-element to avoid the ".tolist()" swizzle error.
        _snapped = self.sim.pathfinder.snap_point(start)
        try:
            start_snapped = [float(_snapped[0]), float(_snapped[1]), float(_snapped[2])]
            if any(math.isnan(v) for v in start_snapped):
                start_snapped = start.tolist()
        except Exception:
            start_snapped = start.tolist()

        goal_hab = (
            self._goal_hab3d
            if self._goal_hab3d is not None
            else np.array([-goal_v6[1].item(), start[1], -goal_v6[0].item()])
        )

        path = ShortestPath()
        path.requested_start = start_snapped
        path.requested_end   = [float(goal_hab[0]), float(goal_hab[1]), float(goal_hab[2])]
        if not self.sim.pathfinder.find_path(path):
            return None

        pts = path.points          # list of mn.Vector3 along the path
        if len(pts) < 2:
            return None

        # Subsample: keep a point every ~waypoint_spacing metres + always keep last
        waypoints: List[torch.Tensor] = []
        prev  = np.array([float(pts[0][0]), float(pts[0][2])])  # (X_hab, Z_hab)
        accum = 0.0
        for i, pt in enumerate(pts):
            curr = np.array([float(pt[0]), float(pt[2])])
            accum += float(np.linalg.norm(curr - prev))
            prev  = curr
            if accum >= waypoint_spacing or i == len(pts) - 1:
                waypoints.append(torch.tensor(
                    [-float(pt[2]), -float(pt[0])],  # V6: x=-Z_hab, y=-X_hab
                    dtype=torch.float32,
                ))
                accum = 0.0

        return waypoints if waypoints else None

    def goal_dist_habitat(self, goal_v6: torch.Tensor) -> float:
        """
        Geodesic distance from current agent to goal.

        Snaps the agent position to the navmesh before querying (state.position
        can be a few mm off-mesh, which causes geodesic_distance to return inf).
        """
        current = np.array(self.agent.get_state().position)
        current_snapped = self.sim.pathfinder.snap_point(current)
        if np.any(np.isnan(current_snapped)):
            current_snapped = current
        if self._goal_hab3d is not None:
            goal_hab = self._goal_hab3d
        else:
            goal_hab = np.array([
                -goal_v6[1].item(),  # X_hab = -y_v6
                current[1],          # Y_hab (guessed from agent floor)
                -goal_v6[0].item(),  # Z_hab = -x_v6
            ])
        try:
            return float(self.sim.pathfinder.geodesic_distance(current_snapped, goal_hab))
        except Exception:
            return float("inf")

    # ------------------------------------------------------------------ #
    # LiDAR simulation from depth
    # ------------------------------------------------------------------ #

    @staticmethod
    def depth_to_lidar(
        depth: torch.Tensor,
        n_beams: int  = 64,
        hfov_deg: float = 90.0,
        band_frac: float = 0.08,   # fraction of image height used for horizontal band
        clip_max: float  = 8.0,    # metres — clip noisy far readings
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate a 1-D horizontal LiDAR scan from a depth image.

        The camera's horizontal FOV covers the robot's forward 90°.
        Left pixel  → angle = +hfov/2 (V6 left = positive y)
        Right pixel → angle = -hfov/2 (V6 right = negative y)
        Center pixel → angle = 0 (straight ahead)

        Returns
        -------
        angles_rad : [n_beams]  beam angles in V6 frame radians
        dists_m    : [n_beams]  distances in metres
        """
        d = depth.squeeze().float()                          # [H, W]
        H, W = d.shape

        # Average a horizontal band in the middle of the image
        bw   = max(1, int(H * band_frac))
        cy   = H // 2
        band = d[max(0, cy - bw // 2): cy + bw // 2 + 1].mean(0)   # [W]

        # Clamp to valid depth range
        band = band.clamp(0.0, clip_max)

        # Resample to n_beams evenly
        idx        = torch.linspace(0, W - 1, n_beams).long()
        dists_m    = band[idx]                                        # [n_beams]

        # Beam angles: left → +hfov/2, right → -hfov/2
        # (image col 0 = left of image = V6 left = positive y angle)
        half_fov   = math.radians(hfov_deg / 2.0)
        angles_rad = torch.linspace(half_fov, -half_fov, n_beams)    # [n_beams]

        return angles_rad, dists_m

    @staticmethod
    def lidar_min_forward(
        angles_rad: torch.Tensor,
        dists_m:    torch.Tensor,
        cone_deg:   float = 20.0,
    ) -> float:
        """Minimum distance to obstacle in the forward cone (±cone_deg)."""
        cone = math.radians(cone_deg)
        fwd  = dists_m[torch.abs(angles_rad) <= cone]
        return float(fwd.min()) if fwd.numel() > 0 else float("inf")

    @staticmethod
    def lidar_side_clearance(
        angles_rad: torch.Tensor,
        dists_m:    torch.Tensor,
    ) -> Tuple[float, float]:
        """Mean clearance on left (+) and right (-) halves."""
        left  = dists_m[angles_rad > 0].mean()
        right = dists_m[angles_rad < 0].mean()
        return float(left), float(right)

    def get_occupancy_gt(self, resolution: float = 0.1, size: int = 64) -> np.ndarray:
        """
        Build a square binary occupancy grid centred on the current agent.

        grid[r, c] = 1  →  cell is occupied (not navigable)
        grid[r, c] = 0  →  cell is free (navigable)

        resolution : metres per cell
        size       : grid side length in cells  (size×size)
        """
        pf  = self.sim.pathfinder
        pos = np.array(self.agent.get_state().position)   # Habitat XYZ
        half = (size // 2) * resolution
        grid = np.zeros((size, size), dtype=np.float32)
        for ri in range(size):
            z_off = (ri - size // 2) * resolution        # forward axis (V6 x → Hab -Z)
            for ci in range(size):
                x_off = (ci - size // 2) * resolution    # left axis   (V6 y → Hab -X)
                pt = np.array([
                    pos[0] - x_off,      # Hab X = -y_v6
                    pos[1],              # keep floor height
                    pos[2] - z_off,      # Hab Z = -x_v6
                ], dtype=np.float32)
                grid[ri, ci] = 0.0 if pf.is_navigable(pt) else 1.0
        return grid                                         # [size, size]

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

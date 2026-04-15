# simulator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import math
import random

import numpy as np


# ============================================================
# Constants
# ============================================================

HEADINGS = ["N", "E", "S", "W"]
HEADING_TO_IDX = {h: i for i, h in enumerate(HEADINGS)}

ACTION_NAMES = ["forward", "turn_left", "turn_right", "stay"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_NAMES)}


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class Pose:
    row: int
    col: int
    heading: str  # one of {"N", "E", "S", "W"}


@dataclass
class StepResult:
    obs: np.ndarray                  # [H, W], float32 in [0, 1]
    reward: float
    done: bool
    info: Dict


# ============================================================
# Simulator
# ============================================================

class TinyIndoorEnv:
    """
    Grid world with:
      - wall/appearance symbols: #, A, B, C, L, M
      - traversable symbols: ., P
      - goal stored internally but NOT rendered into observations

    First-person renderer:
      - grayscale image
      - simple ray-cast style wall slices
      - different symbols produce different brightness/patterns
      - floor patches influence appearance near the lower image region

    The goal is task metadata only; it does not appear in rendered images.
    """

    def __init__(
        self,
        grid_map: Optional[List[str]] = None,
        obs_height: int = 64,
        obs_width: int = 64,
        fov_deg: float = 70.0,
        max_view_dist: int = 20,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
        collision_penalty: float = -0.02,
        seed: Optional[int] = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.obs_height = obs_height
        self.obs_width = obs_width
        self.fov_deg = fov_deg
        self.max_view_dist = max_view_dist

        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.collision_penalty = collision_penalty

        if grid_map is None:
            grid_map = self._default_map()

        self.grid = self._parse_grid(grid_map)
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        self._validate_grid()

        self.blocking_symbols = {"#", "A", "B", "C", "L", "M"}
        self.traversable_symbols = {".", "P"}

        # Internal state
        self.pose: Optional[Pose] = None
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.last_action: Optional[str] = None
        self.steps: int = 0

        # Cache traversable cells for sampling
        self.free_cells = self._find_free_cells()

        if len(self.free_cells) == 0:
            raise ValueError("No traversable cells found in the map.")

    # ========================================================
    # Public API
    # ========================================================

    def reset(
        self,
        start_pose: Optional[Pose] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        min_goal_dist: int = 3,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment.

        Args:
            start_pose: Optional fixed start pose.
            goal_pos: Optional fixed goal position (stored, not rendered).
            min_goal_dist: Minimum Manhattan distance between start and goal
                           when both are sampled / partially sampled.

        Returns:
            obs, info
        """
        if start_pose is None:
            start_row, start_col = self.rng.choice(self.free_cells)
            start_heading = self.rng.choice(HEADINGS)
            start_pose = Pose(start_row, start_col, start_heading)
        else:
            self._validate_pose(start_pose)

        if goal_pos is None:
            goal_pos = self._sample_goal_far_from(
                start=(start_pose.row, start_pose.col),
                min_goal_dist=min_goal_dist,
            )
        else:
            self._validate_goal(goal_pos)
            dist = abs(goal_pos[0] - start_pose.row) + abs(goal_pos[1] - start_pose.col)
            if dist < min_goal_dist:
                # Allowed, but warn via info later if needed
                pass

        self.pose = start_pose
        self.goal_pos = goal_pos
        self.last_action = None
        self.steps = 0

        obs = self.render_first_person()
        info = self._build_info(collision=False, reached_goal=False)

        return obs, info

    def step(self, action: int | str) -> StepResult:
        """
        Apply action and return next observation.

        Actions:
            0 / "forward"
            1 / "turn_left"
            2 / "turn_right"
            3 / "stay"
        """
        if self.pose is None or self.goal_pos is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        action_name = self._normalize_action(action)
        self.last_action = action_name
        self.steps += 1

        collision = False
        reached_goal = False
        reward = self.step_penalty

        if action_name == "turn_left":
            self.pose.heading = HEADINGS[(HEADING_TO_IDX[self.pose.heading] - 1) % 4]

        elif action_name == "turn_right":
            self.pose.heading = HEADINGS[(HEADING_TO_IDX[self.pose.heading] + 1) % 4]

        elif action_name == "stay":
            pass

        elif action_name == "forward":
            dr, dc = self._heading_to_delta(self.pose.heading)
            nr, nc = self.pose.row + dr, self.pose.col + dc

            if not self._in_bounds(nr, nc) or self._is_blocking(nr, nc):
                collision = True
                reward += self.collision_penalty
            else:
                self.pose.row = nr
                self.pose.col = nc

        else:
            raise ValueError(f"Unsupported action: {action_name}")

        if (self.pose.row, self.pose.col) == self.goal_pos:
            reached_goal = True
            reward += self.goal_reward

        done = reached_goal
        obs = self.render_first_person()
        info = self._build_info(collision=collision, reached_goal=reached_goal)

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def render_first_person(self) -> np.ndarray:
        """
        Render a first-person grayscale observation.

        Returns:
            obs: float32 array of shape [obs_height, obs_width], values in [0, 1]
        """
        if self.pose is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        H = self.obs_height
        W = self.obs_width

        img = np.zeros((H, W), dtype=np.float32)

        # Simple sky / ceiling and ground base
        img[: H // 2, :] = 0.02
        img[H // 2 :, :] = 0.08

        base_angle = self._heading_to_angle_deg(self.pose.heading)
        half_fov = self.fov_deg / 2.0

        for x in range(W):
            ray_angle = base_angle - half_fov + (x / max(W - 1, 1)) * self.fov_deg
            hit = self._cast_ray(ray_angle)

            dist = max(hit["distance"], 1e-3)
            symbol = hit["symbol"]
            hit_row, hit_col = hit["cell"]

            # Inverse-distance wall height heuristic
            wall_height = int(min(H, max(2, (H * 0.95) / (dist + 0.25))))
            y0 = max(0, (H - wall_height) // 2)
            y1 = min(H, y0 + wall_height)

            wall_column = self._make_wall_column(
                height=wall_height,
                symbol=symbol,
                dist=dist,
                ray_angle_deg=ray_angle,
                hit_row=hit_row,
                hit_col=hit_col,
            )

            img[y0:y1, x] = wall_column

            # Add floor cues below the wall using forward sample positions
            if y1 < H:
                floor_vals = self._make_floor_column(
                    start_y=y1,
                    H=H,
                    ray_angle_deg=ray_angle,
                )
                img[y1:H, x] = floor_vals

        # Mild vignette-like darkening near edges
        xx = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        vignette = 1.0 - 0.08 * (xx ** 2)
        img *= vignette[None, :]

        # Tiny observation noise helps prevent overfitting to exact pixel patterns
        noise = self.np_rng.normal(loc=0.0, scale=0.004, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

        return img.astype(np.float32)

    def render_topdown_ascii(self, show_goal: bool = True) -> str:
        """
        Debug text rendering of the map with current agent pose.

        show_goal controls whether goal is shown in this ASCII debug view.
        It does NOT affect first-person observations.
        """
        if self.pose is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        arrow = {"N": "^", "E": ">", "S": "v", "W": "<"}[self.pose.heading]

        rows = []
        for r in range(self.rows):
            row_chars = []
            for c in range(self.cols):
                ch = self.grid[r][c]
                if (r, c) == (self.pose.row, self.pose.col):
                    row_chars.append(arrow)
                elif show_goal and self.goal_pos is not None and (r, c) == self.goal_pos:
                    row_chars.append("G")
                else:
                    row_chars.append(ch)
            rows.append(" ".join(row_chars))
        return "\n".join(rows)

    def sample_random_action(self) -> int:
        return self.rng.randrange(len(ACTION_NAMES))

    def get_pose_tuple(self) -> Tuple[int, int, int]:
        if self.pose is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        return self.pose.row, self.pose.col, HEADING_TO_IDX[self.pose.heading]

    # ========================================================
    # Internal map helpers
    # ========================================================

    def _default_map(self) -> List[str]:
        """
        A controlled asymmetric map with:
          - structural walls (#)
          - wall texture variants (A, B, C)
          - landmark walls (L, M)
          - floor patches (P)
          - traversable free cells (.)

        Goal is NOT part of the static map.
        """
        return [
            "##########",
            "#...A....#",
            "#.###.##L#",
            "#.#..P...#",
            "#.#.###..#",
            "#...#..B.#",
            "###.#.##.#",
            "#..M#..P.#",
            "#....C...#",
            "##########",
        ]

    def _parse_grid(self, grid_map: List[str]) -> List[List[str]]:
        if len(grid_map) == 0:
            raise ValueError("grid_map cannot be empty.")
        width = len(grid_map[0])
        if any(len(row) != width for row in grid_map):
            raise ValueError("All rows in grid_map must have equal length.")
        return [list(row) for row in grid_map]

    def _validate_grid(self) -> None:
        valid_symbols = {"#", "A", "B", "C", "L", "M", "P", "."}
        for r in range(self.rows):
            for c in range(self.cols):
                ch = self.grid[r][c]
                if ch not in valid_symbols:
                    raise ValueError(
                        f"Invalid map symbol '{ch}' at ({r}, {c}). "
                        f"Allowed: {sorted(valid_symbols)}"
                    )

    def _find_free_cells(self) -> List[Tuple[int, int]]:
        free = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] in self.traversable_symbols:
                    free.append((r, c))
        return free

    def _validate_pose(self, pose: Pose) -> None:
        if pose.heading not in HEADINGS:
            raise ValueError(f"Invalid heading: {pose.heading}")
        if not self._in_bounds(pose.row, pose.col):
            raise ValueError(f"Pose out of bounds: {pose}")
        if self._is_blocking(pose.row, pose.col):
            raise ValueError(f"Start pose is on a blocking cell: {pose}")

    def _validate_goal(self, goal_pos: Tuple[int, int]) -> None:
        r, c = goal_pos
        if not self._in_bounds(r, c):
            raise ValueError(f"Goal out of bounds: {goal_pos}")
        if self._is_blocking(r, c):
            raise ValueError(f"Goal is on a blocking cell: {goal_pos}")

    def _sample_goal_far_from(
        self,
        start: Tuple[int, int],
        min_goal_dist: int,
    ) -> Tuple[int, int]:
        candidates = []
        sr, sc = start
        for gr, gc in self.free_cells:
            dist = abs(gr - sr) + abs(gc - sc)
            if dist >= min_goal_dist:
                candidates.append((gr, gc))

        if not candidates:
            candidates = list(self.free_cells)

        return self.rng.choice(candidates)

    # ========================================================
    # Internal dynamics helpers
    # ========================================================

    def _normalize_action(self, action: int | str) -> str:
        if isinstance(action, int):
            if action < 0 or action >= len(ACTION_NAMES):
                raise ValueError(f"Action index out of range: {action}")
            return ACTION_NAMES[action]
        if isinstance(action, str):
            if action not in ACTION_TO_IDX:
                raise ValueError(f"Unknown action string: {action}")
            return action
        raise TypeError(f"Action must be int or str, got {type(action)}")

    def _heading_to_delta(self, heading: str) -> Tuple[int, int]:
        if heading == "N":
            return -1, 0
        if heading == "E":
            return 0, 1
        if heading == "S":
            return 1, 0
        if heading == "W":
            return 0, -1
        raise ValueError(f"Invalid heading: {heading}")

    def _heading_to_angle_deg(self, heading: str) -> float:
        # 0 deg = east/right, 90 = south/down in grid row-col geometry
        if heading == "E":
            return 0.0
        if heading == "S":
            return 90.0
        if heading == "W":
            return 180.0
        if heading == "N":
            return 270.0
        raise ValueError(f"Invalid heading: {heading}")

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_blocking(self, r: int, c: int) -> bool:
        return self.grid[r][c] in self.blocking_symbols

    def _build_info(self, collision: bool, reached_goal: bool) -> Dict:
        assert self.pose is not None
        assert self.goal_pos is not None
        return {
            "row": self.pose.row,
            "col": self.pose.col,
            "heading": self.pose.heading,
            "heading_idx": HEADING_TO_IDX[self.pose.heading],
            "goal_row": self.goal_pos[0],
            "goal_col": self.goal_pos[1],
            "collision": collision,
            "reached_goal": reached_goal,
            "steps": self.steps,
            "last_action": self.last_action,
        }

    # ========================================================
    # Ray casting and rendering helpers
    # ========================================================

    def _cast_ray(self, ray_angle_deg: float) -> Dict:
        """
        March a ray from the agent center until hitting a blocking cell or
        max_view_dist. Returns the first blocking symbol encountered.
        """
        assert self.pose is not None

        angle_rad = math.radians(ray_angle_deg)
        dx = math.cos(angle_rad)   # col direction
        dy = math.sin(angle_rad)   # row direction

        # Agent center in continuous coordinates
        r0 = self.pose.row + 0.5
        c0 = self.pose.col + 0.5

        step_size = 0.03
        dist = 0.0

        last_r, last_c = self.pose.row, self.pose.col

        while dist < self.max_view_dist:
            rr = r0 + dy * dist
            cc = c0 + dx * dist
            gr = int(rr)
            gc = int(cc)

            if not self._in_bounds(gr, gc):
                return {
                    "distance": dist,
                    "symbol": "#",
                    "cell": (last_r, last_c),
                }

            last_r, last_c = gr, gc

            if self._is_blocking(gr, gc):
                return {
                    "distance": dist,
                    "symbol": self.grid[gr][gc],
                    "cell": (gr, gc),
                }

            dist += step_size

        return {
            "distance": float(self.max_view_dist),
            "symbol": "#",
            "cell": (last_r, last_c),
        }

    def _symbol_base_intensity(self, symbol: str) -> float:
        """
        Base wall brightness by symbol.
        """
        mapping = {
            "#": 0.85,
            "A": 0.95,
            "B": 0.70,
            "C": 0.50,
            "L": 0.88,
            "M": 0.62,
        }
        return mapping.get(symbol, 0.80)

    def _make_wall_column(
        self,
        height: int,
        symbol: str,
        dist: float,
        ray_angle_deg: float,
        hit_row: int,
        hit_col: int,
    ) -> np.ndarray:
        """
        Produce a 1D column for a wall slice.
        """
        col = np.ones((height,), dtype=np.float32) * self._symbol_base_intensity(symbol)

        # Distance attenuation
        attenuation = 1.0 / (1.0 + 0.12 * dist * dist)
        col *= attenuation * 1.8
        col = np.clip(col, 0.0, 1.0)

        # Vertical shading: slightly darker toward bottom
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)
        col *= (0.95 - 0.15 * yy)

        # Symbol-specific texture
        if symbol == "#":
            # Plain wall with mild grain
            grain = 0.015 * np.sin(np.linspace(0, 10, height, dtype=np.float32))
            col += grain

        elif symbol == "A":
            # Brighter stripe-like texture
            stripes = 0.08 * (np.sin(np.linspace(0, 22, height, dtype=np.float32)) > 0).astype(np.float32)
            col += stripes

        elif symbol == "B":
            # Medium wall with softer banding
            bands = 0.05 * np.sin(np.linspace(0, 16, height, dtype=np.float32))
            col += bands

        elif symbol == "C":
            # Darker wall with sparse highlight
            highlight = np.zeros((height,), dtype=np.float32)
            highlight[::5] = 0.06
            col += highlight

        elif symbol == "L":
            # Landmark 1: strong repeated pulses
            pulses = 0.12 * (np.sin(np.linspace(0, 28, height, dtype=np.float32)) > 0.4).astype(np.float32)
            col += pulses

        elif symbol == "M":
            # Landmark 2: alternating dark/light sections
            alt = np.zeros((height,), dtype=np.float32)
            alt[::6] = 0.10
            alt[3::6] = -0.06
            col += alt

        # Very small cell-dependent modulation
        cell_mod = 0.015 * (((hit_row + hit_col) % 3) - 1)
        col += cell_mod

        # Slight angular modulation
        angle_mod = 0.01 * math.sin(math.radians(ray_angle_deg * 3.0))
        col += angle_mod

        return np.clip(col, 0.0, 1.0).astype(np.float32)

    def _make_floor_column(
        self,
        start_y: int,
        H: int,
        ray_angle_deg: float,
    ) -> np.ndarray:
        """
        Create floor appearance below the wall slice.
        Floor patches 'P' are visible as different brightness near the lower region
        based on approximate forward sampling along the ray.
        """
        assert self.pose is not None

        floor = np.zeros((H - start_y,), dtype=np.float32)

        angle_rad = math.radians(ray_angle_deg)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        r0 = self.pose.row + 0.5
        c0 = self.pose.col + 0.5

        for idx, y in enumerate(range(start_y, H)):
            # Perspective-inspired depth estimate:
            # lower pixels correspond to closer floor points
            rel = (y - H / 2) / max(H / 2, 1)
            rel = max(rel, 1e-3)
            sample_dist = 0.5 + 1.7 / rel

            rr = r0 + dy * sample_dist
            cc = c0 + dx * sample_dist
            gr = int(rr)
            gc = int(cc)

            base = 0.10 + 0.12 * (idx / max(len(floor), 1))

            if self._in_bounds(gr, gc):
                symbol = self.grid[gr][gc]
                if symbol == "P":
                    base += 0.16
                elif symbol in {"A", "L"}:
                    base += 0.03
                elif symbol in {"C", "M"}:
                    base -= 0.03

            floor[idx] = base

        return np.clip(floor, 0.0, 1.0).astype(np.float32)


# ============================================================
# Simple manual test
# ============================================================

if __name__ == "__main__":
    env = TinyIndoorEnv(seed=42)

    obs, info = env.reset()
    print("Initial top-down map:")
    print(env.render_topdown_ascii(show_goal=True))
    print("\nInitial info:", info)
    print("Observation shape:", obs.shape, "dtype:", obs.dtype, "min/max:", obs.min(), obs.max())

    demo_actions = ["forward", "forward", "turn_right", "forward", "stay", "turn_left", "forward", "forward","turn_left", "forward", "turn_right", "forward", "forward", "forward", 
    "turn_left", "forward", "forward", "forward", "turn_right", "forward", "forward","turn_left", "forward", "turn_left", "forward", "forward", "turn_right", "forward", "forward",
     "turn_left", "forward", "forward", "turn_right", "forward", "forward", "forward", "forward","turn_right", "forward", "forward", "forward", "forward", "turn_right"
     , "forward", "forward"]

    for i, action in enumerate(demo_actions, start=1):
        result = env.step(action)
        print(f"\nStep {i} | action={action}")
        print(env.render_topdown_ascii(show_goal=True))
        print("info:", result.info)
        print("reward:", result.reward, "done:", result.done)
        print("obs stats:", result.obs.shape, result.obs.min(), result.obs.max())
        if result.done:
            print("Reached goal.")
            break
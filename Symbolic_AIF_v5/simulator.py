# simulator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
import math
import random
from collections import deque

import numpy as np


# ============================================================
# Constants
# ============================================================

HEADINGS = ["N", "E", "S", "W"]
HEADING_TO_IDX = {h: i for i, h in enumerate(HEADINGS)}

ACTION_NAMES = ["forward", "backward", "turn_left", "turn_right"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_NAMES)}


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class Pose:
    row: int
    col: int
    heading: str


@dataclass
class StepResult:
    obs: np.ndarray
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
      - optional goal stored internally but NOT rendered into observations

    Actions:
      - forward
      - backward
      - turn_left
      - turn_right

    Important:
      - Goal never appears in rendered first-person observations.
      - Map is fully connected over traversable cells.
      - For dataset generation, use reset(use_goal=False).

    Patch design:
      - Patches are fixed in the world
      - Patches are attached to specific wall cells
      - Patches are rendered only in first-person observations
      - Patches are NOT shown in ASCII
      - Patch size changes naturally with distance because they are drawn
        inside wall columns whose height already depends on ray distance
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

        self.blocking_symbols = {"#", "A", "B", "C", "L", "M"}
        self.traversable_symbols = {".", "P"}

        self._validate_grid()

        self.free_cells = self._find_free_cells()
        if len(self.free_cells) == 0:
            raise ValueError("No traversable cells found in the map.")

        self._validate_connected_traversable_space()

        # ----------------------------------------------------
        # Fixed wall patches in world coordinates.
        # These do not change with motion.
        #
        # Keys: (row, col) of wall cells
        # Values: patch type
        # ----------------------------------------------------
        self.wall_patches = self._build_wall_patches()

        self.pose: Optional[Pose] = None
        self.goal_pos: Optional[Tuple[int, int]] = None
        self.use_goal: bool = False
        self.last_action: Optional[str] = None
        self.steps: int = 0

    # ========================================================
    # Public API
    # ========================================================

    def reset(
        self,
        start_pose: Optional[Pose] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        min_goal_dist: int = 3,
        use_goal: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        if start_pose is None:
            start_row, start_col = self.rng.choice(self.free_cells)
            start_heading = self.rng.choice(HEADINGS)
            start_pose = Pose(start_row, start_col, start_heading)
        else:
            self._validate_pose(start_pose)

        self.pose = start_pose
        self.use_goal = use_goal

        if use_goal:
            if goal_pos is None:
                goal_pos = self._sample_goal_far_from(
                    start=(start_pose.row, start_pose.col),
                    min_goal_dist=min_goal_dist,
                )
            else:
                self._validate_goal(goal_pos)
                if not self.is_reachable((start_pose.row, start_pose.col), goal_pos):
                    raise ValueError(
                        f"Provided goal {goal_pos} is not reachable from start "
                        f"{(start_pose.row, start_pose.col)}"
                    )
            self.goal_pos = goal_pos
        else:
            self.goal_pos = None

        self.last_action = None
        self.steps = 0

        obs = self.render_first_person()
        info = self._build_info(collision=False, reached_goal=False)

        return obs, info

    def step(self, action: int | str) -> StepResult:
        if self.pose is None:
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

        elif action_name == "forward":
            nr, nc = self._next_cell(self.pose.row, self.pose.col, self.pose.heading)
            if not self._in_bounds(nr, nc) or self._is_blocking(nr, nc):
                collision = True
                reward += self.collision_penalty
            else:
                self.pose.row = nr
                self.pose.col = nc

        elif action_name == "backward":
            nr, nc = self._prev_cell(self.pose.row, self.pose.col, self.pose.heading)
            if not self._in_bounds(nr, nc) or self._is_blocking(nr, nc):
                collision = True
                reward += self.collision_penalty
            else:
                self.pose.row = nr
                self.pose.col = nc

        else:
            raise ValueError(f"Unsupported action: {action_name}")

        if self.use_goal and self.goal_pos is not None:
            if (self.pose.row, self.pose.col) == self.goal_pos:
                reached_goal = True
                reward += self.goal_reward

        done = reached_goal if self.use_goal else False

        obs = self.render_first_person()
        info = self._build_info(collision=collision, reached_goal=reached_goal)

        return StepResult(obs=obs, reward=reward, done=done, info=info)

    def render_first_person(self) -> np.ndarray:
        """
        Render a first-person grayscale observation.
        """
        if self.pose is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        H = self.obs_height
        W = self.obs_width

        img = np.zeros((H, W), dtype=np.float32)

        # sky / ceiling
        img[: H // 2, :] = 0.02
        # floor base
        img[H // 2 :, :] = 0.08

        base_angle = self._heading_to_angle_deg(self.pose.heading)
        half_fov = self.fov_deg / 2.0

        for x in range(W):
            ray_angle = base_angle - half_fov + (x / max(W - 1, 1)) * self.fov_deg
            hit = self._cast_ray(ray_angle)

            dist = max(hit["distance"], 1e-3)
            symbol = hit["symbol"]
            hit_row, hit_col = hit["cell"]

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

            if y1 < H:
                floor_vals = self._make_floor_column(
                    start_y=y1,
                    H=H,
                    ray_angle_deg=ray_angle,
                )
                img[y1:H, x] = floor_vals

        xx = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        vignette = 1.0 - 0.08 * (xx ** 2)
        img *= vignette[None, :]

        noise = self.np_rng.normal(loc=0.0, scale=0.004, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

        return img.astype(np.float32)

    def render_topdown_ascii(self, show_goal: bool = True) -> str:
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
                elif show_goal and self.use_goal and self.goal_pos is not None and (r, c) == self.goal_pos:
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

    def is_reachable(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> bool:
        if not self._in_bounds(*start) or not self._in_bounds(*goal):
            return False
        if self._is_blocking(*start) or self._is_blocking(*goal):
            return False
        if start == goal:
            return True

        visited: Set[Tuple[int, int]] = set()
        q = deque([start])
        visited.add(start)

        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in visited:
                    continue
                if self._is_blocking(nr, nc):
                    continue
                if (nr, nc) == goal:
                    return True
                visited.add((nr, nc))
                q.append((nr, nc))

        return False

    def shortest_path_length(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[int]:
        if not self._in_bounds(*start) or not self._in_bounds(*goal):
            return None
        if self._is_blocking(*start) or self._is_blocking(*goal):
            return None
        if start == goal:
            return 0

        visited: Set[Tuple[int, int]] = set()
        q = deque([(start[0], start[1], 0)])
        visited.add(start)

        while q:
            r, c, d = q.popleft()
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in visited:
                    continue
                if self._is_blocking(nr, nc):
                    continue
                if (nr, nc) == goal:
                    return d + 1
                visited.add((nr, nc))
                q.append((nr, nc, d + 1))

        return None

    # ========================================================
    # Internal map helpers
    # ========================================================

    def _default_map(self) -> List[str]:
        return [
            "A#######BC",
            "#...P....#",
            "#.###.##L#",
            "#.#......#",
            "#.#.###P.#",
            "#...#....#",
            "#P#.#.##.#",
            "#..M#....#",
            "#....C...#",
            "B#######A#",
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

    def _validate_connected_traversable_space(self) -> None:
        if not self.free_cells:
            raise ValueError("No traversable cells available.")

        start = self.free_cells[0]
        visited = set()
        q = deque([start])
        visited.add(start)

        while q:
            r, c = q.popleft()
            for nr, nc in self._neighbors4(r, c):
                if (nr, nc) in visited:
                    continue
                if self._is_blocking(nr, nc):
                    continue
                visited.add((nr, nc))
                q.append((nr, nc))

        free_set = set(self.free_cells)
        if visited != free_set:
            missing = sorted(list(free_set - visited))
            raise ValueError(
                "Traversable space is not fully connected. "
                f"Unreachable free cells found: {missing[:10]}"
            )

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

    def _neighbors4(self, r: int, c: int) -> List[Tuple[int, int]]:
        nbrs = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self._in_bounds(nr, nc):
                nbrs.append((nr, nc))
        return nbrs

    # ========================================================
    # Fixed wall patch helpers
    # ========================================================

    def _build_wall_patches(self) -> Dict[Tuple[int, int], str]:
        """
        Fixed patches attached to wall cells.

        These are chosen near ambiguous corridors/corners.
        They stay fixed in the world and therefore move naturally
        in the image as the agent moves.

        Types:
          - "corner"
          - "center"
          - "stripe"
        """
        patch_dict: Dict[Tuple[int, int], str] = {
            # upper horizontal corridor / left-right ambiguity
            (0, 8): "corner",
            (2, 8): "stripe",

            # left-middle region near row-5 ambiguity
            (4, 2): "corner",
            (5, 4): "center",
            (6, 2): "stripe",

            # right-middle corridor
            (4, 6): "corner",
            (6, 6): "stripe",

            # lower-right / lower corridor cues
            (8, 5): "center",
            (9, 7): "corner",

            # lower-left cues
            (9, 1): "corner",
            (7, 4): "center",
        }

        # keep only valid blocking cells
        clean_patch_dict: Dict[Tuple[int, int], str] = {}
        for (r, c), patch_type in patch_dict.items():
            if self._in_bounds(r, c) and self._is_blocking(r, c):
                clean_patch_dict[(r, c)] = patch_type

        return clean_patch_dict

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

    def _next_cell(self, row: int, col: int, heading: str) -> Tuple[int, int]:
        dr, dc = self._heading_to_delta(heading)
        return row + dr, col + dc

    def _prev_cell(self, row: int, col: int, heading: str) -> Tuple[int, int]:
        dr, dc = self._heading_to_delta(heading)
        return row - dr, col - dc

    def _heading_to_angle_deg(self, heading: str) -> float:
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
        return {
            "row": self.pose.row,
            "col": self.pose.col,
            "heading": self.pose.heading,
            "heading_idx": HEADING_TO_IDX[self.pose.heading],
            "goal_row": self.goal_pos[0] if self.goal_pos is not None else None,
            "goal_col": self.goal_pos[1] if self.goal_pos is not None else None,
            "use_goal": self.use_goal,
            "collision": collision,
            "reached_goal": reached_goal,
            "steps": self.steps,
            "last_action": self.last_action,
        }

    # ========================================================
    # Ray casting and rendering helpers
    # ========================================================

    def _cast_ray(self, ray_angle_deg: float) -> Dict:
        assert self.pose is not None

        angle_rad = math.radians(ray_angle_deg)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

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
        mapping = {
            "#": 0.85,
            "A": 0.95,
            "B": 0.72,
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
        col = np.ones((height,), dtype=np.float32) * self._symbol_base_intensity(symbol)

        attenuation = 1.0 / (1.0 + 0.12 * dist * dist)
        col *= attenuation * 1.8
        col = np.clip(col, 0.0, 1.0)

        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)
        col *= (0.95 - 0.15 * yy)

        if symbol == "#":
            grain = 0.015 * np.sin(np.linspace(0, 10, height, dtype=np.float32))
            col += grain
        elif symbol == "A":
            stripes = 0.08 * (np.sin(np.linspace(0, 22, height, dtype=np.float32)) > 0).astype(np.float32)
            col += stripes
        elif symbol == "B":
            bands = 0.05 * np.sin(np.linspace(0, 16, height, dtype=np.float32))
            col += bands
        elif symbol == "C":
            highlight = np.zeros((height,), dtype=np.float32)
            highlight[::5] = 0.06
            col += highlight
        elif symbol == "L":
            pulses = 0.12 * (np.sin(np.linspace(0, 28, height, dtype=np.float32)) > 0.4).astype(np.float32)
            col += pulses
        elif symbol == "M":
            alt = np.zeros((height,), dtype=np.float32)
            alt[::6] = 0.10
            alt[3::6] = -0.06
            col += alt

        cell_mod = 0.015 * (((hit_row + hit_col) % 3) - 1)
        col += cell_mod

        angle_mod = 0.01 * math.sin(math.radians(ray_angle_deg * 3.0))
        col += angle_mod

        # --------------------------------------------------
        # FIXED WALL PATCHES (PHYSICALLY CONSISTENT)
        # --------------------------------------------------
        patch_type = self.wall_patches.get((hit_row, hit_col), None)

        if patch_type is not None:
            h = len(col)

            # closer wall => taller column => naturally bigger patch
            patch_size = max(2, int(h * 0.15))
            center = h // 2

            if patch_type == "corner":
                # small bright square near top part
                col[:patch_size] += 0.25

            elif patch_type == "center":
                y0 = max(0, center - patch_size // 2)
                y1 = min(h, center + patch_size // 2)
                col[y0:y1] += 0.25

            elif patch_type == "stripe":
                for i in range(0, h, 6):
                    col[i:i + 2] += 0.15

        return np.clip(col, 0.0, 1.0).astype(np.float32)

    def _make_floor_column(
        self,
        start_y: int,
        H: int,
        ray_angle_deg: float,
    ) -> np.ndarray:
        assert self.pose is not None

        floor = np.zeros((H - start_y,), dtype=np.float32)

        angle_rad = math.radians(ray_angle_deg)
        dx = math.cos(angle_rad)
        dy = math.sin(angle_rad)

        r0 = self.pose.row + 0.5
        c0 = self.pose.col + 0.5

        for idx, y in enumerate(range(start_y, H)):
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


if __name__ == "__main__":
    env = TinyIndoorEnv(seed=42)

    obs, info = env.reset(use_goal=False)
    print(env.render_topdown_ascii(show_goal=True))
    print("info:", info)
    print("obs stats:", obs.shape, obs.min(), obs.max())

    demo_actions = ["forward", "backward", "turn_left", "forward", "turn_right"]

    for i, action in enumerate(demo_actions, start=1):
        result = env.step(action)
        print(f"\nStep {i} | action={action}")
        print(env.render_topdown_ascii(show_goal=True))
        print("info:", result.info)
        print("reward:", result.reward, "done:", result.done)
        print("obs stats:", result.obs.shape, result.obs.min(), result.obs.max())
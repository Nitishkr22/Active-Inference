## simulator.py ##
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class StepResult:
    observation: np.ndarray   # [64, 64] grayscale image in [0, 1]
    reward: float
    done: bool
    info: dict


class TinyIndoorEnv:
    """
    Tiny indoor navigation simulator.

    World:
      - 10x10 occupancy grid
      - 0 = free cell
      - 1 = wall

    Robot:
      - discrete pose: (row, col, heading)
      - heading in {0,1,2,3} = {N, E, S, W}

    Actions:
      0 = forward
      1 = turn_left
      2 = turn_right
      3 = stay

    Observation:
      - 64x64 grayscale first-person synthetic image
      - IMPORTANT:
        Observation is goal-independent.
        The goal is NOT rendered into the image.
    """

    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

    ACTION_FORWARD = 0
    ACTION_LEFT = 1
    ACTION_RIGHT = 2
    ACTION_STAY = 3

    HEADING_NAMES = ["N", "E", "S", "W"]
    ACTION_NAMES = ["forward", "turn_left", "turn_right", "stay"]

    def __init__(
        self,
        height: int = 10,
        width: int = 10,
        obs_size: int = 64,
        max_steps: int = 30,
        start_pos: Tuple[int, int] = (1, 1),
        start_heading: int = EAST,
        goal_pos: Tuple[int, int] = (8, 8),
    ):
        self.height = height
        self.width = width
        self.obs_size = obs_size
        self.max_steps = max_steps

        self.grid = self._build_map()

        self.start_pos = start_pos
        self.start_heading = start_heading
        self.goal_pos = goal_pos

        # Ensure start and goal are valid free cells
        if not self._is_free_static(self.start_pos):
            raise ValueError(f"start_pos {self.start_pos} is not a free cell")
        if not self._is_free_static(self.goal_pos):
            raise ValueError(f"goal_pos {self.goal_pos} is not a free cell")

        self.pos = self.start_pos
        self.heading = self.start_heading
        self.step_count = 0

        # Fixed local visual landmarks
        self.landmark_cells = {
            (3, 5): 0.85,
            (6, 5): 0.65,
            (8, 7): 0.95,
            (2, 3): 0.75,
        }

    # ------------------------------------------------------------------
    # Map construction
    # ------------------------------------------------------------------
    def _build_map(self) -> np.ndarray:
        """
        Build a simple hand-designed indoor map.
        """
        grid = np.ones((self.height, self.width), dtype=np.uint8)

        free_cells = [
            # main corridor
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
            (7, 5), (8, 5),

            # right branch
            (8, 6), (8, 7), (8, 8),

            # side corridor
            (3, 4), (3, 3), (3, 2),

            # small upper branch
            (2, 3), (2, 4),

            # lower side branch
            (6, 4), (6, 3),
        ]

        for r, c in free_cells:
            grid[r, c] = 0

        return grid

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------
    def reset(
        self,
        start_pos: Optional[Tuple[int, int]] = None,
        start_heading: Optional[int] = None,
        goal_pos: Optional[Tuple[int, int]] = None,
        random_start: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> StepResult:
        """
        Reset environment.

        Options:
        - start_pos / start_heading / goal_pos: explicit override
        - random_start=True: sample a random free start cell and heading
          (goal remains current goal unless goal_pos is also provided)
        """
        if rng is None:
            rng = np.random.default_rng()

        if goal_pos is not None:
            if not self._is_free_static(goal_pos):
                raise ValueError(f"goal_pos {goal_pos} is not a free cell")
            self.goal_pos = goal_pos

        if random_start:
            free_cells = self.get_free_cells()
            valid_starts = [p for p in free_cells if p != self.goal_pos]
            self.pos = valid_starts[rng.integers(len(valid_starts))]
            self.heading = int(rng.integers(0, 4))
        else:
            self.pos = start_pos if start_pos is not None else self.start_pos
            self.heading = start_heading if start_heading is not None else self.start_heading

        if not self._is_free(self.pos):
            raise ValueError(f"reset start position {self.pos} is not a free cell")

        self.step_count = 0

        obs = self._render_first_person()
        return StepResult(
            observation=obs,
            reward=0.0,
            done=False,
            info=self._get_info(),
        )

    def step(self, action: int) -> StepResult:
        if action not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid action {action}")

        self.step_count += 1

        if action == self.ACTION_LEFT:
            self.heading = (self.heading - 1) % 4

        elif action == self.ACTION_RIGHT:
            self.heading = (self.heading + 1) % 4

        elif action == self.ACTION_FORWARD:
            next_pos = self._forward_position(self.pos, self.heading)
            if self._is_free(next_pos):
                self.pos = next_pos

        elif action == self.ACTION_STAY:
            pass

        done = (self.pos == self.goal_pos) or (self.step_count >= self.max_steps)
        reward = 1.0 if self.pos == self.goal_pos else 0.0

        obs = self._render_first_person()
        info = self._get_info()
        info["action_name"] = self.ACTION_NAMES[action]

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _forward_position(self, pos: Tuple[int, int], heading: int) -> Tuple[int, int]:
        r, c = pos
        if heading == self.NORTH:
            return (r - 1, c)
        if heading == self.EAST:
            return (r, c + 1)
        if heading == self.SOUTH:
            return (r + 1, c)
        if heading == self.WEST:
            return (r, c - 1)
        raise ValueError("Invalid heading")

    def _is_free_static(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return False
        return self.grid[r, c] == 0

    def _is_free(self, pos: Tuple[int, int]) -> bool:
        return self._is_free_static(pos)

    def _ray_distance(
        self,
        angle_offset: float,
        max_range: int = 8,
        step_size: float = 0.05,
    ) -> float:
        """
        Very simple ray-cast in grid coordinates.
        Returns distance to wall or max_range.
        """
        r, c = self.pos

        y = r + 0.5
        x = c + 0.5

        heading_angle = {
            self.NORTH: -np.pi / 2,
            self.EAST: 0.0,
            self.SOUTH: np.pi / 2,
            self.WEST: np.pi,
        }[self.heading]

        theta = heading_angle + angle_offset
        dy = np.sin(theta)
        dx = np.cos(theta)

        dist = 0.0
        while dist < max_range:
            yy = y + dist * dy
            xx = x + dist * dx

            rr = int(yy)
            cc = int(xx)

            if rr < 0 or rr >= self.height or cc < 0 or cc >= self.width:
                return dist
            if self.grid[rr, cc] == 1:
                return dist

            dist += step_size

        return float(max_range)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_first_person(self) -> np.ndarray:
        """
        Goal-independent first-person observation.

        Includes:
        - wall depth structure
        - floor/ceiling gradient
        - asymmetric shading
        - local landmark patches
        - small heading cue

        IMPORTANT:
        The goal is NOT rendered in the image.
        """
        H = W = self.obs_size
        img = np.zeros((H, W), dtype=np.float32)

        # --------------------------------------------------------------
        # 1) Background + floor/ceiling structure
        # --------------------------------------------------------------
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        img[:] = 0.05
        floor_mask = yy > H // 2
        img[floor_mask] = 0.12 + 0.08 * ((yy[floor_mask] - H // 2) / (H / 2))

        # mild horizontal asymmetry
        img += 0.03 * ((xx / (W - 1)) - 0.5)

        # --------------------------------------------------------------
        # 2) Ray-cast walls
        # --------------------------------------------------------------
        num_rays = W
        fov = np.deg2rad(70.0)

        for i in range(num_rays):
            alpha = (i / max(1, num_rays - 1) - 0.5) * fov
            d = self._ray_distance(alpha)

            wall_height = int(np.clip((1.0 / max(d, 0.1)) * 45, 2, H))
            center = H // 2
            top = max(0, center - wall_height // 2)
            bottom = min(H, center + wall_height // 2)

            base_intensity = np.clip(1.2 / max(d, 0.2), 0.2, 1.0)
            lr_bias = 0.12 * ((i / max(1, W - 1)) - 0.5)

            wall_strip = np.linspace(-0.08, 0.08, bottom - top, dtype=np.float32)
            wall_values = np.clip(base_intensity + lr_bias + wall_strip, 0.0, 1.0)

            img[top:bottom, i] = np.maximum(img[top:bottom, i], wall_values)

        # --------------------------------------------------------------
        # 3) Landmark patch if front cell has one
        # --------------------------------------------------------------
        front_cell = self._forward_position(self.pos, self.heading)
        if front_cell in self.landmark_cells and self._is_free(front_cell):
            patch_intensity = self.landmark_cells[front_cell]

            rr0, rr1 = H // 2 - 10, H // 2 - 2
            cc0, cc1 = W // 2 - 6, W // 2 + 6
            img[rr0:rr1, cc0:cc1] = np.maximum(img[rr0:rr1, cc0:cc1], patch_intensity)

            if self.heading in [self.EAST, self.SOUTH]:
                img[rr0:rr1, cc0:cc0 + 3] *= 0.85
            else:
                img[rr0:rr1, cc1 - 3:cc1] *= 0.85

        # --------------------------------------------------------------
        # 4) Small deterministic heading cue
        # --------------------------------------------------------------
        if self.heading == self.NORTH:
            img[2:6, 2:6] = 0.9
        elif self.heading == self.EAST:
            img[2:6, W - 6:W - 2] = 0.9
        elif self.heading == self.SOUTH:
            img[H - 6:H - 2, W - 6:W - 2] = 0.9
        elif self.heading == self.WEST:
            img[H - 6:H - 2, 2:6] = 0.9

        return np.clip(img, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------
    def render_topdown_ascii(self):
        """
        Debug visualization in terminal.
        Goal is shown only in top-down debug view, not in observation.
        """
        chars = []
        heading_symbol = {
            self.NORTH: "^",
            self.EAST: ">",
            self.SOUTH: "v",
            self.WEST: "<",
        }[self.heading]

        for r in range(self.height):
            row = []
            for c in range(self.width):
                if (r, c) == self.pos:
                    row.append(heading_symbol)
                elif (r, c) == self.goal_pos:
                    row.append("G")
                elif self.grid[r, c] == 1:
                    row.append("#")
                else:
                    row.append(".")
            chars.append(" ".join(row))
        print("\n".join(chars))

    def render(self):
        self.render_topdown_ascii()

    def _get_info(self):
        return {
            "pos": self.pos,
            "heading": self.heading,
            "heading_name": self.HEADING_NAMES[self.heading],
            "goal_pos": self.goal_pos,
        }

    def set_state(self, pos: Tuple[int, int], heading: int):
        if not self._is_free(pos):
            raise ValueError(f"Cannot set state to non-free cell {pos}")
        if heading not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid heading {heading}")
        self.pos = pos
        self.heading = heading

    def set_goal(self, goal_pos: Tuple[int, int]):
        if not self._is_free_static(goal_pos):
            raise ValueError(f"goal_pos {goal_pos} is not a free cell")
        self.goal_pos = goal_pos

    def get_free_cells(self) -> List[Tuple[int, int]]:
        free_cells = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == 0:
                    free_cells.append((r, c))
        return free_cells
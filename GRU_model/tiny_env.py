# import numpy as np
# from dataclasses import dataclass


# @dataclass
# class StepResult:
#     observation: np.ndarray   # 64x64 grayscale image in [0, 1]
#     reward: float
#     done: bool
#     info: dict


# class TinyIndoorEnv:
#     """
#     A tiny indoor navigation simulator.

#     World:
#       - 10x10 occupancy grid
#       - 0 = free cell
#       - 1 = wall
#       - one goal cell

#     Robot:
#       - discrete pose: (row, col, heading)
#       - heading in {0,1,2,3} = {N, E, S, W}

#     Actions:
#       0 = forward
#       1 = turn_left
#       2 = turn_right
#       3 = stay

#     Observation:
#       - 64x64 grayscale first-person synthetic image
#     """

#     NORTH = 0
#     EAST = 1
#     SOUTH = 2
#     WEST = 3

#     ACTION_FORWARD = 0
#     ACTION_LEFT = 1
#     ACTION_RIGHT = 2
#     ACTION_STAY = 3

#     HEADING_NAMES = ["N", "E", "S", "W"]
#     ACTION_NAMES = ["forward", "turn_left", "turn_right", "stay"]

#     def __init__(self, height: int = 10, width: int = 10, obs_size: int = 64, max_steps: int = 30):
#         self.height = height
#         self.width = width
#         self.obs_size = obs_size
#         self.max_steps = max_steps

#         self.grid = self._build_map()
#         self.goal_pos = (8, 8) 

#         # Make sure goal is on a free cell
#         self.grid[self.goal_pos] = 0

#         self.start_pos = (1, 1)
#         self.start_heading = self.EAST

#         self.pos = self.start_pos
#         self.heading = self.start_heading
#         self.step_count = 0

#     def _build_map(self) -> np.ndarray:
#         """
#         Build a simple hand-designed indoor map:
#         - outer walls
#         - one main corridor
#         - one side corridor
#         - one small room-ish opening
#         """
#         grid = np.ones((self.height, self.width), dtype=np.uint8)

#         # Carve free space
#         free_cells = [
#             # main corridor
#             (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
#             (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
#             (7, 5), (8, 5),

#             # right branch toward goal
#             (8, 6), (8, 7), (8, 8),

#             # side corridor
#             (3, 4), (3, 3), (3, 2),

#             # small upper branch
#             (2, 3), (2, 4),

#             # lower side branch
#             (6, 4), (6, 3)
#         ]

#         for r, c in free_cells:
#             grid[r, c] = 0

#         return grid

#     def reset(self) -> StepResult:
#         self.pos = self.start_pos
#         self.heading = self.start_heading
#         self.step_count = 0

#         obs = self._render_first_person()
#         return StepResult(
#             observation=obs,
#             reward=0.0,
#             done=False,
#             info={
#                 "pos": self.pos,
#                 "heading": self.heading,
#                 "heading_name": self.HEADING_NAMES[self.heading],
#                 "goal_pos": self.goal_pos,
#             },
#         )

#     def step(self, action: int) -> StepResult:
#         assert action in [0, 1, 2, 3], f"Invalid action {action}"
#         self.step_count += 1

#         if action == self.ACTION_LEFT:
#             self.heading = (self.heading - 1) % 4
#         elif action == self.ACTION_RIGHT:
#             self.heading = (self.heading + 1) % 4
#         elif action == self.ACTION_FORWARD:
#             next_pos = self._forward_position(self.pos, self.heading)
#             if self._is_free(next_pos):
#                 self.pos = next_pos
#         elif action == self.ACTION_STAY:
#             pass

#         done = self.pos == self.goal_pos or self.step_count >= self.max_steps
#         reward = 1.0 if self.pos == self.goal_pos else 0.0

#         obs = self._render_first_person()
#         return StepResult(
#             observation=obs,
#             reward=reward,
#             done=done,
#             info={
#                 "pos": self.pos,
#                 "heading": self.heading,
#                 "heading_name": self.HEADING_NAMES[self.heading],
#                 "goal_pos": self.goal_pos,
#                 "action_name": self.ACTION_NAMES[action],
#             },
#         )

#     def _forward_position(self, pos, heading):
#         r, c = pos
#         if heading == self.NORTH:
#             return (r - 1, c)
#         if heading == self.EAST:
#             return (r, c + 1)
#         if heading == self.SOUTH:
#             return (r + 1, c)
#         if heading == self.WEST:
#             return (r, c - 1)
#         raise ValueError("Invalid heading")

#     def _is_free(self, pos) -> bool:
#         r, c = pos
#         if r < 0 or r >= self.height or c < 0 or c >= self.width:
#             return False
#         return self.grid[r, c] == 0  # returns True if free, False if wall

#     def _ray_distance(self, angle_offset: float, max_range: int = 8, step_size: float = 0.05) -> float:
#         """
#         Very simple ray-cast in grid coordinates.
#         Returns distance to wall or max_range.
#         """
#         r, c = self.pos

#         # robot is at cell center
#         y = r + 0.5
#         x = c + 0.5

#         heading_angle = {
#             self.NORTH: -np.pi / 2,
#             self.EAST: 0.0,
#             self.SOUTH: np.pi / 2,
#             self.WEST: np.pi,
#         }[self.heading]

#         theta = heading_angle + angle_offset
#         dy = np.sin(theta)
#         dx = np.cos(theta)

#         dist = 0.0
#         while dist < max_range:
#             yy = y + dist * dy
#             xx = x + dist * dx

#             rr = int(yy)
#             cc = int(xx)

#             if rr < 0 or rr >= self.height or cc < 0 or cc >= self.width:
#                 return dist
#             if self.grid[rr, cc] == 1:
#                 return dist
#             dist += step_size

#         return float(max_range)

#     def _goal_visible(self) -> bool:
#         """
#         Crude visibility: goal is visible if it is in same row/col corridor segment
#         and in front of robot without walls in between.
#         """
#         r, c = self.pos
#         gr, gc = self.goal_pos

#         if self.heading == self.EAST and r == gr and gc > c:
#             for cc in range(c + 1, gc + 1):
#                 if self.grid[r, cc] == 1:
#                     return False
#             return True

#         if self.heading == self.WEST and r == gr and gc < c:
#             for cc in range(gc, c):
#                 if self.grid[r, cc] == 1:
#                     return False
#             return True

#         if self.heading == self.SOUTH and c == gc and gr > r:
#             for rr in range(r + 1, gr + 1):
#                 if self.grid[rr, c] == 1:
#                     return False
#             return True

#         if self.heading == self.NORTH and c == gc and gr < r:
#             for rr in range(gr, r):
#                 if self.grid[rr, c] == 1:
#                     return False
#             return True

#         return False

#     def _render_first_person(self) -> np.ndarray:
#         """
#         Render a synthetic first-person observation.

#         Image semantics:
#         - bright vertical bars = nearby walls
#         - darker background = free corridor
#         - bright square near image center = goal marker if visible
#         """
#         H = W = self.obs_size  ## image size in pixels
#         img = np.zeros((H, W), dtype=np.float32)

#         # Background/floor
#         img[:] = 0.15

#         # Cast multiple rays across the field of view
#         num_rays = W
#         fov = np.deg2rad(70.0)

#         for i in range(num_rays):
#             alpha = (i / max(1, num_rays - 1) - 0.5) * fov
#             d = self._ray_distance(alpha)

#             # Inverse depth -> wall slice height
#             wall_height = int(np.clip((1.0 / max(d, 0.1)) * 45, 2, H))
#             center = H // 2
#             top = max(0, center - wall_height // 2)
#             bottom = min(H, center + wall_height // 2)

#             intensity = np.clip(1.2 / max(d, 0.2), 0.2, 1.0) # closer walls are brighter (0-->black, 1-->white)
#             img[top:bottom, i] = np.maximum(img[top:bottom, i], intensity)
#             # print(f"Ray {i}, alpha={alpha:.2f}, distance={d:.2f}")
#         # Draw goal marker if visible
#         if self._goal_visible():
#             rr0, rr1 = H // 2 - 8, H // 2 + 8
#             cc0, cc1 = W // 2 - 8, W // 2 + 8
#             img[rr0:rr1, cc0:cc1] = 1.0

#         img = np.clip(img, 0.0, 1.0)

#         return img

#     def render_topdown_ascii(self):
#         """
#         Debug visualization in terminal.
#         """
#         chars = []
#         heading_symbol = {
#             self.NORTH: "^",
#             self.EAST: ">",
#             self.SOUTH: "v",
#             self.WEST: "<",
#         }[self.heading]

#         for r in range(self.height):
#             row = []
#             for c in range(self.width):
#                 if (r, c) == self.pos:
#                     row.append(heading_symbol)
#                 elif (r, c) == self.goal_pos:
#                     row.append("G")
#                 elif self.grid[r, c] == 1:
#                     row.append("#")
#                 else:
#                     row.append(".")
#             chars.append(" ".join(row))
#         print("\n".join(chars))

## change rendere ##
## simulator.py ##

import numpy as np
from dataclasses import dataclass


@dataclass
class StepResult:
    observation: np.ndarray   # 64x64 grayscale image in [0, 1]
    reward: float
    done: bool
    info: dict


class TinyIndoorEnv:
    """
    A tiny indoor navigation simulator.

    World:
      - 10x10 occupancy grid
      - 0 = free cell
      - 1 = wall
      - one goal cell

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

    def __init__(self, height: int = 10, width: int = 10, obs_size: int = 64, max_steps: int = 30):
        self.height = height
        self.width = width
        self.obs_size = obs_size
        self.max_steps = max_steps

        self.grid = self._build_map()
        self.goal_pos = (8, 8)

        # Make sure goal is on a free cell
        self.grid[self.goal_pos] = 0

        self.start_pos = (1, 1)
        self.start_heading = self.EAST

        self.pos = self.start_pos
        self.heading = self.start_heading
        self.step_count = 0
        self.landmark_cells = {
                (3, 5): 0.85,
                (6, 5): 0.65,
                (8, 7): 0.95,
                (2, 3): 0.75,
            }

    def _build_map(self) -> np.ndarray:
        """
        Build a simple hand-designed indoor map:
        - outer walls
        - one main corridor
        - one side corridor
        - one small room-ish opening
        """
        grid = np.ones((self.height, self.width), dtype=np.uint8)

        # Carve free space
        free_cells = [
            # main corridor
            (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
            (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
            (7, 5), (8, 5),

            # right branch toward goal
            (8, 6), (8, 7), (8, 8),

            # side corridor
            (3, 4), (3, 3), (3, 2),

            # small upper branch
            (2, 3), (2, 4),

            # lower side branch
            (6, 4), (6, 3)
        ]

        for r, c in free_cells:
            grid[r, c] = 0

        return grid

    def reset(self) -> StepResult:
        self.pos = self.start_pos
        self.heading = self.start_heading
        self.step_count = 0

        obs = self._render_first_person()
        return StepResult(
            observation=obs,
            reward=0.0,
            done=False,
            info={
                "pos": self.pos,
                "heading": self.heading,
                "heading_name": self.HEADING_NAMES[self.heading],
                "goal_pos": self.goal_pos,
            },
        )

    def step(self, action: int) -> StepResult:
        assert action in [0, 1, 2, 3], f"Invalid action {action}"
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

        done = self.pos == self.goal_pos or self.step_count >= self.max_steps
        reward = 1.0 if self.pos == self.goal_pos else 0.0

        obs = self._render_first_person()
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "pos": self.pos,
                "heading": self.heading,
                "heading_name": self.HEADING_NAMES[self.heading],
                "goal_pos": self.goal_pos,
                "action_name": self.ACTION_NAMES[action],
            },
        )

    def _forward_position(self, pos, heading):
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

    def _is_free(self, pos) -> bool:
        r, c = pos
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return False
        return self.grid[r, c] == 0

    def _ray_distance(self, angle_offset: float, max_range: int = 8, step_size: float = 0.05) -> float:
        """
        Very simple ray-cast in grid coordinates.
        Returns distance to wall or max_range.
        """
        r, c = self.pos

        # robot is at cell center
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

    def _goal_visible(self) -> bool:
        """
        Crude visibility: goal is visible if it is in same row/col corridor segment
        and in front of robot without walls in between.
        """
        r, c = self.pos
        gr, gc = self.goal_pos

        if self.heading == self.EAST and r == gr and gc > c:
            for cc in range(c + 1, gc + 1):
                if self.grid[r, cc] == 1:
                    return False
            return True

        if self.heading == self.WEST and r == gr and gc < c:
            for cc in range(gc, c):
                if self.grid[r, cc] == 1:
                    return False
            return True

        if self.heading == self.SOUTH and c == gc and gr > r:
            for rr in range(r + 1, gr + 1):
                if self.grid[rr, c] == 1:
                    return False
            return True

        if self.heading == self.NORTH and c == gc and gr < r:
            for rr in range(gr, r):
                if self.grid[rr, c] == 1:
                    return False
            return True

        return False

    # def _render_first_person(self) -> np.ndarray:
    #     """
    #     Render a synthetic first-person observation.

    #     Image semantics:
    #     - bright vertical bars = nearby walls
    #     - darker background = free corridor
    #     - bright square near image center = goal marker if visible
    #     """
    #     H = W = self.obs_size
    #     img = np.zeros((H, W), dtype=np.float32)

    #     # Background/floor
    #     img[:] = 0.15

    #     # Cast multiple rays across the field of view
    #     num_rays = W
    #     fov = np.deg2rad(70.0)

    #     for i in range(num_rays):
    #         alpha = (i / max(1, num_rays - 1) - 0.5) * fov
    #         d = self._ray_distance(alpha)

    #         # Inverse depth -> wall slice height
    #         wall_height = int(np.clip((1.0 / max(d, 0.1)) * 45, 2, H))
    #         center = H // 2
    #         top = max(0, center - wall_height // 2)
    #         bottom = min(H, center + wall_height // 2)

    #         intensity = np.clip(1.2 / max(d, 0.2), 0.2, 1.0)
    #         img[top:bottom, i] = np.maximum(img[top:bottom, i], intensity)

    #     # Draw goal marker if visible
    #     if self._goal_visible():
    #         rr0, rr1 = H // 2 - 8, H // 2 + 8
    #         cc0, cc1 = W // 2 - 8, W // 2 + 8
    #         img[rr0:rr1, cc0:cc1] = 1.0

    #     return np.clip(img, 0.0, 1.0)
    def _render_first_person(self) -> np.ndarray:
        """
        Improved synthetic first-person observation.

        Adds:
        - asymmetric left/right wall brightness
        - floor gradient / horizon structure
        - landmark patches tied to visible front cell
        - goal marker if visible
        """
        H = W = self.obs_size
        img = np.zeros((H, W), dtype=np.float32)

        # ---------------------------
        # 1) Background + floor/ceiling structure
        # ---------------------------
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        # ceiling darker, floor slightly brighter
        img[:] = 0.05
        floor_mask = yy > H // 2
        img[floor_mask] = 0.12 + 0.08 * ((yy[floor_mask] - H // 2) / (H / 2))

        # mild horizontal gradient to break symmetry
        img += 0.03 * ((xx / (W - 1)) - 0.5)

        # ---------------------------
        # 2) Ray-cast walls
        # ---------------------------
        num_rays = W
        fov = np.deg2rad(70.0)

        for i in range(num_rays):
            alpha = (i / max(1, num_rays - 1) - 0.5) * fov
            d = self._ray_distance(alpha)

            wall_height = int(np.clip((1.0 / max(d, 0.1)) * 45, 2, H))
            center = H // 2
            top = max(0, center - wall_height // 2)
            bottom = min(H, center + wall_height // 2)

            # base wall brightness from depth
            base_intensity = np.clip(1.2 / max(d, 0.2), 0.2, 1.0)

            # left/right asymmetry: left rays darker, right rays brighter
            lr_bias = 0.12 * ((i / max(1, W - 1)) - 0.5)

            # vertical shading: upper part of wall slightly darker
            wall_strip = np.linspace(-0.08, 0.08, bottom - top, dtype=np.float32)
            wall_values = np.clip(base_intensity + lr_bias + wall_strip, 0.0, 1.0)

            img[top:bottom, i] = np.maximum(img[top:bottom, i], wall_values)

        # ---------------------------
        # 3) Add a landmark patch if front cell has one
        # ---------------------------
        # Define a few landmark cells once in the map.
        # You can move these later if you want.
        # landmark_cells = {
        #     (3, 5): 0.85,  # bright patch
        #     (6, 5): 0.65,  # medium patch
        #     (8, 7): 0.95,  # strong patch near goal corridor
        #     (2, 3): 0.75,
        # }
        landmark_cells = self.landmark_cells

        front_cell = self._forward_position(self.pos, self.heading)
        if front_cell in landmark_cells and self._is_free(front_cell):
            patch_intensity = landmark_cells[front_cell]

            # draw patch slightly above horizon, centered
            rr0, rr1 = H // 2 - 10, H // 2 - 2
            cc0, cc1 = W // 2 - 6, W // 2 + 6
            img[rr0:rr1, cc0:cc1] = np.maximum(img[rr0:rr1, cc0:cc1], patch_intensity)

            # give patch slight left/right asymmetry by heading
            if self.heading in [self.EAST, self.SOUTH]:
                img[rr0:rr1, cc0:cc0+3] *= 0.85
            else:
                img[rr0:rr1, cc1-3:cc1] *= 0.85

        # ---------------------------
        # 4) Goal marker if visible
        # ---------------------------
        if self._goal_visible():
            rr0, rr1 = H // 2 - 8, H // 2 + 8
            cc0, cc1 = W // 2 - 8, W // 2 + 8

            # strong square marker
            img[rr0:rr1, cc0:cc1] = 1.0

            # make it structured instead of plain block
            img[rr0+2:rr1-2, cc0+2:cc1-2] = 0.2
            img[rr0+5:rr1-5, cc0+5:cc1-5] = 1.0

        # ---------------------------
        # 5) Small deterministic heading cue
        # ---------------------------
        # Tiny marker in one image corner depending on heading.
        # This is intentionally small, not dominating the whole image.
        if self.heading == self.NORTH:
            img[2:6, 2:6] = 0.9
        elif self.heading == self.EAST:
            img[2:6, W-6:W-2] = 0.9
        elif self.heading == self.SOUTH:
            img[H-6:H-2, W-6:W-2] = 0.9
        elif self.heading == self.WEST:
            img[H-6:H-2, 2:6] = 0.9

        return np.clip(img, 0.0, 1.0)


    def render_topdown_ascii(self):
        """
        Debug visualization in terminal.
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

    def set_state(self, pos, heading):
        self.pos = pos
        self.heading = heading

    def get_free_cells(self):
        free_cells = []
        for r in range(self.height):
            for c in range(self.width):
                if self.grid[r, c] == 0:
                    free_cells.append((r, c))
        return free_cells
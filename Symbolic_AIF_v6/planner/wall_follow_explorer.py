"""
wall_follow_explorer.py — Systematic wall-following exploration.

Replaces random-waypoint exploration (Priority 2 during EXPLORE phase).
Cycles through 4 cardinal headings (E, N, W, S) and for each:
  1. SEEK  — move forward in that heading until wall contact or max_seek steps.
  2. FOLLOW — right-hand wall-follow after contact; exits on full circuit or time-out.
After all 4 headings: DONE → caller falls through to EFE info-gain.

Design:
  • Fully reactive (no map required) — works on a real robot with only odometry
    and collision detection.
  • 30° discrete turn steps matching Habitat action space.
  • Right-hand rule:
      - After wall contact:  turn LEFT 90° (wall now on our RIGHT).
      - While following: every PROBE_INTERVAL forward steps try TURN_R to check
        if the wall continues or a gap (doorway) has appeared.
      - On FORWARD blocked:  TURN_L to navigate an inner corner.
  • Gap detection is implicit: consecutive successful TURN_R + FORWARD steps
    indicate a passage (doorway) the agent travels through naturally.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import List, Optional, Set

import torch


class _Phase(str, Enum):
    ALIGN  = "ALIGN"    # turning to face seek heading
    SEEK   = "SEEK"     # moving forward to find a wall
    FOLLOW = "FOLLOW"   # following wall boundary (right-hand rule)
    DONE   = "DONE"     # all headings explored


# Cardinal seek headings in V6 frame (x=forward, y=left, θ ccw from +x)
# E = 0°, N = 90°, W = 180°, S = -90°
_HEADINGS = [0.0, math.pi / 2, math.pi, -math.pi / 2]

# How many consecutive forward steps before probing right (for gap/corner check)
_PROBE_INTERVAL = 4


class WallFollowExplorer:
    """
    Reactive wall-following exploration state machine.

    Call ``get_action(pose_v6, collided)`` once per step from Priority-2 in
    the main loop.  Returns an action int (0/1/2) while exploration is active,
    or ``None`` when DONE (falling through to the EFE planner).

    Parameters
    ----------
    turn_step   : radians per TURN_L / TURN_R action (must match Habitat config)
    max_seek    : max FORWARD steps in SEEK before giving up and trying next heading
    max_follow  : max steps in FOLLOW before stopping (safety limit)
    return_dist : distance to start-of-follow position that counts as "full circuit"
    """

    def __init__(
        self,
        turn_step:   float = math.radians(30),
        max_seek:    int   = 20,
        max_follow:  int   = 80,
        return_dist: float = 0.8,
    ):
        self._ts           = turn_step
        self._max_seek     = max_seek
        self._max_follow   = max_follow
        self._return_dist  = return_dist

        # State
        self._hi           = 0                  # which heading we are currently on
        self._phase        = _Phase.ALIGN
        self._seek_n       = 0                   # steps taken in SEEK
        self._follow_n     = 0                   # steps taken in FOLLOW
        self._init_pos:    Optional[torch.Tensor] = None  # position at FOLLOW start
        self._fwd_since_probe = 0               # fwd steps since last right-probe
        self._in_probe     = False               # True when we just issued TURN_R probe
        self._last_collided = False

        # Action queue (used for multi-step turn sequences)
        self._q: List[int] = []

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    @property
    def done(self) -> bool:
        return self._phase == _Phase.DONE

    @property
    def phase_name(self) -> str:
        """Human-readable phase + heading for logging."""
        if self._phase == _Phase.DONE:
            return "WF-DONE"
        hname = ["E", "N", "W", "S"][self._hi % 4]
        return f"WF-{self._phase.value}({hname})"

    @property
    def heading_idx(self) -> int:
        return self._hi

    def get_action(
        self,
        pose_v6:  torch.Tensor,  # [3] GT pose (x, y, θ)
        collided: bool,
    ) -> Optional[int]:
        """
        Return the next exploration action, or None to pass through to EFE.

        ``collided`` is the result of the *previous* step (True if the action
        taken last step resulted in a collision).  FORWARDs that collide are
        detected here; TURN_L/TURN_R never collide.
        """
        self._last_collided = collided

        # Drain pending queue first (alignment turns, initial 90° left)
        if self._q:
            return self._q.pop(0)

        if self._phase == _Phase.DONE:
            return None

        curr_h = float(pose_v6[2])

        # --- ALIGN: turn to face the current seek heading ---
        if self._phase == _Phase.ALIGN:
            acts = self._align_turns(curr_h, _HEADINGS[self._hi])
            if acts:
                self._q.extend(acts[1:])
                return acts[0]
            # Aligned — start seeking
            self._phase  = _Phase.SEEK
            self._seek_n = 0
            # fall-through to SEEK below

        # --- SEEK: move forward until wall or step limit ---
        if self._phase == _Phase.SEEK:
            if collided:
                return self._start_follow(pose_v6)

            if self._seek_n >= self._max_seek:
                return self._next_heading(pose_v6, collided)

            self._seek_n += 1
            return 0  # FORWARD

        # --- FOLLOW: right-hand wall-following ---
        if self._phase == _Phase.FOLLOW:
            self._follow_n += 1

            # Time limit
            if self._follow_n >= self._max_follow:
                return self._next_heading(pose_v6, collided)

            # Full-circuit check
            if self._follow_n > 15 and self._init_pos is not None:
                d = float((pose_v6[:2] - self._init_pos.to(pose_v6.device)).norm())
                if d < self._return_dist:
                    return self._next_heading(pose_v6, collided)

            return self._follow_action(collided)

        return None  # should not reach here

    # ------------------------------------------------------------------ #
    # Internal state transitions
    # ------------------------------------------------------------------ #

    def _start_follow(self, pose_v6: torch.Tensor) -> int:
        """Switch to FOLLOW after wall contact.  Queue 90° left turn."""
        self._phase          = _Phase.FOLLOW
        self._follow_n       = 0
        self._init_pos       = pose_v6[:2].detach().clone()
        self._fwd_since_probe = 0
        self._in_probe       = False
        # Turn 90° left (3 × TURN_L at 30°/step): wall now on our right
        self._q = [1, 1, 1]
        return self._q.pop(0)

    def _next_heading(self, pose_v6: torch.Tensor, collided: bool) -> Optional[int]:
        """Advance to the next cardinal heading, or transition to DONE."""
        self._hi += 1
        if self._hi >= len(_HEADINGS):
            self._phase = _Phase.DONE
            return None
        # Reset for next heading
        self._phase          = _Phase.ALIGN
        self._seek_n         = 0
        self._follow_n       = 0
        self._init_pos       = None
        self._fwd_since_probe = 0
        self._in_probe       = False
        self._q.clear()
        # Recurse to emit the first alignment turn immediately
        return self.get_action(pose_v6, False)

    def _follow_action(self, collided: bool) -> int:
        """
        One step of right-hand wall following.

        Wall is on our RIGHT (we turned left 90° when we first hit it).

        Rules:
          • Forward by default.
          • Every _PROBE_INTERVAL successful forward steps: issue TURN_R to
            check whether the wall has a gap (the probe step).
            - If the next FORWARD after the probe succeeds: we rounded an
              outer corner or found a gap → stay turned right, continue.
            - If the next FORWARD fails: undo (TURN_L) and resume straight.
          • If FORWARD is blocked (inner corner): TURN_L.
        """
        if self._in_probe:
            # We just issued TURN_R; now issue FORWARD to test the gap
            self._in_probe = False
            if collided:
                # FORWARD after TURN_R hit a wall → undo the TURN_R (back to original dir)
                self._fwd_since_probe = 0
                return 1  # TURN_L (undo)
            else:
                # Successfully moved after TURN_R → outer corner or gap, stay turned
                self._fwd_since_probe = 0
                return 0  # FORWARD

        if collided:
            # Blocked ahead → inner corner: turn left to go around it
            self._fwd_since_probe = 0
            return 1  # TURN_L

        # Normal forward movement
        self._fwd_since_probe += 1
        if self._fwd_since_probe >= _PROBE_INTERVAL:
            # Issue right-probe: TURN_R, then next call will try FORWARD
            self._fwd_since_probe = 0
            self._in_probe        = True
            return 2  # TURN_R

        return 0  # FORWARD

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def _align_turns(self, curr_h: float, tgt_h: float) -> List[int]:
        """
        Compute the minimal list of TURN_L (1) / TURN_R (2) actions to
        rotate from curr_h to tgt_h in multiples of turn_step.
        """
        diff = math.atan2(math.sin(tgt_h - curr_h), math.cos(tgt_h - curr_h))
        n    = round(diff / self._ts)
        if n > 0:
            return [1] * n    # TURN_L
        elif n < 0:
            return [2] * (-n) # TURN_R
        return []

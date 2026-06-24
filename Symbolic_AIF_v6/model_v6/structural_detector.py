"""
structural_detector.py — Detects walls and doorways from RGBD depth images.

Algorithm:
  1. Extract a mid-height horizontal band from the depth image, skipping the
     floor (bottom rows) and ceiling (top rows).
  2. For each pixel column compute depth quartiles (Q25, Q50, Q75).
     IQR = Q75 − Q25 measures depth spread: low IQR = flat planar surface.
  3. Classify columns as "wall" when median depth is in range AND IQR is small.
  4. Group contiguous wall columns into wall segments (min width in pixels).
  5. Find gaps between adjacent wall segments whose real-world width matches a
     standard door opening (0.5 – 2.5 m) and whose gap depth is significantly
     larger than the surrounding wall (i.e. you can see through).
  6. Convert each wall segment / gap to a world-frame Detection compatible with
     the existing SlotMemory pipeline.

Output class IDs (appended to COCO 91-class vocabulary):
  WALL_ID    = 92   — impassable planar obstacle
  DOORWAY_ID = 93   — navigable passage between rooms

No ML involved — works zero-shot in any RGBD environment.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Tuple

import numpy as np
import torch

from .belief_state import Detection
from .config import ModelV6Config, PerceptionConfig, StructuralConfig

# ── Structural category IDs ────────────────────────────────────────────────────
WALL_ID    = 92
DOORWAY_ID = 93


class StructuralDetector:
    """
    Detects walls and doorways from a single RGBD frame.

    No learnable parameters.  Returns List[Detection] using the same interface
    as Perception so results can be merged and fed to SlotMemory unchanged.

    Usage (called once per step inside WorldModelV6.forward_step):
        struct_dets = self.struct_det.detect(depth, cam_to_world)
        detections  = detections + [d.to(device) for d in struct_dets]
    """

    def __init__(self, cfg: ModelV6Config):
        self.scfg = cfg.structural
        self.pcfg = cfg.perception

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def detect(
        self,
        depth:        torch.Tensor,   # [1, H, W]  float32 metres
        cam_to_world: torch.Tensor,   # [4, 4]
    ) -> List[Detection]:
        sc   = self.scfg
        pcfg = self.pcfg

        depth_map = depth.squeeze(0).cpu().float()   # [H, W]
        H, W = depth_map.shape

        # ---- 1. Mid-height analysis band (avoids floor + ceiling) ----
        r0 = int(H * sc.band_frac_top)
        r1 = int(H * sc.band_frac_bot)
        if r1 <= r0:
            return []
        band = depth_map[r0:r1, :]    # [band_h, W]

        # Valid-pixel mask (within usable depth range)
        valid      = (band >= sc.wall_min_depth) & (band <= sc.wall_max_depth)
        valid_frac = valid.float().mean(dim=0).numpy()   # [W]  fraction of valid rows per column

        # NaN-fill invalid pixels for nanpercentile
        band_np = band.numpy().copy()
        band_np[~valid.numpy()] = np.nan

        # ---- 2. Per-column depth quartiles ----
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            col_q25 = np.nanpercentile(band_np, 25, axis=0)   # [W]
            col_q50 = np.nanpercentile(band_np, 50, axis=0)   # [W]
            col_q75 = np.nanpercentile(band_np, 75, axis=0)   # [W]
        col_iqr = col_q75 - col_q25                            # [W]

        # ---- 3. Smooth median depths (box filter, robust to single-column noise) ----
        k = sc.smooth_kernel
        kernel = np.ones(k) / k
        col_smooth = np.convolve(col_q50, kernel, mode="same")

        # ---- 4. Classify wall columns ----
        # Criteria: enough valid pixels, median depth in range, flat IQR.
        wall_col = (
            (valid_frac >= 0.30)
            & (~np.isnan(col_q50))
            & (col_iqr < sc.wall_iqr_max)
        )

        # ---- 5. Find contiguous wall segments ----
        segments = _find_segments(wall_col, sc.wall_min_cols)

        # ---- 6. Find doorway gaps between segments ----
        doorways = _find_doorways(segments, col_smooth, sc, pcfg.fx)

        # ---- 7. Convert to Detection objects ----
        dets: List[Detection] = []
        v_mid = float(r0 + r1) / 2.0   # representative pixel row for depth-lifting

        # Wall detections (one per segment, at segment centre)
        for u0, u1 in segments[:sc.max_wall_dets]:
            u_c = (u0 + u1) / 2.0
            d   = float(np.nanmedian(col_smooth[u0 : u1 + 1]))
            if np.isnan(d) or d <= 0:
                continue
            pos  = _lift_to_world(u_c, v_mid, d, pcfg, cam_to_world)
            conf = _wall_conf(col_iqr[u0 : u1 + 1], sc)
            dets.append(Detection(
                class_id   = WALL_ID,
                class_conf = conf,
                pos_world  = pos,
                bbox_img   = (float(u0), float(r0), float(u1), float(r1)),
            ))

        # Doorway detections (one per gap, at gap centre at wall-plane depth)
        for u0, u1, d_door in doorways[:sc.max_door_dets]:
            u_c = (u0 + u1) / 2.0
            pos = _lift_to_world(u_c, v_mid, d_door, pcfg, cam_to_world)
            width_m = (u1 - u0) * d_door / pcfg.fx
            conf    = _door_conf(width_m, sc)
            dets.append(Detection(
                class_id   = DOORWAY_ID,
                class_conf = conf,
                pos_world  = pos,
                bbox_img   = (float(u0), float(r0), float(u1), float(r1)),
            ))

        return dets


# ------------------------------------------------------------------ #
# Segment / gap finders
# ------------------------------------------------------------------ #

def _find_segments(
    wall_col: np.ndarray,   # [W] bool
    min_len:  int,
) -> List[Tuple[int, int]]:
    """
    Return (start_col, end_col) pairs for contiguous True runs of length >= min_len.
    Both indices are inclusive.
    """
    segments: List[Tuple[int, int]] = []
    W = len(wall_col)
    i = 0
    while i < W:
        if wall_col[i]:
            j = i
            while j < W and wall_col[j]:
                j += 1
            if j - i >= min_len:
                segments.append((i, j - 1))
            i = j
        else:
            i += 1
    return segments


def _find_doorways(
    segments:   List[Tuple[int, int]],
    col_smooth: np.ndarray,             # [W] smoothed median depth
    sc:         StructuralConfig,
    fx:         float,
) -> List[Tuple[int, int, float]]:
    """
    Scan gaps between adjacent wall segments for doorway candidates.

    A gap qualifies as a doorway when:
      - Its real-world width at the wall-plane depth is in [door_min, door_max].
      - The median depth inside the gap is at least door_depth_jump deeper than
        the surrounding wall (the robot can see through into the next room).

    Returns list of (gap_start, gap_end, door_depth_m) — indices inclusive,
    door_depth_m is the distance to the doorway plane (= average adjacent wall depth).
    """
    doorways: List[Tuple[int, int, float]] = []
    for k in range(len(segments) - 1):
        gap_start = segments[k][1] + 1
        gap_end   = segments[k + 1][0] - 1
        if gap_end < gap_start:
            continue

        # Distance to doorway plane ≈ average depth of the two flanking walls
        d_left  = float(np.nanmedian(col_smooth[segments[k][0]     : segments[k][1] + 1]))
        d_right = float(np.nanmedian(col_smooth[segments[k + 1][0] : segments[k + 1][1] + 1]))
        if np.isnan(d_left) or np.isnan(d_right):
            continue
        d_wall = (d_left + d_right) / 2.0

        # Depth inside the gap — must be substantially deeper to confirm it is open
        gap_vals = col_smooth[gap_start : gap_end + 1]
        with np.errstate(all="ignore"):
            d_gap = float(np.nanmedian(gap_vals))
        if np.isnan(d_gap):
            d_gap = d_wall + sc.door_depth_jump + 1.0   # treat all-NaN gap as open

        if d_gap < d_wall + sc.door_depth_jump:
            continue   # gap not clearly open — probably occluded, not a door

        # Real-world horizontal width at the wall-plane distance
        width_m = (gap_end - gap_start) * d_wall / fx
        if sc.door_min_width_m <= width_m <= sc.door_max_width_m:
            doorways.append((gap_start, gap_end, d_wall))

    return doorways


# ------------------------------------------------------------------ #
# Confidence helpers
# ------------------------------------------------------------------ #

def _wall_conf(iqr_segment: np.ndarray, sc: StructuralConfig) -> float:
    """Wall confidence: 1.0 when perfectly flat (IQR=0), 0.4 at iqr_max."""
    mean_iqr = float(np.nanmean(iqr_segment))
    conf = 1.0 - (mean_iqr / max(sc.wall_iqr_max, 1e-6))
    return float(np.clip(conf, 0.40, 1.0))


def _door_conf(width_m: float, sc: StructuralConfig) -> float:
    """Doorway confidence: peaks at 0.9 m (standard door), falls off toward limits."""
    ideal     = 0.90   # metres — standard interior door
    door_span = max(sc.door_max_width_m - sc.door_min_width_m, 1e-6)
    conf = 1.0 - abs(width_m - ideal) / door_span
    return float(np.clip(conf, 0.40, 1.0))


# ------------------------------------------------------------------ #
# Geometry: pixel + depth → world-frame 3D point
# ------------------------------------------------------------------ #

def _lift_to_world(
    u: float, v: float, d: float,
    pcfg:         PerceptionConfig,
    cam_to_world: torch.Tensor,     # [4, 4]
) -> torch.Tensor:
    """
    Back-project pixel (u, v) at depth d to a world-frame 3D point.

    Camera frame: x=right, y=down, z=forward (standard pinhole).
    cam_to_world rotates/translates to the V6 world frame.
    """
    x_cam = (u - pcfg.cx) * d / pcfg.fx
    y_cam = (v - pcfg.cy) * d / pcfg.fy
    z_cam = d
    pos_cam = torch.tensor([x_cam, y_cam, z_cam], dtype=torch.float32)
    p_h = torch.cat([pos_cam, torch.ones(1, dtype=torch.float32)], dim=0)  # [4]
    p_w = (cam_to_world.cpu().float() @ p_h)[:3]                           # [3]
    return p_w

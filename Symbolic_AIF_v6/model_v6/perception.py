"""
perception.py — RGBD observation → list of Detection objects in world frame.

Pipeline:
  1. Object detector  (Faster R-CNN or YOLOv8)
     RGB [3, H, W] → [(class_id, score, bbox)]
  2. Depth lifting
     For each detection bbox: sample depth at centroid → 3D point in camera frame
  3. Camera → World transform
     Apply cam_to_world 4×4 matrix → world-frame position [x, y, z]

The detector is frozen (no gradients). Swap backends via PerceptionConfig.detector_backbone.

Camera coordinate convention (standard pinhole, right-handed):
  x = right,  y = down,  z = forward (into scene)

World frame (Habitat nav plane):
  x = forward (−z_cam when camera faces −z in Habitat)
  y = left
  z = up

The cam_to_world matrix is provided by the caller (from Habitat pose sensor or
odometry + initial transform). It transforms points from camera frame to world frame.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from .belief_state import Detection
from .config import ModelV6Config, PerceptionConfig


class Perception(nn.Module):
    """
    Converts an RGBD frame + current cam_to_world pose into a list of Detections.

    No learnable parameters — the backbone is a frozen pre-trained detector.
    """

    def __init__(self, cfg: ModelV6Config):
        super().__init__()
        self.cfg   = cfg
        self.pcfg  = cfg.perception
        self._build_detector()

    def _build_detector(self) -> None:
        pcfg = self.pcfg
        if pcfg.detector_backbone == "fasterrcnn":
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model   = fasterrcnn_resnet50_fpn(weights=weights)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            self.detector = model
        else:
            raise ValueError(
                f"Unknown detector backbone: {pcfg.detector_backbone!r}. "
                "Supported: 'fasterrcnn'."
            )

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def forward(
        self,
        rgb:          torch.Tensor,        # [3, H, W]  float32  0–1
        depth:        torch.Tensor,        # [1, H, W]  float32  metres
        cam_to_world: torch.Tensor,        # [4, 4]
    ) -> List[Detection]:
        """
        Returns a list of Detection objects in world-frame coordinates.
        May return an empty list when nothing is detected or depth is invalid.
        """
        pcfg = self.pcfg

        # ---- 1. Object detection on RGB ----
        boxes, class_ids, scores = self._detect(rgb)

        if len(class_ids) == 0:
            return []

        # ---- 2. Filter by allowed classes ----
        if pcfg.allowed_coco_ids is not None:
            keep = [i for i, c in enumerate(class_ids)
                    if c in pcfg.allowed_coco_ids]
            boxes     = [boxes[i]     for i in keep]
            class_ids = [class_ids[i] for i in keep]
            scores    = [scores[i]    for i in keep]

        if len(class_ids) == 0:
            return []

        # ---- 3. Depth sampling ----
        dets: List[Detection] = []
        depth_map = depth.squeeze(0)  # [H, W]

        for i, (box, cls_id, score) in enumerate(zip(boxes, class_ids, scores)):
            x1, y1, x2, y2 = box
            # Centroid pixel
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0

            d = _sample_depth(depth_map, u, v, pcfg)
            if d is None:
                continue  # invalid depth — skip detection

            # ---- 4. Pixel + depth → camera-frame 3D point ----
            pos_cam = _pixel_to_camera(u, v, d, pcfg)

            # ---- 5. Camera frame → world frame ----
            pos_world = _cam_to_world_point(pos_cam, cam_to_world)

            dets.append(Detection(
                class_id=int(cls_id),
                class_conf=float(score),
                pos_world=pos_world,
                bbox_img=(float(x1), float(y1), float(x2), float(y2)),
            ))

        return dets[:pcfg.detector_max_det]

    # ------------------------------------------------------------------ #
    # Detector backend
    # ------------------------------------------------------------------ #

    def _detect(
        self, rgb: torch.Tensor  # [3, H, W]  float32  0–1
    ):
        """Returns (boxes_list, class_id_list, score_list)."""
        pcfg = self.pcfg
        # Faster R-CNN expects list of images
        result = self.detector([rgb])[0]

        boxes     = result["boxes"].cpu().tolist()   # [[x1,y1,x2,y2], ...]
        labels    = result["labels"].cpu().tolist()  # 1-indexed COCO
        scores    = result["scores"].cpu().tolist()

        # COCO labels are 1-indexed; convert to 0-indexed
        filtered = [
            (boxes[i], labels[i] - 1, scores[i])
            for i in range(len(labels))
            if scores[i] >= pcfg.detector_conf_threshold
        ]

        if not filtered:
            return [], [], []

        # Sort by score descending, cap at max_det
        filtered.sort(key=lambda x: -x[2])
        filtered = filtered[:pcfg.detector_max_det]

        boxes_out  = [f[0] for f in filtered]
        cls_out    = [f[1] for f in filtered]
        scores_out = [f[2] for f in filtered]
        return boxes_out, cls_out, scores_out


# ------------------------------------------------------------------ #
# Geometry utilities
# ------------------------------------------------------------------ #

def _sample_depth(
    depth_map: torch.Tensor,   # [H, W]  metres
    u: float, v: float,
    pcfg: PerceptionConfig,
    patch_radius: int = 3,
) -> Optional[float]:
    """
    Sample depth at (u, v) by taking the median in a small patch.
    Returns None if the depth is invalid (zero, out of range).
    """
    H, W = depth_map.shape
    u_i  = int(round(u))
    v_i  = int(round(v))
    r    = patch_radius

    u0, u1 = max(0, u_i - r), min(W, u_i + r + 1)
    v0, v1 = max(0, v_i - r), min(H, v_i + r + 1)

    patch = depth_map[v0:v1, u0:u1]
    valid = patch[(patch >= pcfg.depth_min) & (patch <= pcfg.depth_max)]

    if valid.numel() == 0:
        return None

    return float(valid.median().item())


def _pixel_to_camera(
    u: float, v: float, d: float,
    pcfg: PerceptionConfig,
) -> torch.Tensor:
    """
    Back-project pixel (u, v) with depth d to camera-frame 3D point.
    Camera: x right, y down, z forward.
    """
    x_cam = (u - pcfg.cx) * d / pcfg.fx
    y_cam = (v - pcfg.cy) * d / pcfg.fy
    z_cam = d
    return torch.tensor([x_cam, y_cam, z_cam], dtype=torch.float32)


def _cam_to_world_point(
    pos_cam: torch.Tensor,     # [3]
    cam_to_world: torch.Tensor,  # [4, 4]
) -> torch.Tensor:
    """Apply homogeneous transform."""
    p_h = torch.cat([pos_cam, torch.ones(1, dtype=pos_cam.dtype,
                                          device=pos_cam.device)], dim=0)  # [4]
    p_w = (cam_to_world.to(pos_cam.device) @ p_h)[:3]                     # [3]
    return p_w


# ------------------------------------------------------------------ #
# Camera intrinsic helper (for Habitat RGBD sensor)
# ------------------------------------------------------------------ #

def habitat_cam_to_world(
    agent_pos:   torch.Tensor,   # [3]  Habitat world (x, y=up, z=back)
    agent_quat:  torch.Tensor,   # [4]  (w, x, y, z)
) -> torch.Tensor:
    """
    Build a 4×4 cam_to_world matrix from Habitat agent pose.

    Habitat convention: agent looks along -Z, Y is up.
    We convert to the V6 world frame: X=forward, Y=left, Z=up.

    This is a placeholder implementation — the caller (habitat_adapter) will
    provide the exact transform based on the configured sensor rig.
    """
    from torchvision.ops import box_area  # noqa — just checking torchvision available
    # Build rotation from quaternion
    w, qx, qy, qz = agent_quat.unbind(0)
    R = torch.tensor([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - w*qz),       2*(qx*qz + w*qy)      ],
        [2*(qx*qy + w*qz),        1 - 2*(qx**2 + qz**2),  2*(qy*qz - w*qx)      ],
        [2*(qx*qz - w*qy),        2*(qy*qz + w*qx),        1 - 2*(qx**2 + qy**2)],
    ], dtype=torch.float32)

    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = R
    T[:3,  3] = agent_pos.float()
    return T

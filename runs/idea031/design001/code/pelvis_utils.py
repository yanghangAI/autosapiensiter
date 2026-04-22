"""Shared pelvis unprojection and absolute MPJPE utilities.

Used by both Pose3dRegressionHead and Pose3dTransformerHead.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def recover_pelvis_3d(
    pelvis_depth: torch.Tensor,
    pelvis_uv: torch.Tensor,
    K: np.ndarray,
    crop_h: int,
    crop_w: int,
) -> torch.Tensor:
    """Unproject pelvis (depth, uv) to absolute 3D via crop intrinsics.

    BEDLAM2 convention: X=forward (depth), Y=left, Z=up.
    Projection: u = fx*(-Y/X) + cx,  v = fy*(-Z/X) + cy
    Inverse:    Y = -(u_px - cx) * X / fx,  Z = -(v_px - cy) * X / fy

    Args:
        pelvis_depth: (B, 1) forward distance in metres.
        pelvis_uv: (B, 2) normalised [-1, 1] pelvis position.
        K: (3, 3) crop intrinsic matrix (numpy).
        crop_h: Crop height in pixels.
        crop_w: Crop width in pixels.

    Returns:
        (B, 3) absolute pelvis [X, Y, Z] in metres.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = pelvis_depth[:, 0]                              # (B,)
    u_px = (pelvis_uv[:, 0] + 1.0) / 2.0 * crop_w     # (B,)
    v_px = (pelvis_uv[:, 1] + 1.0) / 2.0 * crop_h     # (B,)
    Y = -(u_px - cx) * X / fx                           # (B,)
    Z = -(v_px - cy) * X / fy                           # (B,)

    return torch.stack([X, Y, Z], dim=-1)                # (B, 3)


def compute_mpjpe_abs(
    pred_joints: torch.Tensor,
    gt_joints: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    pred_uv: torch.Tensor,
    gt_uv: torch.Tensor,
    batch_data_samples: Sequence,
) -> torch.Tensor:
    """Compute absolute MPJPE (mm) using predicted pelvis.

    Reconstructs absolute joint positions for both pred and GT,
    then computes MPJPE.

    Args:
        pred_joints: (B, J, 3) predicted root-relative joints (metres).
        gt_joints: (B, J, 3) GT root-relative joints (metres).
        pred_depth: (B, 1) predicted pelvis depth.
        gt_depth: (B, 1) GT pelvis depth.
        pred_uv: (B, 2) predicted pelvis UV.
        gt_uv: (B, 2) GT pelvis UV.
        batch_data_samples: list of data samples with metainfo['K'].

    Returns:
        Scalar tensor: absolute MPJPE in mm.
    """
    B = pred_joints.size(0)

    # Collect per-sample K and crop dimensions
    abs_pred_list = []
    abs_gt_list = []
    for i in range(B):
        ds = batch_data_samples[i]
        K = np.asarray(ds.metainfo.get('K'), dtype=np.float32)
        img_shape = ds.metainfo.get('img_shape', (640, 384))
        crop_h, crop_w = int(img_shape[0]), int(img_shape[1])

        pred_pelvis = recover_pelvis_3d(
            pred_depth[i:i+1], pred_uv[i:i+1], K, crop_h, crop_w)  # (1, 3)
        gt_pelvis = recover_pelvis_3d(
            gt_depth[i:i+1], gt_uv[i:i+1], K, crop_h, crop_w)      # (1, 3)

        abs_pred_list.append(pred_joints[i] + pred_pelvis)   # (J, 3)
        abs_gt_list.append(gt_joints[i] + gt_pelvis)         # (J, 3)

    abs_pred = torch.stack(abs_pred_list)    # (B, J, 3)
    abs_gt = torch.stack(abs_gt_list)        # (B, J, 3)

    return (abs_pred - abs_gt).norm(dim=-1).mean() * 1000.0


def uv_to_grid_coords(uv_norm: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
    """Convert (u_norm, v_norm) in [-1, 1] to (row, col) feature-grid coordinates.

    Args:
        uv_norm: (..., 2) tensor, last dim is (u, v) in [-1, 1].
        feat_h: feature map height (e.g., 40).
        feat_w: feature map width  (e.g., 24).

    Returns:
        (..., 2) tensor, last dim is (row, col) in float grid units.
    """
    u_grid = (uv_norm[..., 0] + 1.0) * 0.5 * (feat_w - 1)   # col in [0, W-1]
    v_grid = (uv_norm[..., 1] + 1.0) * 0.5 * (feat_h - 1)   # row in [0, H-1]
    return torch.stack([v_grid, u_grid], dim=-1)            # (..., 2): (row, col)


def build_gaussian_heatmap_2d(
    center_hw: torch.Tensor,
    feat_h: int,
    feat_w: int,
    sigma: float,
) -> torch.Tensor:
    """Build an L1-normalized (sum=1) Gaussian target heatmap flattened to (B, feat_h*feat_w)."""
    device = center_hw.device
    dtype = center_hw.dtype
    h_idx = torch.arange(feat_h, device=device, dtype=dtype)
    w_idx = torch.arange(feat_w, device=device, dtype=dtype)
    grid_h, grid_w = torch.meshgrid(h_idx, w_idx, indexing='ij')   # (H, W)
    grid = torch.stack([grid_h, grid_w], dim=-1).view(-1, 2)       # (H*W, 2)
    mu = center_hw.unsqueeze(1)                                     # (B, 1, 2)
    g = grid.unsqueeze(0)                                           # (1, H*W, 2)
    dist2 = ((mu - g) ** 2).sum(dim=-1)                             # (B, H*W)
    hm = torch.exp(-dist2 / (2.0 * sigma ** 2))
    hm = hm / hm.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return hm                                                       # (B, H*W)

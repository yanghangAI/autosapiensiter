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


def unproject_grid_to_metric_3d(
    depth_grid: torch.Tensor,
    K_batch: torch.Tensor,
    crop_hw: torch.Tensor,
    feat_h: int,
    feat_w: int,
    d_min: float = 0.1,
    d_max: float = 50.0,
) -> torch.Tensor:
    """Unproject a feature-grid depth map to camera-frame metric 3D.

    BEDLAM2 convention (same as ``recover_pelvis_3d``):
        X = d  (forward distance in metres)
        Y = -(u_px - cx) * X / fx
        Z = -(v_px - cy) * X / fy

    Pixel centres on the crop:
        u_px = (w + 0.5) * crop_w / W'
        v_px = (h + 0.5) * crop_h / H'

    Args:
        depth_grid: (B, 1, H', W') depth in metres, already resized to feature
            grid.
        K_batch:    (B, 3, 3) per-sample crop intrinsics.
        crop_hw:    (B, 2) per-sample (crop_h, crop_w) in pixels.
        feat_h, feat_w: feature-map spatial dims.
        d_min, d_max: soft clamp bounds applied to depth before unprojection.

    Returns:
        (B, H'*W', 3) metric XYZ in metres on the same device/dtype as
        ``depth_grid``.
    """
    B = depth_grid.shape[0]
    device = depth_grid.device
    out_dtype = depth_grid.dtype

    w_idx = torch.arange(feat_w, dtype=torch.float32, device=device)
    h_idx = torch.arange(feat_h, dtype=torch.float32, device=device)
    grid_v, grid_u = torch.meshgrid(h_idx, w_idx, indexing='ij')  # (H', W')

    crop_h = crop_hw[:, 0].view(B, 1, 1).float()
    crop_w = crop_hw[:, 1].view(B, 1, 1).float()
    u_px = (grid_u.unsqueeze(0) + 0.5) * (crop_w / float(feat_w))
    v_px = (grid_v.unsqueeze(0) + 0.5) * (crop_h / float(feat_h))

    K = K_batch.float()
    fx = K[:, 0, 0].view(B, 1, 1)
    fy = K[:, 1, 1].view(B, 1, 1)
    cx = K[:, 0, 2].view(B, 1, 1)
    cy = K[:, 1, 2].view(B, 1, 1)

    d = depth_grid[:, 0].float()
    d = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    d = d.clamp(min=d_min, max=d_max)

    X = d
    Y = -(u_px - cx) * X / fx
    Z = -(v_px - cy) * X / fy

    P = torch.stack([X, Y, Z], dim=-1)              # (B, H', W', 3)
    P = P.reshape(B, feat_h * feat_w, 3)
    return P.to(out_dtype)


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

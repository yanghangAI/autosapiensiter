# Design 001 — Auxiliary 2D Reprojection Loss on Body Joints (lambda=0.5, minimal)

**Design Description:** Add a single auxiliary Smooth-L1 reprojection loss (weight=0.5) that projects the predicted absolute body joints (pred_pelvis + pred_root_relative for indices 0-21) through per-sample K to normalised pixel coordinates and supervises against the GT absolute-joint projections, coupling the joint and pelvis pathways through camera geometry. No other baseline change.

**Starting Point:** `baseline/`

---

## Overview

The baseline head regresses three quantities with independent losses: root-relative joints, pelvis depth, pelvis UV. There is no loss-level coupling between them. This design adds a single auxiliary term `L_reproj = smooth_l1(pred_body_joints_2d, gt_body_joints_2d)` with weight `lambda_reproj = 0.5`, where both 2D positions are produced by projecting absolute 3D joints (pelvis + root-relative offset) through the per-sample crop intrinsic matrix `K`.

The reprojection is applied ONLY to body joints (indices 0-21), matching the existing body-only supervision convention and the evaluation metric.

All architecture, optimizer, LR schedule, data pipeline, hooks, seed, batch size, accumulation, and evaluation settings are unchanged from the baseline.

---

## Files to Change

1. `pelvis_utils.py` — add new helper `project_joints_to_2d`.
2. `pose3d_transformer_head.py` — compute and add the reprojection loss term in `loss()`.
3. `config.py` — add `reproj_loss_weight=0.5` as a head kwarg.

No other files are changed.

---

## Algorithm Changes

### `pelvis_utils.py`

Add a new function at module level (below `recover_pelvis_3d`, above or below `compute_mpjpe_abs` — either is fine):

```python
def project_joints_to_2d(
    joints_abs: torch.Tensor,
    K: np.ndarray,
    crop_h: int,
    crop_w: int,
    x_min: float = 0.01,
) -> torch.Tensor:
    """Project absolute camera-frame joints to normalised [-1, 1] pixel coords.

    BEDLAM2 convention (same as recover_pelvis_3d):
        u_px = fx * (-Y / X) + cx
        v_px = fy * (-Z / X) + cy
    Then normalise:
        u_norm = 2 * u_px / crop_w - 1
        v_norm = 2 * v_px / crop_h - 1

    Args:
        joints_abs: (B, J, 3) absolute 3D joints [X, Y, Z] in metres.
        K: (3, 3) crop intrinsic matrix (numpy.ndarray).
        crop_h: Crop height in pixels.
        crop_w: Crop width in pixels.
        x_min: Clamp for forward distance X to avoid divide-by-zero (default 0.01).

    Returns:
        (B, J, 2) normalised pixel coordinates in [-1, 1] convention.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = joints_abs[..., 0].clamp(min=x_min)   # (B, J)
    Y = joints_abs[..., 1]                     # (B, J)
    Z = joints_abs[..., 2]                     # (B, J)

    u_px = fx * (-Y / X) + cx                  # (B, J)
    v_px = fy * (-Z / X) + cy                  # (B, J)

    u_norm = 2.0 * u_px / float(crop_w) - 1.0
    v_norm = 2.0 * v_px / float(crop_h) - 1.0

    return torch.stack([u_norm, v_norm], dim=-1)  # (B, J, 2)
```

Constraints:
- Fully differentiable (pure torch ops on inputs; K/crop dims are python floats/ints).
- Clamp `X >= x_min=0.01` BEFORE the division to avoid NaN/Inf gradients (identical to the convention in `SubtractRootJoint` / `recover_pelvis_3d`).
- Input `K` is numpy (matching how it is read from `ds.metainfo['K']`), the intrinsics are extracted as python floats, so no device-placement headaches.
- Works for any leading batch shape (uses `...` indexing), but the head calls it with `(1, J, 3)` per sample in a Python loop (see below).

No changes to `recover_pelvis_3d` or `compute_mpjpe_abs`.

### `pose3d_transformer_head.py`

#### 1. Imports

Add `recover_pelvis_3d` and `project_joints_to_2d` to the existing pelvis_utils import. Replace:
```python
from pelvis_utils import compute_mpjpe_abs as _compute_mpjpe_abs
```
with:
```python
from pelvis_utils import (
    compute_mpjpe_abs as _compute_mpjpe_abs,
    recover_pelvis_3d as _recover_pelvis_3d,
    project_joints_to_2d as _project_joints_to_2d,
)
```
Add `import numpy as np` if not already present (baseline file currently does not import numpy — add the line `import numpy as np` near the other top-level imports, right after `import torch`).

#### 2. `Pose3dTransformerHead.__init__` — new parameter

Add `reproj_loss_weight: float = 0.0` to the `__init__` signature (after `loss_weight_uv`, before `init_cfg`):

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    loss_joints: ConfigType = dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth: ConfigType = dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv: ConfigType = dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth: float = 1.0,
    loss_weight_uv: float = 1.0,
    reproj_loss_weight: float = 0.0,
    init_cfg: OptConfigType = None,
):
    ...
    self.reproj_loss_weight = reproj_loss_weight
```

Store as `self.reproj_loss_weight = reproj_loss_weight`. Default is 0.0 so the head is backward-compatible with any config that does not set it.

#### 3. `loss()` — add reprojection term

Inside `Pose3dTransformerHead.loss`, AFTER the existing `losses['loss/uv/train'] = ...` line and BEFORE the `with torch.no_grad():` block (i.e., at the same indentation as the other loss assignments), add:

```python
# ── Auxiliary 2D reprojection loss (body joints only) ────────────────
if self.reproj_loss_weight > 0.0:
    _BODY_J = list(range(0, 22))  # body joints; same set as joint-loss
    B = pred['joints'].size(0)
    pred_2d_list = []
    gt_2d_list = []
    for i in range(B):
        ds = batch_data_samples[i]
        K = np.asarray(ds.metainfo.get('K'), dtype=np.float32)
        img_shape = ds.metainfo.get('img_shape', (640, 384))
        crop_h_i, crop_w_i = int(img_shape[0]), int(img_shape[1])

        # Absolute pelvis (differentiable wrt pred_depth, pred_uv)
        pred_pelvis_i = _recover_pelvis_3d(
            pred['pelvis_depth'][i:i+1],
            pred['pelvis_uv'][i:i+1],
            K, crop_h_i, crop_w_i,
        )  # (1, 3)
        gt_pelvis_i = _recover_pelvis_3d(
            gt_depth[i:i+1], gt_uv[i:i+1], K, crop_h_i, crop_w_i,
        )  # (1, 3)

        # Absolute body joints (broadcast pelvis over J)
        pred_body_abs_i = (
            pred['joints'][i:i+1, _BODY_J] + pred_pelvis_i.unsqueeze(1)
        )  # (1, 22, 3)
        gt_body_abs_i = (
            gt_joints[i:i+1, _BODY_J] + gt_pelvis_i.unsqueeze(1)
        )  # (1, 22, 3)

        pred_2d_i = _project_joints_to_2d(
            pred_body_abs_i, K, crop_h_i, crop_w_i)  # (1, 22, 2)
        gt_2d_i = _project_joints_to_2d(
            gt_body_abs_i, K, crop_h_i, crop_w_i)    # (1, 22, 2)

        pred_2d_list.append(pred_2d_i)
        gt_2d_list.append(gt_2d_i)

    pred_2d = torch.cat(pred_2d_list, dim=0)  # (B, 22, 2)
    gt_2d = torch.cat(gt_2d_list, dim=0)      # (B, 22, 2)

    # Smooth-L1 (beta=0.05 matches the other losses' beta); reduction=mean.
    reproj_raw = torch.nn.functional.smooth_l1_loss(
        pred_2d, gt_2d, beta=0.05, reduction='mean')
    losses['loss/reproj/train'] = self.reproj_loss_weight * reproj_raw
```

Key constraints:
- The per-sample Python loop mirrors `compute_mpjpe_abs` — do not refactor/vectorise; stay close to the existing style.
- `gt_depth` (shape `(B, 1)`) and `gt_uv` (shape `(B, 2)`) are already available in `loss()` by this point from the earlier assembly; reuse them.
- `pred['pelvis_depth']` is `(B, 1)`, `pred['pelvis_uv']` is `(B, 2)` — both are already in the `pred` dict from `forward()`.
- Use `torch.nn.functional.smooth_l1_loss` (not `F.l1_loss`), with `beta=0.05` to match the beta of the existing SoftWeightSmoothL1Loss modules.
- `reduction='mean'` — averages over `B * 22 * 2 = 44*B` scalars.
- The loss key is exactly `'loss/reproj/train'` (same prefix convention as `loss/joints/train`, `loss/depth/train`, `loss/uv/train`).
- The `if self.reproj_loss_weight > 0.0` guard keeps the zero-weight path free of the Python loop (important for any re-run of the baseline config).
- Do NOT add `.detach()` on any tensor inside the reprojection computation — gradients must flow through BOTH `pred_pelvis_i` and `pred['joints']`. (The GT branch uses `gt_depth`/`gt_uv`/`gt_joints` which are non-leaf but have `requires_grad=False`; that is fine — smooth_l1 does not require gt to be differentiable.)

#### 4. No changes to `forward()` or `predict()`

`forward()` output dict is unchanged (still contains `joints`, `pelvis_depth`, `pelvis_uv`). `predict()` returns the same InstanceData structure.

---

## Config Changes

### `config.py`

In the `head=dict(...)` inside `model`, add `reproj_loss_weight=0.5`:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
    reproj_loss_weight=0.5,
),
```

All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, seed, pretrained weights) are identical to the baseline. `custom_imports` list is unchanged.

---

## Exact Config Values (unchanged from baseline except reproj_loss_weight)

| Parameter | Value |
|-----------|-------|
| optimizer | AdamW, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.03 |
| backbone lr_mult | 0.1 |
| clip_grad max_norm | 1.0 |
| accumulative_counts | 8 (effective batch 32) |
| LR schedule | LinearLR (epoch 0-3, start_factor=0.333) + CosineAnnealingLR (epoch 3-20, eta_min=0), both convert_to_iter_based=True |
| seed | 2026 |
| batch_size | 4 |
| hidden_dim | 256 |
| num_heads | 8 |
| dropout | 0.1 |
| loss_joints loss_weight | 1.0 |
| loss_depth loss_weight | 1.0 (× loss_weight_depth=1.0) |
| loss_uv loss_weight | 1.0 (× loss_weight_uv=1.0) |
| **reproj_loss_weight** | **0.5 (new)** |
| num_epochs | 20 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Loss restricted to body joints 0-21 only. Reprojection loss also restricted to body joints 0-21 (`_BODY_J = list(range(0, 22))`).
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present; keep unchanged.
4. No Python `import` statements in `config.py` — use only `__import__()` or literals. `reproj_loss_weight=0.5` is a float literal.
5. Head file uses ABSOLUTE imports (since it lives outside the mmpose package). The existing `from pelvis_utils import ...` pattern is correct; extend it rather than adding a relative import.
6. `K` is read from `ds.metainfo['K']` exactly as in `compute_mpjpe_abs`. Convert to `numpy.asarray(..., dtype=np.float32)`; do NOT attempt to read it as a torch tensor.
7. `img_shape` is read from `ds.metainfo.get('img_shape', (640, 384))` — same fallback as `compute_mpjpe_abs`.
8. Use `smooth_l1_loss` with `beta=0.05` (matches other losses), `reduction='mean'`.
9. Projection normalises to `[-1, 1]` using `(2*u_px/crop_w - 1, 2*v_px/crop_h - 1)` to match the `pelvis_uv` convention so pred and GT are in the same numerical regime.
10. Clamp `X >= 0.01` (same as `SubtractRootJoint`) in `project_joints_to_2d` to avoid NaN gradients when a joint is near/behind the camera.
11. The reprojection loss term is added to the `losses` dict under key `'loss/reproj/train'`. Do not add any new entries to the MPJPE-averaging no_grad block; those stay unchanged.
12. No changes to `bedlam_metric.py`, dataset, transforms, backbone, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or `tools/train.py`.
13. Do not add any new hook, scheduler, or optimizer. This design is a pure loss augmentation.
14. `reproj_loss_weight` default value in `__init__` must be `0.0` (so omitting the kwarg reproduces baseline behaviour exactly).

---

## Expected Behaviour After Change

- Training emits an additional scalar `loss/reproj/train` at every logging step, initially in the same order of magnitude as the UV loss (both are normalised to `[-1, 1]`), decreasing as the model learns geometric consistency.
- Validation metrics (`composite_val`, `mpjpe_body_val`, `mpjpe_pelvis_val`, `mpjpe_rel_val`, `mpjpe_hand_val`, `mpjpe_abs_val`) are computed by the unchanged `BedlamMPJPEMetric` — no change to evaluation.
- `MetricsCSVHook` writes the same columns as before; the new train-time scalar is picked up by MMEngine's standard logger/tensorboard (if enabled) but does not modify the CSV schema.
- Per-iteration overhead: one Python loop over `B=4` samples computing two small unprojections + two small projections + one smooth_l1 — negligible (~1 ms) on 1080 Ti.
- Expected result vs. baseline (`composite_val ~168.7`, `mpjpe_pelvis_val ~176`, `mpjpe_abs ~455`): `composite_val` improves by a mild margin (target `< 165`), `mpjpe_abs` improves notably (target `< 420`), `mpjpe_pelvis_val` improves slightly (target `< 173`). Body MPJPE expected neutral to mild positive.
- At inference, no reprojection is computed (the loss path is training-only); the head's `predict()` is unchanged so downstream evaluation is bit-identical to the baseline architecture for equivalent weights.

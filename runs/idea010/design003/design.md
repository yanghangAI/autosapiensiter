# Design 003 — Depth-Weighted Reprojection Loss (geometry-aware, lambda=1.0)

**Design Description:** Add body-joint and pelvis reprojection losses as in design002, but weight each joint's Smooth-L1 pixel error by `X_i / fx` (predicted absolute depth over focal length) to approximately convert normalised pixel error into 3D-equivalent millimetre error, upweighting distant joints whose pixel footprint per mm is smaller; overall weight `reproj_loss_weight=1.0`.

**Starting Point:** `baseline/`

---

## Overview

A naive 2D pixel L1 loss under-weights errors on joints that are far from the camera: a given pixel error at large X corresponds to a larger metric 3D error than the same pixel error at small X, because projection shrinks with `1/X`. For a per-joint weight
```
w_i = X_i / fx
```
the pixel error `|pred_2d - gt_2d| * w_i` is roughly proportional to the 3D-equivalent metric error (up to the crop-coordinate normalisation). Multiplying each joint's Smooth-L1 term by its predicted absolute depth over the crop's `fx` re-balances the reprojection loss into a (crop-scale-normalised) approximation of 3D metric error.

This design inherits Design 002's body-joint + explicit pelvis reprojection terms and adds the depth-weighting multiplier. All other settings identical.

All architecture, optimizer, LR schedule, data pipeline, hooks, seed, batch size, accumulation, and evaluation settings are unchanged from the baseline.

---

## Files to Change

1. `pelvis_utils.py` — add new helper `project_joints_to_2d` (same as design001/002).
2. `pose3d_transformer_head.py` — add depth-weighted reprojection loss terms in `loss()`.
3. `config.py` — add `reproj_loss_weight=1.0`, `reproj_include_pelvis=True`, `reproj_depth_weighted=True` as head kwargs.

No other files are changed.

---

## Algorithm Changes

### `pelvis_utils.py`

**Identical** to design001/002. Add the function:

```python
def project_joints_to_2d(
    joints_abs: torch.Tensor,
    K: np.ndarray,
    crop_h: int,
    crop_w: int,
    x_min: float = 0.01,
) -> torch.Tensor:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    X = joints_abs[..., 0].clamp(min=x_min)
    Y = joints_abs[..., 1]
    Z = joints_abs[..., 2]

    u_px = fx * (-Y / X) + cx
    v_px = fy * (-Z / X) + cy

    u_norm = 2.0 * u_px / float(crop_w) - 1.0
    v_norm = 2.0 * v_px / float(crop_h) - 1.0

    return torch.stack([u_norm, v_norm], dim=-1)
```

### `pose3d_transformer_head.py`

#### 1. Imports

Replace:
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
Add `import numpy as np` near the other top-level imports (right after `import torch`).

#### 2. `Pose3dTransformerHead.__init__` — three new parameters

Add three parameters after `loss_weight_uv`, before `init_cfg`:

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
    reproj_include_pelvis: bool = False,
    reproj_depth_weighted: bool = False,
    init_cfg: OptConfigType = None,
):
    ...
    self.reproj_loss_weight = reproj_loss_weight
    self.reproj_include_pelvis = reproj_include_pelvis
    self.reproj_depth_weighted = reproj_depth_weighted
```

#### 3. `loss()` — depth-weighted reprojection terms

Inside `Pose3dTransformerHead.loss`, AFTER `losses['loss/uv/train'] = ...` and BEFORE the `with torch.no_grad():` block, add:

```python
# ── Auxiliary 2D reprojection loss (optionally depth-weighted) ───────
if self.reproj_loss_weight > 0.0:
    _BODY_J = list(range(0, 22))
    B = pred['joints'].size(0)
    joints_loss_terms = []
    pelvis_loss_terms = []
    for i in range(B):
        ds = batch_data_samples[i]
        K = np.asarray(ds.metainfo.get('K'), dtype=np.float32)
        img_shape = ds.metainfo.get('img_shape', (640, 384))
        crop_h_i, crop_w_i = int(img_shape[0]), int(img_shape[1])
        fx_i = float(K[0, 0])

        pred_pelvis_i = _recover_pelvis_3d(
            pred['pelvis_depth'][i:i+1],
            pred['pelvis_uv'][i:i+1],
            K, crop_h_i, crop_w_i,
        )  # (1, 3)
        gt_pelvis_i = _recover_pelvis_3d(
            gt_depth[i:i+1], gt_uv[i:i+1], K, crop_h_i, crop_w_i,
        )  # (1, 3)

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

        # Per-joint Smooth-L1 with reduction='none' → (1, 22, 2).
        err_i = torch.nn.functional.smooth_l1_loss(
            pred_2d_i, gt_2d_i, beta=0.05, reduction='none')

        if self.reproj_depth_weighted:
            # Use PREDICTED absolute X, clamped to x_min=0.01, as the weight.
            # Weight is detached from the graph (used only as a gain on the
            # error, not as a learning target) to keep gradients flowing
            # through pred_2d, not through the weight magnitude.
            X_body = pred_body_abs_i[..., 0].clamp(min=0.01)  # (1, 22)
            w_i = (X_body / fx_i).detach().unsqueeze(-1)      # (1, 22, 1)
            err_i = err_i * w_i

        # Reduce to scalar for this sample
        joints_loss_terms.append(err_i.mean())

        if self.reproj_include_pelvis:
            pred_pelvis_2d_i = _project_joints_to_2d(
                pred_pelvis_i.unsqueeze(1), K, crop_h_i, crop_w_i)  # (1, 1, 2)
            gt_pelvis_2d_i = _project_joints_to_2d(
                gt_pelvis_i.unsqueeze(1), K, crop_h_i, crop_w_i)    # (1, 1, 2)
            err_p_i = torch.nn.functional.smooth_l1_loss(
                pred_pelvis_2d_i, gt_pelvis_2d_i, beta=0.05, reduction='none')
            if self.reproj_depth_weighted:
                X_p = pred_pelvis_i[..., 0].clamp(min=0.01)       # (1,)
                w_p = (X_p / fx_i).detach().view(1, 1, 1)
                err_p_i = err_p_i * w_p
            pelvis_loss_terms.append(err_p_i.mean())

    joints_loss = torch.stack(joints_loss_terms).mean()
    losses['loss/reproj/train'] = self.reproj_loss_weight * joints_loss

    if self.reproj_include_pelvis:
        pelvis_loss = torch.stack(pelvis_loss_terms).mean()
        losses['loss/reproj_pelvis/train'] = (
            self.reproj_loss_weight * pelvis_loss)
```

Key constraints:
- Weight `w_i = X / fx` is a **detached** tensor — it serves as a geometry-aware gain, NOT as a target to be optimised. Leaving it attached to the graph would create a pathological gradient that rewards the model for predicting smaller X (which would lower the weight and the loss without reducing error). Detaching aligns the loss with the intent: change behaviour only through the error term.
- X used for weighting is the PREDICTED absolute X: `pred_body_abs_i[..., 0]` for body joints, `pred_pelvis_i[..., 0]` for the pelvis term. Clamp to `>= 0.01` before dividing.
- `fx_i` is extracted as a python float from `K[0, 0]`; broadcasting works because it's a scalar divisor.
- Per-sample Python loop mirrors `compute_mpjpe_abs`; do not refactor.
- `err_i.mean()` reduces over (1, 22, 2); `err_p_i.mean()` reduces over (1, 1, 2). Both yield a scalar per sample. `torch.stack(...).mean()` gives the batch-mean scalar.
- Two losses dict keys exactly as before: `'loss/reproj/train'` and `'loss/reproj_pelvis/train'`.
- Do NOT `.detach()` any of the error tensors or their inputs (pred_2d, etc.) — only the `w_i` weight is detached.

#### 4. No changes to `forward()` or `predict()`.

---

## Config Changes

### `config.py`

In the `head=dict(...)` inside `model`, add three kwargs:

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
    reproj_loss_weight=1.0,
    reproj_include_pelvis=True,
    reproj_depth_weighted=True,
),
```

All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, seed, pretrained weights) are identical to the baseline. `custom_imports` list is unchanged.

---

## Exact Config Values (unchanged from baseline except three new kwargs)

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
| **reproj_loss_weight** | **1.0 (new)** |
| **reproj_include_pelvis** | **True (new)** |
| **reproj_depth_weighted** | **True (new)** |
| num_epochs | 20 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Loss restricted to body joints 0-21 only. Reprojection loss also restricted to body joints 0-21.
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present; keep unchanged.
4. No Python `import` statements in `config.py` — use only `__import__()` or literals. All three new kwargs are float/bool literals.
5. Head file uses ABSOLUTE imports; extend the existing `from pelvis_utils import ...` line as shown.
6. `K` is read per-sample from `ds.metainfo['K']`, converted with `numpy.asarray(..., dtype=np.float32)`.
7. `img_shape` defaults to `(640, 384)` if missing.
8. Smooth-L1 with `beta=0.05`, `reduction='none'` to preserve per-element weighting; final reductions are `.mean()` per sample and `torch.stack(...).mean()` over the batch.
9. Clamp `X >= 0.01` both inside `project_joints_to_2d` and when computing the depth-weight `w_i`.
10. The depth-weight `w_i` MUST be detached (`.detach()` applied to `X/fx` before multiplying). This is load-bearing — see the algorithm section above.
11. Weight normalisation: absolute raw magnitudes of the depth-weighted loss will be much smaller than the unweighted version because `X/fx ≈ 2.5m / 500 ≈ 5e-3`. This is expected and is NOT a bug; the overall `reproj_loss_weight=1.0` compensates. Do NOT add a re-scaling factor.
12. Default parameter values in `__init__` are `reproj_loss_weight=0.0`, `reproj_include_pelvis=False`, `reproj_depth_weighted=False` — so omitting them reproduces baseline behaviour exactly.
13. When `reproj_depth_weighted=False` (e.g., a later config reusing this code), the loss falls back to the un-weighted behaviour of design002. This keeps the head usable for multiple idea combinations.
14. No changes to `bedlam_metric.py`, dataset, transforms, backbone, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or `tools/train.py`.

---

## Expected Behaviour After Change

- Training emits `loss/reproj/train` and `loss/reproj_pelvis/train`, with **numerically smaller** magnitudes than design002 because of the `X/fx` scaling factor. This is expected.
- Per-iteration overhead: very small delta over design002 (one extra pointwise multiply and a clamp on X).
- Expected vs. baseline and design002:
  - If the 3D-equivalent rescaling is meaningful, `mpjpe_abs` should improve further than design002 (target `mpjpe_abs < 380`) because distant joints get proportionally stronger supervision.
  - `mpjpe_body_val` may also see a small gain from geometric re-balancing.
  - `composite_val` target `< 158`.
  - Risk: depth-weighting is an approximation; it could under-weight near-camera joints (e.g., hands close to the lens) and slightly hurt hand MPJPE. Acceptable because hand MPJPE is not in the composite metric and the design is body-only.
- At inference, no reprojection is computed; `predict()` is unchanged.

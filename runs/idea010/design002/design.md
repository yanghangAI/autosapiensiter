# Design 002 — Reprojection Loss with Explicit Pelvis Term (lambda=1.0, stronger coupling)

**Design Description:** Add two auxiliary Smooth-L1 reprojection losses: one on body joints 0-21 (weight=1.0) and one on the pelvis-only 2D projection (weight=1.0), giving the pelvis pathway a direct image-space supervision signal in addition to the joint-wide reprojection; both share the same total weight factor `reproj_loss_weight=1.0`.

**Starting Point:** `baseline/`

---

## Overview

Design 002 extends Design 001 by explicitly adding a **pelvis-only** reprojection term in addition to the body-joint reprojection term, and doubles the overall auxiliary weight from 0.5 to 1.0. Rationale: body-joint reprojection couples joint + pelvis pathways via abs-joint positions, but its gradient on pelvis depth/UV is diluted across 22 joints. A separate pelvis reprojection term concentrates the image-space supervision directly on the pelvis prediction, attacking the pelvis MPJPE plateau (174-185 mm across all prior ideas) that the idea targets.

Concretely, in addition to the body-joint 2D loss we add:
```
L_pelvis_reproj = smooth_l1(pred_pelvis_2d, gt_pelvis_2d)
```
where `pred_pelvis_2d` is the projection of `recover_pelvis_3d(pred_depth, pred_uv, K, H, W)` through K, and `gt_pelvis_2d` is the analogous projection of `recover_pelvis_3d(gt_depth, gt_uv, K, H, W)`. Note: because the unprojection and reprojection are algebraic inverses, `pred_pelvis_2d` is numerically equal to `pred_uv` up to the X-clamp — the gradient signal is not redundant with `loss/uv/train` because (a) the clamp introduces gating, and (b) the two losses can be independently rebalanced via the lambda, giving the optimizer a stronger geometric-consistency pressure when lambda is non-zero. Most importantly, the **body joint reprojection** term provides a non-trivial gradient to pelvis depth/UV because the body-joint 2D projections depend on the absolute pelvis position.

All architecture, optimizer, LR schedule, data pipeline, hooks, seed, batch size, accumulation, and evaluation settings are unchanged from the baseline.

---

## Files to Change

1. `pelvis_utils.py` — add new helper `project_joints_to_2d` (same as design001).
2. `pose3d_transformer_head.py` — add two reprojection loss terms in `loss()`.
3. `config.py` — add `reproj_loss_weight=1.0` and `reproj_include_pelvis=True` as head kwargs.

No other files are changed.

---

## Algorithm Changes

### `pelvis_utils.py`

**Identical** to design001. Add a new function `project_joints_to_2d`:

```python
def project_joints_to_2d(
    joints_abs: torch.Tensor,
    K: np.ndarray,
    crop_h: int,
    crop_w: int,
    x_min: float = 0.01,
) -> torch.Tensor:
    """Project absolute camera-frame joints to normalised [-1, 1] pixel coords.

    BEDLAM2 convention:
        u_px = fx * (-Y / X) + cx
        v_px = fy * (-Z / X) + cy
    Then normalise:
        u_norm = 2 * u_px / crop_w - 1
        v_norm = 2 * v_px / crop_h - 1
    """
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

#### 2. `Pose3dTransformerHead.__init__` — two new parameters

Add two parameters after `loss_weight_uv`, before `init_cfg`:

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
    init_cfg: OptConfigType = None,
):
    ...
    self.reproj_loss_weight = reproj_loss_weight
    self.reproj_include_pelvis = reproj_include_pelvis
```

#### 3. `loss()` — two reprojection terms

Inside `Pose3dTransformerHead.loss`, AFTER `losses['loss/uv/train'] = ...` and BEFORE the `with torch.no_grad():` block, add:

```python
# ── Auxiliary 2D reprojection loss (body joints + optional pelvis) ───
if self.reproj_loss_weight > 0.0:
    _BODY_J = list(range(0, 22))
    B = pred['joints'].size(0)
    pred_2d_joints_list = []
    gt_2d_joints_list = []
    pred_pelvis_2d_list = []
    gt_pelvis_2d_list = []
    for i in range(B):
        ds = batch_data_samples[i]
        K = np.asarray(ds.metainfo.get('K'), dtype=np.float32)
        img_shape = ds.metainfo.get('img_shape', (640, 384))
        crop_h_i, crop_w_i = int(img_shape[0]), int(img_shape[1])

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

        pred_2d_joints_list.append(_project_joints_to_2d(
            pred_body_abs_i, K, crop_h_i, crop_w_i))  # (1, 22, 2)
        gt_2d_joints_list.append(_project_joints_to_2d(
            gt_body_abs_i, K, crop_h_i, crop_w_i))    # (1, 22, 2)

        if self.reproj_include_pelvis:
            pred_pelvis_2d_list.append(_project_joints_to_2d(
                pred_pelvis_i.unsqueeze(1), K, crop_h_i, crop_w_i))  # (1, 1, 2)
            gt_pelvis_2d_list.append(_project_joints_to_2d(
                gt_pelvis_i.unsqueeze(1), K, crop_h_i, crop_w_i))    # (1, 1, 2)

    pred_2d = torch.cat(pred_2d_joints_list, dim=0)   # (B, 22, 2)
    gt_2d = torch.cat(gt_2d_joints_list, dim=0)       # (B, 22, 2)

    reproj_joints_raw = torch.nn.functional.smooth_l1_loss(
        pred_2d, gt_2d, beta=0.05, reduction='mean')
    losses['loss/reproj/train'] = self.reproj_loss_weight * reproj_joints_raw

    if self.reproj_include_pelvis:
        pred_pelvis_2d = torch.cat(pred_pelvis_2d_list, dim=0)  # (B, 1, 2)
        gt_pelvis_2d = torch.cat(gt_pelvis_2d_list, dim=0)      # (B, 1, 2)
        reproj_pelvis_raw = torch.nn.functional.smooth_l1_loss(
            pred_pelvis_2d, gt_pelvis_2d, beta=0.05, reduction='mean')
        losses['loss/reproj_pelvis/train'] = (
            self.reproj_loss_weight * reproj_pelvis_raw)
```

Key constraints:
- Two separate losses dict keys: `'loss/reproj/train'` (body joints) and `'loss/reproj_pelvis/train'` (pelvis-only). Both are gated by the weight `self.reproj_loss_weight`; the pelvis term is additionally gated by `self.reproj_include_pelvis`.
- Both losses use Smooth-L1 with `beta=0.05`, `reduction='mean'`.
- The pelvis projection uses `pred_pelvis_i.unsqueeze(1)` to get shape `(1, 1, 3)` so `project_joints_to_2d` accepts it uniformly.
- Do NOT add `.detach()` on any tensor in the pred branch — gradients must flow through `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']`.
- The per-sample Python loop mirrors `compute_mpjpe_abs`; do not vectorise.

#### 4. No changes to `forward()` or `predict()`.

---

## Config Changes

### `config.py`

In the `head=dict(...)` inside `model`, add two kwargs:

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
),
```

All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, seed, pretrained weights) are identical to the baseline. `custom_imports` list is unchanged.

---

## Exact Config Values (unchanged from baseline except two new kwargs)

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
| num_epochs | 20 |

---

## Constraints and Invariants the Builder Must Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. Loss restricted to body joints 0-21 only. Reprojection loss also restricted to body joints 0-21.
3. `custom_imports` in `config.py` must include `'pose3d_transformer_head'` — already present; keep unchanged.
4. No Python `import` statements in `config.py` — use only `__import__()` or literals. `reproj_loss_weight=1.0` is a float; `reproj_include_pelvis=True` is a bool.
5. Head file uses ABSOLUTE imports; extend the existing `from pelvis_utils import ...` line as shown.
6. `K` is read per-sample from `ds.metainfo['K']`, converted with `numpy.asarray(..., dtype=np.float32)`.
7. `img_shape` defaults to `(640, 384)` if missing (same as `compute_mpjpe_abs`).
8. Use `smooth_l1_loss` with `beta=0.05`, `reduction='mean'` for BOTH new loss terms.
9. Clamp `X >= 0.01` in `project_joints_to_2d`.
10. Losses dict keys are exactly `'loss/reproj/train'` and `'loss/reproj_pelvis/train'`.
11. `reproj_include_pelvis` controls the pelvis-only term; when False, only the body-joint reprojection term is added (matches design001 but with a different weight).
12. Default parameter values in `__init__` are `reproj_loss_weight=0.0` and `reproj_include_pelvis=False` — so omitting them reproduces baseline behaviour exactly.
13. Pelvis reprojection applies the SAME lambda (`reproj_loss_weight`) as the body-joint reprojection; do not introduce a separate scalar.
14. No changes to `bedlam_metric.py`, dataset, transforms, backbone, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or `tools/train.py`.

---

## Expected Behaviour After Change

- Training emits two new scalars `loss/reproj/train` and `loss/reproj_pelvis/train`. The pelvis term will be small (mainly a consistency check on `pelvis_uv` vs. its unproject/reproject round-trip) and will shape the pelvis gradient more directly than the body-joint term alone.
- Validation metrics computed unchanged by `BedlamMPJPEMetric`.
- Per-iteration overhead: same order as design001 (Python loop over `B=4`).
- Expected vs. baseline (`composite_val ~168.7`, `mpjpe_pelvis_val ~176`, `mpjpe_abs ~455`):
  - Stronger pelvis-pathway supervision ⇒ expect larger pelvis MPJPE gain than design001 (target `mpjpe_pelvis_val < 170`).
  - Stronger overall coupling ⇒ expect larger `mpjpe_abs` gain (target `< 400`).
  - `composite_val` target `< 160`.
  - Risk: with `lambda=1.0`, early-training instability is more likely than design001's 0.5 — mitigated by `smooth_l1` (bounded gradient) and the X-clamp.
- At inference, no reprojection is computed; `predict()` is unchanged.

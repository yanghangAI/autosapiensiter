# Design 003 — Implementation Summary

**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pelvis_utils.py`: Added the same `project_joints_to_2d(joints_abs, K, crop_h, crop_w, x_min=0.01)` helper as design001/002 (BEDLAM2 projection to normalised `[-1, 1]` pixel coords with `X >= x_min` clamp).
- `code/pose3d_transformer_head.py`: Added `import numpy as np`; extended the `from pelvis_utils import ...` with `recover_pelvis_3d` and `project_joints_to_2d`. Added three new `__init__` kwargs `reproj_loss_weight: float = 0.0`, `reproj_include_pelvis: bool = False`, `reproj_depth_weighted: bool = False` (stored on `self`). In `loss()`, after `loss/uv/train` and before the no_grad MPJPE block, added a gated (`if self.reproj_loss_weight > 0.0`) depth-weighted reprojection loss: for each sample computes absolute body joints (pred and GT), projects them to 2D, takes per-element `smooth_l1_loss(beta=0.05, reduction='none')`, and, when `reproj_depth_weighted` is true, multiplies by a **detached** per-joint weight `w_i = clamp(pred_X, min=0.01) / fx` (shape `(1, 22, 1)`). The per-sample means are stacked and averaged to yield `losses['loss/reproj/train']` scaled by `reproj_loss_weight`. When `reproj_include_pelvis` is true, the pelvis is projected the same way with analogous detached depth weighting `w_p = clamp(pred_pelvis_X, 0.01) / fx` and written to `losses['loss/reproj_pelvis/train']` with the same scale. Weight is detached to avoid a pathological gradient incentive to shrink predicted X.
- `code/config.py`: Added `reproj_loss_weight=1.0`, `reproj_include_pelvis=True`, `reproj_depth_weighted=True` to the `head=dict(...)` kwargs; no other changes.

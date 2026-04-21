# Design 001 — Implementation Summary

**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pelvis_utils.py`: Added new helper `project_joints_to_2d(joints_abs, K, crop_h, crop_w, x_min=0.01)` that projects `(B, J, 3)` absolute camera-frame joints through `K` using the BEDLAM2 convention `u_px = fx*(-Y/X) + cx`, `v_px = fy*(-Z/X) + cy`, then normalises to `[-1, 1]` matching the `pelvis_uv` convention. Clamps `X >= x_min` before the division to avoid NaN/Inf gradients.
- `code/pose3d_transformer_head.py`: Added `import numpy as np`; extended the `from pelvis_utils import ...` to also import `recover_pelvis_3d` and `project_joints_to_2d`. Added new `__init__` kwarg `reproj_loss_weight: float = 0.0` (stored as `self.reproj_loss_weight`). In `loss()`, after the `loss/uv/train` line and before the `with torch.no_grad()` MPJPE block, added a gated (`if self.reproj_loss_weight > 0.0`) auxiliary 2D reprojection loss over body joints 0-21: it builds per-sample absolute body joints (pred_pelvis + pred_joints and gt_pelvis + gt_joints via `recover_pelvis_3d`), projects both with `project_joints_to_2d`, computes `smooth_l1_loss(beta=0.05, reduction='mean')`, scales by `self.reproj_loss_weight`, and writes to `losses['loss/reproj/train']`.
- `code/config.py`: Added `reproj_loss_weight=0.5` to the `head=dict(...)` kwargs; no other changes.

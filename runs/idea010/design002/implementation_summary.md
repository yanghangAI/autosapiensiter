# Design 002 — Implementation Summary

**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pelvis_utils.py`: Added the same `project_joints_to_2d(joints_abs, K, crop_h, crop_w, x_min=0.01)` helper as design001 (projects absolute camera-frame joints to normalised `[-1, 1]` pixel coords using the BEDLAM2 projection convention and the same `X >= x_min` clamp as `SubtractRootJoint`).
- `code/pose3d_transformer_head.py`: Added `import numpy as np`; extended the `from pelvis_utils import ...` with `recover_pelvis_3d` and `project_joints_to_2d`. Added two new `__init__` kwargs `reproj_loss_weight: float = 0.0` and `reproj_include_pelvis: bool = False` (stored on `self`). In `loss()`, after `loss/uv/train` and before the no_grad MPJPE block, added a `if self.reproj_loss_weight > 0.0` block that (a) builds per-sample absolute body joints for pred and GT, projects them, and writes `losses['loss/reproj/train'] = self.reproj_loss_weight * smooth_l1_loss(pred_2d, gt_2d, beta=0.05, reduction='mean')`, and (b) when `self.reproj_include_pelvis` is true, also projects the per-sample pelvis (via `recover_pelvis_3d`) through `project_joints_to_2d` for both pred and GT and writes an analogous `losses['loss/reproj_pelvis/train']` term using the same `reproj_loss_weight` scale.
- `code/config.py`: Added `reproj_loss_weight=1.0` and `reproj_include_pelvis=True` to the `head=dict(...)` kwargs; no other changes.

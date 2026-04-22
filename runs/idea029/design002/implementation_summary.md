**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pelvis_utils.py`: Added `recover_abs_joints_batched` function (identical to design001); reconstructs absolute body joint positions with gradient, returning `(B, num_body_joints, 3)` tensors.
- `code/pose3d_transformer_head.py`: Identical changes to design001 — added import, four new `__init__` kwargs, attribute storage, and absolute joint loss block; the `abs_axis_weights` buffer path is exercised here since `abs_joint_axis_weights=[2.0, 1.0, 1.0]` is passed from config.
- `code/config.py`: Added `abs_joint_loss_weight=0.5`, `abs_joint_indices=22`, and `abs_joint_axis_weights=[2.0, 1.0, 1.0]` to the `head=dict(...)` block; the X-axis (forward/depth) residuals are weighted 2× to amplify gradient to the depth head for the dominant depth-axis absolute error.

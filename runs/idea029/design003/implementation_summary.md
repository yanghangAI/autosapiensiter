**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pelvis_utils.py`: Added `recover_abs_joints_batched` function (identical to designs 001/002); reconstructs absolute body joint positions with gradient, returning `(B, num_body_joints, 3)` tensors.
- `code/pose3d_transformer_head.py`: Identical changes to designs 001/002 — added import, four new `__init__` kwargs, attribute storage, and absolute joint loss block; for this design `abs_joint_pelvis_grad_scale=0.5 < 1.0` triggers the selective stop-gradient branch that computes `pred_abs = 0.5 * pred_abs_full + 0.5 * pred_abs_det`, giving full gradient to the joint branch and half gradient to the pelvis branch.
- `code/config.py`: Added `abs_joint_loss_weight=0.5`, `abs_joint_indices=22`, and `abs_joint_pelvis_grad_scale=0.5` to the `head=dict(...)` block to enable selective pelvis gradient scaling and prevent the absolute loss from over-competing with `L_depth`/`L_uv`.

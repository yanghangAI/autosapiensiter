**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pelvis_utils.py`: Added `recover_abs_joints_batched` function after `compute_mpjpe_abs`; this function reconstructs predicted and GT absolute body joint positions (with full gradient flow) by adding per-sample unprojected pelvis 3D to root-relative joints, returning `(B, num_body_joints, 3)` tensors.
- `code/pose3d_transformer_head.py`: Added import of `recover_abs_joints_batched`; added four new `__init__` kwargs (`abs_joint_loss_weight=0.0`, `abs_joint_indices=22`, `abs_joint_axis_weights=None`, `abs_joint_pelvis_grad_scale=1.0`) with attributes stored in `__init__`; added absolute body joint smooth-L1 loss block (λ=0.5, β=0.05, uniform per-axis weights, full gradient to both joint and pelvis heads) in `loss()` after the existing three loss lines.
- `code/config.py`: Added `abs_joint_loss_weight=0.5` and `abs_joint_indices=22` to the `head=dict(...)` block to enable the absolute joint loss at uniform weight.

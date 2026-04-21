**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `use_uncertainty_weighting: bool = False`, `uncertainty_pelvis_only: bool = False`, and `joint_loss_scale: float = 1.0` constructor parameters. When `uncertainty_pelvis_only=True`, registers `log_var_depth` and `log_var_uv` as learnable `nn.Parameter` scalars (initialised to 0). In `loss()`, `joint_loss_scale` is applied as a fixed multiplier to the raw joint loss before the conditional — when `uncertainty_pelvis_only=True`, depth and UV use the uncertainty formula while the scaled joint loss stays fixed. `_train_mpjpe` computation is unaffected by `joint_loss_scale` (it remains a plain MPJPE diagnostic in mm).
- `code/config.py`: Added `uncertainty_pelvis_only=True` and `joint_loss_scale=2.0` to the head dict, encoding the composite metric's 0.67:0.33 body/pelvis weighting as a fixed prior (2.0 ≈ 0.67/0.33) for the joint loss anchor.

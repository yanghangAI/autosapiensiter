**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `use_uncertainty_weighting: bool = False` and `uncertainty_pelvis_only: bool = False` constructor parameters. When `uncertainty_pelvis_only=True`, registers only two learnable `nn.Parameter` scalars (`log_var_depth`, `log_var_uv`, both initialised to 0) — no `log_var_joints`. In `loss()`, joint loss is always applied with fixed weight 1.0 (anchored), while depth and UV losses apply the uncertainty formula (`exp(-lv) * raw + lv`) with clamped log-vars when `uncertainty_pelvis_only=True`. This protects body MPJPE from being down-weighted while self-balancing the two pelvis sub-tasks.
- `code/config.py`: Added `uncertainty_pelvis_only=True` to the head dict; `use_uncertainty_weighting` is not set (defaults to False) so both flags are never active simultaneously.

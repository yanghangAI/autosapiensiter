**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `use_uncertainty_weighting: bool = False` constructor parameter and stored as `self.use_uncertainty_weighting`. When True, registers three learnable `nn.Parameter` scalars (`log_var_joints`, `log_var_depth`, `log_var_uv`, all initialised to 0). In `loss()`, raw losses are computed first, then if `use_uncertainty_weighting=True` the Kendall & Gal uncertainty formula (`exp(-lv) * raw_loss + lv`) is applied with each `log_var` clamped to `[-4, 4]` via a local variable (so gradients flow). When False, behaviour is identical to baseline.
- `code/config.py`: Added `use_uncertainty_weighting=True` to the head dict; `loss_weight_depth` and `loss_weight_uv` remain at 1.0 as effective no-ops since the uncertainty formula subsumes them.

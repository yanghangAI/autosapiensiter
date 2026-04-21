**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `per_joint_difficulty_weighting`, `ema_alpha`, and `ema_momentum` constructor parameters; registered `joint_err_ema` (ones(22)) and `_train_iter` (zeros(1, long)) buffers when weighting is enabled; added `_get_adaptive_weights()` method that computes per-joint weights via linear power-law normalisation (alpha=0.5) over the EMA error estimates; replaced the body joint loss line with a conditional block that updates the EMA each step and applies per-joint weights via manual smooth-L1 computation (beta=0.05) when `per_joint_difficulty_weighting=True`, falling back to the original `loss_joints_module` call otherwise.
- `code/config.py`: Added three literal kwargs to `model.head` dict: `per_joint_difficulty_weighting=True`, `ema_alpha=0.5`, `ema_momentum=0.99` to enable mild (alpha=0.5, square-root) per-joint difficulty weighting.

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `per_joint_difficulty_weighting`, `ema_alpha`, `ema_momentum`, `weight_norm`, and `weight_temperature` constructor parameters; registered `joint_err_ema` (ones(22)) and `_train_iter` (zeros(1, long)) buffers when weighting is enabled; added `_get_adaptive_weights()` method supporting both `'softmax'` normalisation (temperature-scaled softmax, `w = 22 * softmax(ema / T)`) and `'linear'` fallback; replaced the body joint loss line with a conditional block that updates the EMA each step and applies per-joint weights via manual smooth-L1 computation (beta=0.05) when `per_joint_difficulty_weighting=True`; added `import torch.nn.functional as F` at module top-level.
- `code/config.py`: Added five literal kwargs to `model.head` dict: `per_joint_difficulty_weighting=True`, `ema_alpha=1.0`, `ema_momentum=0.99`, `weight_norm='softmax'`, `weight_temperature=1.0` to enable full-proportional (alpha=1.0) softmax-normalised per-joint difficulty weighting.

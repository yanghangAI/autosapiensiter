# Design 001 — Implementation Summary

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pose3d_transformer_head.py`: Added six new `__init__` kwargs (`depth_head_type`, `num_depth_bins`, `depth_range_min`, `depth_range_max`, `depth_soft_label_sigma`, `depth_aux_reg_weight`) with defaults that reproduce baseline regression behaviour. When `depth_head_type != 'regression'`, `self.depth_out` is allocated as `Linear(hidden_dim, K)` and a non-persistent `log_bin_centres` buffer (log-uniform) is registered. `forward()` computes a soft-argmax expectation `pelvis_depth = (softmax(logits) * exp(log_bin_centres)).sum(...)` in classification modes and now returns `depth_logits` and `depth_bin_centres` alongside `pelvis_depth`/`pelvis_uv`/`joints`. `loss()` now branches by mode: for classification it computes SORD soft-target cross-entropy using a Gaussian target in log-depth space with σ = `depth_soft_label_sigma × bin_width_log` (target is `.detach()`ed). An auxiliary `F.smooth_l1_loss` on the expected depth is emitted as `loss/depth_reg/train` when `depth_aux_reg_weight > 0` (inactive for Design 001 since the weight is 0.0). The shared head also handles `classification_adaptive` (Design 003) via an extra `depth_bins_head` that only activates when requested. Added `import torch.nn.functional as F` at the top to avoid a local import inside `loss()`.
- `code/config.py`: Added the six new head kwargs to the `head=dict(...)` block with Design-001 values: `depth_head_type='classification'`, `num_depth_bins=64`, `depth_range_min=1.0`, `depth_range_max=15.0`, `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight=0.0`. All values are literal `str`/`int`/`float` (MMEngine-safe). No other config values changed.

**Note:** The head file is shared verbatim across designs 001/002/003 (same implementation; designs differ only via config kwargs). This is a shared-signature requirement stated in all three design.md files.

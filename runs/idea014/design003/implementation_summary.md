# Design 003 — Implementation Summary

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pose3d_transformer_head.py`: Identical to Design 001/002 (shared head across idea014). When `depth_head_type == 'classification_adaptive'`, `__init__` additionally allocates `self.depth_bins_head = nn.Linear(hidden_dim, num_depth_bins)` and includes it in the `_init_head_weights` trunc-normal init loop. In `forward()` adaptive branch, per-sample widths are produced by `softmax(depth_bins_head(pelvis_token)) * (depth_range_max - depth_range_min)`, cumsum'd with a prepended zero column to form K+1 edges in `[z_min, z_max]`, and midpoints form K per-sample bin centres. The soft-argmax expectation is computed over these per-sample centres. In `loss()`, the SORD target is computed against `log(pred['depth_bin_centres'])` per sample; `sigma_log` is `depth_soft_label_sigma × median(|Δ log_centre|)` (per sample) in adaptive mode. Crucially, `target = target.detach()` prevents gradient flow from CE into the width head — widths are trained solely by the SmoothL1-on-expectation aux loss (`depth_aux_reg_weight = 0.3`). `bin_centres.clamp(min=depth_range_min*1e-3).log()` guards against `log(0)` NaN.
- `code/config.py`: Same six new head kwargs as Designs 001/002, with Design-003 values: `depth_head_type='classification_adaptive'`, `num_depth_bins=64`, `depth_range_min=1.0`, `depth_range_max=15.0`, `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight=0.3`. All literal str/int/float.

**Note:** The head file is identical across all three idea014 designs; Design 003 activates via the `'classification_adaptive'` string which triggers allocation of the second `depth_bins_head` linear and the adaptive forward/loss branches.

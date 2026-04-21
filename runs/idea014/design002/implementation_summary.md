# Design 002 — Implementation Summary

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pose3d_transformer_head.py`: Identical to Design 001 (shared head across all idea014 designs). The shared implementation supports all four `depth_head_type` values (`regression`, `classification`, `classification_hybrid`, `classification_adaptive`). For `classification_hybrid`, `forward()` computes the fixed log-uniform soft-argmax expectation exactly like Design 001, and `loss()` activates the `if self.depth_aux_reg_weight > 0.0:` branch to emit `loss/depth_reg/train = depth_aux_reg_weight * F.smooth_l1_loss(pelvis_depth_expected, gt_depth, beta=0.05)` — gradients flow through the expectation into the logits in addition to the SORD CE. Target is `.detach()`ed uniformly across modes.
- `code/config.py`: Same six new head kwargs as Design 001, with Design-002 values: `depth_head_type='classification_hybrid'`, `num_depth_bins=64`, `depth_range_min=1.0`, `depth_range_max=15.0`, `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight=0.3`. All literal str/int/float.

**Note:** The head file is identical to Design 001's head; only config values differ. This produces the hybrid CE + SmoothL1 depth loss.

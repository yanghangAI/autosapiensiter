# Design Review Log — idea014 / design002

## 2026-04-17 — APPROVED

- Reviewer mode: design review
- Idea: idea014 (Anchor-Based Pelvis Depth via Discretized Classification Head)
- Design: 002 (Classification + SORD soft CE + auxiliary SmoothL1
  regression hybrid, fixed log-uniform bins)
- Starting point: baseline/
- Files to change (declared): pose3d_transformer_head.py, config.py
- Invariant files unchanged: confirmed (pelvis_utils.py,
  bedlam_metric.py, bedlam2_dataset.py, bedlam2_transforms.py,
  sapiens_rgbd.py, data preprocessor, infra/*, train.py wrapper,
  tools/train.py).
- Key acceptance points:
  - Shared 6-kwarg signature identical to Design 001; only
    `depth_head_type='classification_hybrid'` and
    `depth_aux_reg_weight=0.3` differ.
  - Head allocation identical to Design 001: single
    `Linear(256, 64)` + `log_bin_centres` buffer. Adaptive head NOT
    allocated.
  - `forward()` identical to Design 001 (fixed log-uniform centres).
  - `loss()`: SORD soft-CE (same as Design 001) PLUS
    `loss/depth_reg/train = 0.3 × F.smooth_l1_loss(pelvis_depth, gt_depth, beta=0.05, reduction='mean')`.
  - `F.smooth_l1_loss` uses `beta=0.05` (explicit), reduction='mean'.
  - `depth_aux_reg_weight` applied INSIDE the aux term; does NOT
    multiply `loss_weight_depth`.
  - Gradient from SmoothL1 flows through expectation → softmax →
    `depth_out.weight`, coherent with CE gradient path.
  - Four emitted loss keys (`loss/joints/train`, `loss/depth/train`,
    `loss/uv/train`, `loss/depth_reg/train`); `MetricsCSVHook`
    auto-columns the new key.
  - `predict()` and MPJPE no-grad block unchanged.
  - Config: `depth_head_type='classification_hybrid'`,
    `num_depth_bins=64`, `depth_range_min=1.0`,
    `depth_range_max=15.0`, `depth_soft_label_sigma=1.5`,
    `depth_aux_reg_weight=0.3`. Literals only.
  - Parameter delta identical to Design 001 (+16 191). Runtime
    overhead still negligible.
- Verdict: APPROVED.

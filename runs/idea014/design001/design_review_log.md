# Design Review Log — idea014 / design001

## 2026-04-17 — APPROVED

- Reviewer mode: design review
- Idea: idea014 (Anchor-Based Pelvis Depth via Discretized Classification Head)
- Design: 001 (Pure classification — fixed log-uniform bins, SORD soft
  CE, no aux regression)
- Starting point: baseline/
- Files to change (declared): pose3d_transformer_head.py, config.py
- Invariant files unchanged: confirmed (pelvis_utils.py,
  bedlam_metric.py, bedlam2_dataset.py, bedlam2_transforms.py,
  sapiens_rgbd.py, data preprocessor, infra/*, train.py wrapper,
  tools/train.py).
- Key acceptance points:
  - Shared 6-kwarg `__init__` signature for all three idea014 designs;
    defaults reproduce baseline exactly (`depth_head_type='regression'`,
    etc.). Placed after `loss_weight_uv`, before `init_cfg`.
  - Conditional head allocation: `Linear(256, 64)` replaces
    `Linear(256, 1)` in classification mode; `log_bin_centres` buffer
    registered non-persistent with `torch.linspace(log 1, log 15, 64)`.
  - `forward()` returns same `(B, 1) pelvis_depth` (soft-argmax
    expectation) PLUS `depth_logits` `(B, K)` and
    `depth_bin_centres` `(B, K)` for loss consumption.
  - `loss()` computes SORD soft-CE with σ = 1.5 × bin_width_log;
    GT clamped to `[1, 15]` before `log()`. Aux branch gated
    `if depth_aux_reg_weight > 0.0`; Design 001 sets it to 0.0 so the
    branch is a no-op.
  - `predict()` and MPJPE no-grad block untouched — both read
    `pred['pelvis_depth']`.
  - Config: `depth_head_type='classification'`, `num_depth_bins=64`,
    `depth_range_min=1.0`, `depth_range_max=15.0`,
    `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight=0.0`. All
    literals — MMEngine-compliant, no `import`.
  - Three loss keys identical to baseline. `_compute_mpjpe_abs`
    contract `(B, 1)` metric scalar preserved.
  - Parameter delta: +16 191 float32 weights; runtime <0.1% overhead.
- Verdict: APPROVED.

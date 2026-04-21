# Code Review Log — idea010/design003

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Implementation of Design 003 (depth-weighted reprojection loss + pelvis term, lambda=1.0). `review-check-implementation` passes. `pelvis_utils.py` adds only `project_joints_to_2d`; `pose3d_transformer_head.py` adds three kwargs (`reproj_loss_weight`, `reproj_include_pelvis`, `reproj_depth_weighted`) with baseline-preserving defaults, per-element `smooth_l1_loss(beta=0.05, reduction='none')`, and the load-bearing **detached** depth weight `w = clamp(pred_X, 0.01) / fx` for both body joints and pelvis; `config.py` sets the three kwargs to `1.0/True/True`. No invariant file modified; `train.py` byte-identical to baseline. Reduced test-train completed 81 iters + full val epoch with no errors; iter-50 log shows `loss/reproj/train ≈ 4e-4` and `loss/reproj_pelvis/train ≈ 5.5e-5` (small magnitudes expected per design due to X/fx scaling) and `grad_norm: 8.23` (much lower than design002's 752.8, consistent with intended geometry-aware down-weighting).

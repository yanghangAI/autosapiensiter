# Code Review Log — idea010/design002

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Implementation of Design 002 (body-joint reprojection + explicit pelvis reprojection, lambda=1.0). `review-check-implementation` passes. `pelvis_utils.py` adds only `project_joints_to_2d`; `pose3d_transformer_head.py` adds `reproj_loss_weight` and `reproj_include_pelvis` kwargs (defaults 0.0/False), the reprojection block after `loss/uv/train` with both `loss/reproj/train` and `loss/reproj_pelvis/train` emitted under the same lambda; `config.py` sets `reproj_loss_weight=1.0` and `reproj_include_pelvis=True`. No invariant file modified; `train.py` byte-identical to baseline. Reduced test-train completed 81 iters + full val epoch with no errors; iter-50 log shows both new scalars active (`loss/reproj/train ≈ 12.40`, `loss/reproj_pelvis/train ≈ 3.17`).

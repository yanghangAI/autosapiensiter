# Code Review Log — idea010/design001

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Implementation of Design 001 (auxiliary 2D reprojection Smooth-L1 loss on body joints 0-21 with lambda=0.5). `review-check-implementation` passes. All three files changed match the design spec exactly: `pelvis_utils.py` adds `project_joints_to_2d` only, `pose3d_transformer_head.py` adds `reproj_loss_weight` kwarg (default 0.0) and an `if self.reproj_loss_weight > 0.0` block placed after `loss/uv/train` and before the no_grad MPJPE block, `config.py` adds `reproj_loss_weight=0.5` in the head dict. No invariant files modified; `train.py` byte-identical to baseline. Reduced test-train completed 81 iters + full val epoch with no errors; training log confirms `loss/reproj/train` is emitted and non-trivial (~7.05 at iter 50).

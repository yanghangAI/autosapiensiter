## Code Review — idea029/design003 — APPROVED

**Date:** 2026-04-21

### Automated check
`python scripts/cli.py review-check-implementation runs/idea029/design003` — PASSED.

### Files changed vs. design.md
`implementation_summary.md` lists three files: `code/pelvis_utils.py`, `code/pose3d_transformer_head.py`, `code/config.py`. All required by design.md. No extra files changed.

### pelvis_utils.py
Identical to designs 001/002 (confirmed via diff). Correct.

### pose3d_transformer_head.py
Identical to designs 001/002 (confirmed via diff). For this design, `abs_joint_pelvis_grad_scale=0.5 < 1.0` triggers the `if self.abs_joint_pelvis_grad_scale < 1.0:` branch in `loss()`. The implementation:
- Calls `_recover_abs_joints_batched` twice: once with full `pred_depth`/`pred_uv` (`pred_abs_full`), once with `.detach()` versions (`pred_abs_det`).
- Computes `pred_abs = alpha * pred_abs_full + (1.0 - alpha) * pred_abs_det` with `alpha=0.5`.
- This correctly gives relative joint branch gradient scale 1.0 and pelvis branch gradient scale 0.5, matching the design intent.
`abs_joint_axis_weights` is `None` for this design, so no per-axis weighting applied.

### config.py
`head=dict(...)` contains `abs_joint_loss_weight=0.5`, `abs_joint_indices=22`, and `abs_joint_pelvis_grad_scale=0.5` after `loss_weight_uv=1.0`. No `abs_joint_axis_weights` kwarg (correct). All values are float/int literals. No Python imports added.

### test_output
Test train completed ("Done training! [test] Finished."). SLURM log shows `loss/abs_joints/train: 0.569470` at iter 50 — nearly identical to design001's 0.569480, which is expected because the stop-gradient only changes the gradient flow, not the loss value itself (the loss is computed on `pred_abs` which mixes full and detached pelvis, producing the same forward-pass value as the full-gradient version for this weighting). No runtime errors.

### Invariant check
All invariants preserved. Identical `pose3d_transformer_head.py`, `pelvis_utils.py`, and `train.py` as design001.

**VERDICT: APPROVED**

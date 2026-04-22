## Code Review — idea029/design002 — APPROVED

**Date:** 2026-04-21

### Automated check
`python scripts/cli.py review-check-implementation runs/idea029/design002` — PASSED.

### Files changed vs. design.md
`implementation_summary.md` lists three files: `code/pelvis_utils.py`, `code/pose3d_transformer_head.py`, `code/config.py`. All required by design.md. No extra files changed.

### pelvis_utils.py
Identical to design001 (confirmed via diff). `recover_abs_joints_batched` matches design spec exactly. No gradient-breaking operations.

### pose3d_transformer_head.py
Identical to design001 (confirmed via diff). The `abs_axis_weights` buffer path (`register_buffer`) is present and will be exercised for this design since `abs_joint_axis_weights=[2.0, 1.0, 1.0]` is passed from config. The `loss()` block correctly applies `abs_loss_raw * self.abs_axis_weights.view(1, 1, 3)` when `self.abs_axis_weights is not None`.

### config.py
`head=dict(...)` contains `abs_joint_loss_weight=0.5`, `abs_joint_indices=22`, and `abs_joint_axis_weights=[2.0, 1.0, 1.0]` after `loss_weight_uv=1.0`. No `abs_joint_pelvis_grad_scale` kwarg (correct: default 1.0). All values are float/int/list-of-float literals. No Python imports added.

### test_output
Test train completed ("Done training! [test] Finished."). SLURM log shows `loss/abs_joints/train: 1.002668` at iter 50 — approximately 2× the design001 value of 0.569, consistent with the X-axis (forward) residuals being 2× weighted and X-axis dominating absolute errors. No runtime errors.

### Invariant check
All invariants preserved (same as design001 — identical `pose3d_transformer_head.py` and `pelvis_utils.py`, unchanged `train.py`).

**VERDICT: APPROVED**

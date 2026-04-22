## Code Review — idea029/design001 — APPROVED

**Date:** 2026-04-21

### Automated check
`python scripts/cli.py review-check-implementation runs/idea029/design001` — PASSED.

### Files changed vs. design.md
`implementation_summary.md` lists three files: `code/pelvis_utils.py`, `code/pose3d_transformer_head.py`, `code/config.py`. All three are required by design.md. No extra files changed. `train.py` is identical to baseline (verified via diff).

### pelvis_utils.py
`recover_abs_joints_batched` added after `compute_mpjpe_abs`, exactly as specified. Function signature, per-sample loop structure, use of `recover_pelvis_3d`, slicing to `[:num_body_joints]`, and return of `torch.stack(pred_abs_list), torch.stack(gt_abs_list)` all match the design spec verbatim. No `.detach()`, `.norm()`, or `* 1000.0` applied to gradient-carrying tensors.

### pose3d_transformer_head.py
- Import of `recover_abs_joints_batched as _recover_abs_joints_batched` present at line 37.
- Four new `__init__` kwargs added after `loss_weight_uv`: `abs_joint_loss_weight=0.0`, `abs_joint_indices=22`, `abs_joint_axis_weights=None`, `abs_joint_pelvis_grad_scale=1.0`. Placement and defaults match design.
- `__init__` body stores all four attributes; `register_buffer('abs_axis_weights', w)` path for non-None case present.
- `loss()` block inserted after `losses['loss/uv/train']`, before `with torch.no_grad():`. Smooth-L1 with `beta_abs=0.05`, axis-weight branch, loss key `losses['loss/abs_joints/train']` — all match design exactly.
- `predict()` method unchanged.
- `_BODY`, joint/depth/UV loss lines, `with torch.no_grad():` block all unchanged.

### config.py
`head=dict(...)` contains `abs_joint_loss_weight=0.5` and `abs_joint_indices=22` after `loss_weight_uv=1.0`. No `abs_joint_axis_weights` or `abs_joint_pelvis_grad_scale` kwargs present (correct: this is Design A, uniform weight, full pelvis gradient). All values are float/int literals. No Python import statements added.

### test_output
Test train ran to completion ("Done training! [test] Finished."). SLURM log shows `loss/abs_joints/train: 0.569480` at iter 50, confirming the new loss term is active and in the expected magnitude range (0.5× linear smooth-L1 at typical early absolute errors ~1–3m → ~0.5–1.5). No runtime errors or warnings indicating correctness issues.

### Invariant check
`persistent_workers=False`, seed `2026`, batch 4, accum 8, `max_keep_ckpts=1`, `resume=True`, AMP via `FixedAmpOptimWrapper`, `_BODY=list(range(0,22))` — all unchanged. No modifications to evaluation metric, dataset, transforms, backbone, data preprocessor, or infra files.

**VERDICT: APPROVED**

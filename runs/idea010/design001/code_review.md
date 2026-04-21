# Code Review — idea010/design001

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 001 (Auxiliary 2D Reprojection Loss on body joints, lambda=0.5) is implemented faithfully and completely. All three target files were modified exactly as specified, no invariant file was touched, and the reduced test-train ran to completion (81 train iters + full val epoch) with the new `loss/reproj/train` scalar being computed and propagated.

---

## Infrastructure Check

- `python scripts/cli.py review-check-implementation runs/idea010/design001` → PASS.

## `implementation_summary.md` as Checklist

**Files changed:**
- [x] `code/pelvis_utils.py` — required by design.
- [x] `code/pose3d_transformer_head.py` — required by design.
- [x] `code/config.py` — required by design.

No extra/unauthorised files listed; `code/train.py` is present but is byte-identical to `baseline/train.py` (unchanged).

**Claimed changes vs. actual code:**
- [x] `project_joints_to_2d` added to `pelvis_utils.py` at module level (lines 49–88); signature, formula, clamp `X>=x_min=0.01`, and `[-1,1]` normalisation all match the design snippet verbatim.
- [x] `import numpy as np` added to head file (line 28).
- [x] `from pelvis_utils import (...)` extended with `recover_pelvis_3d as _recover_pelvis_3d` and `project_joints_to_2d as _project_joints_to_2d` (lines 37–41).
- [x] `__init__` signature adds `reproj_loss_weight: float = 0.0` at the correct position (after `loss_weight_uv`, before `init_cfg`, line 166); stored on `self` (line 179). Default 0.0 preserves baseline behaviour.
- [x] `loss()` gains the `if self.reproj_loss_weight > 0.0` block (lines 313–357), placed AFTER the `loss/uv/train` line and BEFORE the `with torch.no_grad()` MPJPE block, exactly as specified.
- [x] `_BODY_J = list(range(0, 22))`, per-sample K via `np.asarray(ds.metainfo.get('K'), dtype=np.float32)`, `img_shape` fallback `(640, 384)`, pred/GT absolute body joints via `recover_pelvis_3d` + `pred['joints'][..., 0:22]`, projection via `project_joints_to_2d`, `smooth_l1_loss(beta=0.05, reduction='mean')`, key `'loss/reproj/train'`, scaled by `self.reproj_loss_weight` — all match the design.
- [x] No `.detach()` on differentiable tensors — gradients flow through `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']`.
- [x] `forward()` and `predict()` unchanged.
- [x] `config.py` adds `reproj_loss_weight=0.5` in `head=dict(...)`; no other config changes.

## Design-Detail Fidelity

Every numbered invariant in the design's "Constraints and Invariants" section (14 items) is satisfied:
1. `persistent_workers=False` preserved.
2. Body joints 0-21 only.
3. `custom_imports` unchanged.
4. No `import` statements in `config.py`; `reproj_loss_weight=0.5` is a float literal.
5. Absolute import pattern preserved.
6. `K` via `np.asarray(..., dtype=np.float32)`.
7. `img_shape` default `(640, 384)`.
8. `smooth_l1_loss` with `beta=0.05`, `reduction='mean'`.
9. `[-1, 1]` normalisation using `2*u/W - 1`.
10. `X >= 0.01` clamp inside `project_joints_to_2d`.
11. Loss dict key exactly `'loss/reproj/train'`.
12. No invariant file modified.
13. No new hook/scheduler/optimizer.
14. `reproj_loss_weight` default `0.0`.

## Invariant-File Compliance

- `diff baseline/pelvis_utils.py design001/code/pelvis_utils.py` → only the new `project_joints_to_2d` function added.
- `diff baseline/pose3d_transformer_head.py design001/code/pose3d_transformer_head.py` → adds numpy import, extends pelvis_utils import, adds `reproj_loss_weight` kwarg + attr, inserts reprojection block; nothing else touched.
- `diff baseline/config.py design001/code/config.py` → only `output_dir` (per-design patch) and the new `reproj_loss_weight=0.5` kwarg.
- `train.py` byte-identical to baseline.
- No changes under `infra/`, `mmpose/evaluation/`, `mmpose/datasets/`, backbone, or data preprocessor.

## Test-Output Sanity

- Reduced test-train ran to completion: 81 training iterations (full epoch 1) logged in `test_output/iter_metrics.csv` and validation completed (1 row in `test_output/metrics.csv`).
- No Error/Traceback/Exception/NaN strings in `slurm_test_55670396.out` or `20260417_135424.log`.
- The training log shows `loss/reproj/train: 7.054009` at iter 50 alongside the baseline loss terms, confirming the reprojection term is active, non-trivial, and correctly back-propagating (visible grad_norm increase vs. baseline is consistent with extra gradient coupling).
- `iter_metrics.csv` still has only the baseline three columns (`loss_joints_train`, `loss_depth_train`, `loss_uv_train`) — this is correct: the CSV schema is defined by the invariant `infra/metrics_csv_hook.py` and is intentionally unchanged.

## Issues

None.

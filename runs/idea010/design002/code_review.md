# Code Review — idea010/design002

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 002 (body-joint reprojection + explicit pelvis reprojection term, lambda=1.0) is implemented faithfully and completely. All three target files were modified exactly as specified; no invariant file was touched; the reduced test-train ran to completion (81 train iters + full val epoch) with both `loss/reproj/train` and `loss/reproj_pelvis/train` scalars emitted and non-trivial.

---

## Infrastructure Check

- `python scripts/cli.py review-check-implementation runs/idea010/design002` → PASS.

## `implementation_summary.md` as Checklist

**Files changed:**
- [x] `code/pelvis_utils.py` — required.
- [x] `code/pose3d_transformer_head.py` — required.
- [x] `code/config.py` — required.

No extra/unauthorised files; `code/train.py` is byte-identical to `baseline/train.py`.

**Claimed changes vs. actual code:**
- [x] `project_joints_to_2d` added to `pelvis_utils.py` (lines 49–88) — identical to design001 and matches the shared design snippet exactly.
- [x] `import numpy as np` added (line 28).
- [x] `from pelvis_utils import (...)` extended with `recover_pelvis_3d` and `project_joints_to_2d`.
- [x] `__init__` signature adds BOTH `reproj_loss_weight: float = 0.0` and `reproj_include_pelvis: bool = False` at the correct position (lines 166–167); both stored on `self` (lines 180–181). Defaults preserve baseline behaviour.
- [x] `loss()` block (lines 315–369) placed AFTER `loss/uv/train` and BEFORE the no_grad MPJPE block, as required.
- [x] `if self.reproj_loss_weight > 0.0` guard wraps the full block; body-joint reprojection is always run inside this guard; pelvis-only term is additionally gated by `if self.reproj_include_pelvis`.
- [x] Both losses use `smooth_l1_loss(beta=0.05, reduction='mean')`, scaled by the same `self.reproj_loss_weight`.
- [x] Pelvis projection uses `pred_pelvis_i.unsqueeze(1)` / `gt_pelvis_i.unsqueeze(1)` to shape `(1,1,3)` so `project_joints_to_2d` receives a uniform `(B, J, 3)` tensor — matches the design.
- [x] Dict keys exactly `'loss/reproj/train'` and `'loss/reproj_pelvis/train'`.
- [x] No `.detach()` on differentiable tensors.
- [x] `forward()` and `predict()` unchanged.
- [x] `config.py` adds `reproj_loss_weight=1.0` and `reproj_include_pelvis=True` in `head=dict(...)`; no other changes.

## Design-Detail Fidelity

Every numbered invariant in the design's "Constraints and Invariants" section (14 items) is satisfied:
1. `persistent_workers=False` preserved.
2. Body joints 0-21 only.
3. `custom_imports` unchanged.
4. No `import` statements in `config.py`; new kwargs are float/bool literals.
5. Absolute import pattern preserved in head file.
6. `K` via `np.asarray(..., dtype=np.float32)`.
7. `img_shape` default `(640, 384)`.
8. Both new loss terms use `smooth_l1_loss(beta=0.05, reduction='mean')`.
9. `X >= 0.01` clamp inside `project_joints_to_2d`.
10. Dict keys correct.
11. `reproj_include_pelvis` gates only the pelvis term.
12. Defaults `reproj_loss_weight=0.0`, `reproj_include_pelvis=False`.
13. Both losses scaled by the SAME `self.reproj_loss_weight` (no separate scalar).
14. No invariant file modified.

## Invariant-File Compliance

- `diff` against baseline shows `pelvis_utils.py` adds only `project_joints_to_2d`; `pose3d_transformer_head.py` adds only the imports, two `__init__` kwargs, two attribute assignments, and the reprojection block; `config.py` differs only in `output_dir` (per-design patch) and the two new head kwargs.
- `train.py` byte-identical to baseline.
- No changes under `infra/`, `mmpose/evaluation/`, `mmpose/datasets/`, backbone, or data preprocessor.

## Test-Output Sanity

- Reduced test-train completed 81 training iterations (full epoch 1) and produced a validation row in `test_output/metrics.csv`.
- No Error/Traceback/Exception/NaN strings in `slurm_test_55670397.out` or `20260417_135348.log`.
- Training log at iter 50 shows `loss/reproj/train: 12.397838` and `loss/reproj_pelvis/train: 3.165267` — both active and non-trivial. Total `loss: 17.56` is consistent with lambda=1.0 on the two new terms plus the baseline three losses. `grad_norm: 752.8` is higher than design001's but still finite and no optimizer failure occurred (clip_grad max_norm=1.0 handles it).
- `iter_metrics.csv` preserves the invariant 3-column schema per `infra/metrics_csv_hook.py`.

## Issues

None.

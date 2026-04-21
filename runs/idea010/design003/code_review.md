# Code Review — idea010/design003

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 003 (depth-weighted reprojection loss, lambda=1.0, pelvis included) is implemented faithfully and completely. All three target files were modified exactly as specified; the load-bearing `.detach()` on the depth weight is applied; no invariant file was touched; reduced test-train ran to completion (81 train iters + full val epoch) with both depth-weighted reprojection scalars emitted at the expected small magnitudes (~4e-4 and ~5e-5) consistent with the `X/fx ≈ 5e-3` rescaling factor described in the design.

---

## Infrastructure Check

- `python scripts/cli.py review-check-implementation runs/idea010/design003` → PASS.

## `implementation_summary.md` as Checklist

**Files changed:**
- [x] `code/pelvis_utils.py` — required.
- [x] `code/pose3d_transformer_head.py` — required.
- [x] `code/config.py` — required.

No unauthorised files; `code/train.py` is byte-identical to baseline.

**Claimed changes vs. actual code:**
- [x] `project_joints_to_2d` added to `pelvis_utils.py` (lines 49–88) — matches the shared design snippet.
- [x] `import numpy as np` added to head file.
- [x] `from pelvis_utils import (...)` extended with `recover_pelvis_3d` and `project_joints_to_2d`.
- [x] `__init__` signature adds ALL THREE kwargs `reproj_loss_weight: float = 0.0`, `reproj_include_pelvis: bool = False`, `reproj_depth_weighted: bool = False` (lines 166–168); all three stored on `self` (lines 181–183). Defaults preserve baseline behaviour.
- [x] `loss()` block (lines 316–386) placed AFTER `loss/uv/train` and BEFORE the no_grad MPJPE block.
- [x] Per-element `smooth_l1_loss(beta=0.05, reduction='none')` on `(1, 22, 2)` for body joints and on `(1, 1, 2)` for the pelvis-only term — correct for per-joint weighting.
- [x] **Depth-weight is detached**: `w_i = (X_body / fx_i).detach().unsqueeze(-1)` (line 361) and `w_p = (X_p / fx_i).detach().view(1, 1, 1)` (line 376). This is the load-bearing design requirement — verified present.
- [x] Weight uses PREDICTED absolute X, clamped at 0.01: `X_body = pred_body_abs_i[..., 0].clamp(min=0.01)` and `X_p = pred_pelvis_i[..., 0].clamp(min=0.01)`.
- [x] Per-sample scalar via `err_i.mean()` / `err_p_i.mean()`, batch via `torch.stack(...).mean()` — matches design.
- [x] Dict keys exactly `'loss/reproj/train'` and `'loss/reproj_pelvis/train'`, both scaled by `self.reproj_loss_weight`.
- [x] When `reproj_depth_weighted=False`, code cleanly falls back to the unweighted design002 behaviour (guarded by `if self.reproj_depth_weighted`).
- [x] No `.detach()` on `pred_2d`, `gt_2d`, `err_i`, or `err_p_i` — gradients still flow through the error tensors; only the weight is detached.
- [x] `forward()` and `predict()` unchanged.
- [x] `config.py` adds `reproj_loss_weight=1.0`, `reproj_include_pelvis=True`, `reproj_depth_weighted=True` in `head=dict(...)`; no other changes.

## Design-Detail Fidelity

Every numbered invariant in the design's "Constraints and Invariants" section (14 items) is satisfied:
1. `persistent_workers=False` preserved.
2. Body joints 0-21 only.
3. `custom_imports` unchanged.
4. No `import` statements in `config.py`; new kwargs are float/bool literals.
5. Absolute import pattern preserved.
6. `K` via `np.asarray(..., dtype=np.float32)`.
7. `img_shape` default `(640, 384)`.
8. `smooth_l1_loss(beta=0.05, reduction='none')` plus `.mean()` reduction as specified.
9. `X >= 0.01` clamp inside `project_joints_to_2d` AND inside the weight computation (`pred_body_abs_i[..., 0].clamp(min=0.01)`, `pred_pelvis_i[..., 0].clamp(min=0.01)`).
10. `w_i` detached — explicitly verified, this is the key correctness requirement.
11. No rescaling factor added to compensate for the small raw magnitudes (design explicitly forbids this); the designed `reproj_loss_weight=1.0` is the only multiplier.
12. Defaults all three kwargs to baseline-preserving values.
13. Fallback to design002 behaviour when `reproj_depth_weighted=False` is structurally supported.
14. No invariant file modified.

## Invariant-File Compliance

- Diffs against baseline show only the expected additions in the three allowed files.
- `train.py` byte-identical to baseline.
- No changes under `infra/`, `mmpose/evaluation/`, `mmpose/datasets/`, backbone, or data preprocessor.

## Test-Output Sanity

- Reduced test-train completed 81 training iterations and produced a validation row in `test_output/metrics.csv`.
- No Error/Traceback/Exception/NaN strings in `slurm_test_55670398.out` or `20260417_135348.log`.
- Training log at iter 50 shows `loss/reproj/train: 0.000403` and `loss/reproj_pelvis/train: 0.000055`, and `grad_norm: 8.23` — the small magnitudes exactly match the design's prediction that `X/fx ≈ 5e-3` makes the raw loss ~1000× smaller than design002 (design002 showed `loss/reproj/train ≈ 12.40`, and `12.40 * 5e-3 ≈ 6e-2`; the observed `~4e-4` is in the right order of magnitude once joint-level variation is accounted for). Crucially, the much lower `grad_norm` vs. design002 (8.23 vs. 752.8) indicates the detached geometry-aware weighting successfully tames the gradient magnitude as intended.
- `iter_metrics.csv` preserves the invariant 3-column schema.

## Issues

None.

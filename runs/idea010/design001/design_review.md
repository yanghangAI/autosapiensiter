# Design Review — idea010/design001

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 001 (Auxiliary 2D Reprojection Loss on Body Joints, lambda=0.5) is complete, unambiguous, and implementation-ready. All required details are specified at the exact code level, and the design is confined to the three allowed files.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present and accurate.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to change: `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py` — only allowed files.
- [x] Exact algorithmic changes specified with full code snippets:
  - New `project_joints_to_2d(joints_abs, K, crop_h, crop_w, x_min=0.01)` in `pelvis_utils.py`, with explicit formula matching `recover_pelvis_3d` convention and `X`-clamp.
  - Import rewrite in head file (adds `recover_pelvis_3d`, `project_joints_to_2d`, `numpy as np`).
  - `__init__` signature addition: `reproj_loss_weight: float = 0.0` (default preserves baseline behaviour).
  - Full `loss()` insertion block, placed AFTER `losses['loss/uv/train']` and BEFORE the `with torch.no_grad()` MPJPE block, specified exactly.
- [x] Exact config values: `reproj_loss_weight=0.5` in head dict; all other values baseline-identical and tabulated.
- [x] Training, loss, data, inference changes: training-only loss augmentation; `predict()` unchanged; no dataset/transform/hook change.
- [x] Constraints and invariants section exhaustively enumerated (14 items), including: body joint indices 0-21, smooth_l1 with beta=0.05 reduction='mean', `X>=0.01` clamp, `[-1, 1]` normalisation matching `pelvis_uv`, no `.detach()` on differentiable tensors, key name `'loss/reproj/train'`.
- [x] Edge cases: small-X numerical guard, early-training instability handled via smooth_l1 bounded gradient, GT-side requires_grad=False acknowledged.

### Feasibility

- [x] `gt_joints`, `gt_depth`, `gt_uv`, and `pred` are all in scope at the insertion point (verified against baseline `loss()` at lines 276-304).
- [x] `pred['pelvis_depth']` shape `(B, 1)` and `pred['pelvis_uv']` shape `(B, 2)` match `recover_pelvis_3d` signature.
- [x] Per-sample K read pattern mirrors the existing `compute_mpjpe_abs` loop exactly (numpy float32 conversion, `img_shape` fallback `(640, 384)`).
- [x] `project_joints_to_2d` is purely differentiable torch ops; K/crop dims are python scalars so no device placement issue.
- [x] Projection formula is algebraically consistent with the `recover_pelvis_3d` inverse (verified: `u = fx*(-Y/X)+cx`, then normalise `2*u/W - 1`).
- [x] Negligible compute overhead on 1080 Ti (B=4 python loop).

### Invariant Compliance

- [x] No modifications to: `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, `tools/train.py`.
- [x] Loss still restricted to body joints (0-21); reprojection also body-only.
- [x] `persistent_workers=False` invariant preserved (no dataloader change).
- [x] No Python `import` statements in `config.py` (`reproj_loss_weight=0.5` is a float literal).
- [x] Head file uses absolute imports; extends existing `from pelvis_utils import ...` statement.
- [x] `custom_imports` list unchanged.
- [x] `predict()` unchanged; evaluation metric bit-identical for equivalent weights.

### Implementation Readiness

The Builder can implement this without guessing. Every changed line is specified, every constant is given, defaults are chosen to preserve baseline behaviour, and the loss insertion location is named relative to existing lines.

---

## Issues

None.

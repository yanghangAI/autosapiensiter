# Design Review — idea010/design002

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 002 (Reprojection Loss with Explicit Pelvis Term, lambda=1.0) is complete, unambiguous, and implementation-ready. It cleanly extends Design 001 with a second reprojection term on the pelvis and a stronger overall weight, fully specified at the code level.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present; explicitly states both body-joint and pelvis-only reprojection terms and the shared `reproj_loss_weight=1.0` gating.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to change: `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py` only.
- [x] Algorithm changes specified with full code snippets:
  - `pelvis_utils.py` helper `project_joints_to_2d` identical to design001 (fully specified).
  - Head `__init__` signature additions: `reproj_loss_weight: float = 0.0`, `reproj_include_pelvis: bool = False` (both defaults reproduce baseline).
  - Full `loss()` insertion block with both body-joint and pelvis-only reprojection terms, with correct shape annotations `(1, 22, 2)` / `(1, 1, 2)`.
  - Two distinct loss keys specified: `'loss/reproj/train'` and `'loss/reproj_pelvis/train'`.
  - Pelvis reprojection uses `pred_pelvis_i.unsqueeze(1)` to fit the `(B, J, 3)` interface — explicitly noted.
- [x] Exact config values: `reproj_loss_weight=1.0`, `reproj_include_pelvis=True`; all other values baseline-identical and tabulated.
- [x] Training/loss/inference changes: training-only; `forward()` and `predict()` unchanged.
- [x] Constraints and invariants section exhaustively enumerated (14 items): body-only indices 0-21, smooth_l1 beta=0.05 reduction='mean' for BOTH terms, same lambda for both, X-clamp, etc.
- [x] Edge cases: early-training instability acknowledged and mitigated via smooth_l1 bounded gradient and X-clamp; gradient through all three prediction heads preserved (no `.detach()`).

### Feasibility

- [x] All in-scope tensors (`gt_joints`, `gt_depth`, `gt_uv`, `pred`) are available at the insertion point (verified against baseline).
- [x] The pelvis reprojection loss, as the design itself notes, is partially redundant with `loss/uv/train` (the projection is an algebraic inverse of the unprojection up to the X-clamp). The design explicitly acknowledges this and justifies why the term is nevertheless non-redundant: (a) clamp gating, (b) independent lambda rebalancing. This is a reasonable and transparent design choice; the body-joint reprojection term is the primary coupling signal.
- [x] The body-joint reprojection term remains the substantive coupling mechanism — its gradient flows into `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']` non-trivially.
- [x] Per-sample K loop exactly mirrors existing code pattern in `compute_mpjpe_abs`.
- [x] Compute overhead: same order as design001 (B=4 python loop + one extra pelvis projection per sample).

### Invariant Compliance

- [x] No modifications to: evaluation metric, dataset, transforms, backbone, data preprocessor, `infra/*`, `train.py`, `tools/train.py`.
- [x] Loss still body-only (indices 0-21); pelvis term is on the pelvis alone, outside the body joint set — consistent with evaluation `mpjpe/pelvis/val` concept.
- [x] `persistent_workers=False` preserved.
- [x] No Python `import` statements in `config.py`; both new kwargs are bool/float literals.
- [x] Head file uses absolute imports; extension of existing pelvis_utils import line.
- [x] `custom_imports` unchanged.
- [x] `predict()` unchanged.

### Implementation Readiness

The Builder can implement this without guessing. Every line is specified, defaults are backward-compatible with baseline, and the dependency on design001's helper is explicit.

---

## Issues

None. The design acknowledges the potential loss redundancy with `loss/uv/train` and provides adequate justification; the primary coupling comes from the body-joint term.

# Design Review — idea010/design003

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 003 (Depth-Weighted Reprojection Loss, lambda=1.0) is complete, unambiguous, and implementation-ready. It extends Design 002 with a per-joint `w_i = X_i / fx` geometry-aware weight (detached to avoid gradient pathology). All required details are specified at the code level.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present; explicitly describes the `X/fx` weighting and its geometric rationale.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to change: `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py` only.
- [x] Algorithm changes specified with full code snippets:
  - `pelvis_utils.py` helper `project_joints_to_2d` identical to design001/002 (fully specified).
  - Head `__init__` signature additions: `reproj_loss_weight: float = 0.0`, `reproj_include_pelvis: bool = False`, `reproj_depth_weighted: bool = False` (all defaults preserve baseline).
  - Full `loss()` block using `smooth_l1_loss(..., reduction='none')` and per-sample `.mean()` + `torch.stack(...).mean()` reductions.
  - Depth weight computed as `(X / fx).detach().unsqueeze(-1)` for body joints and `(X / fx).detach().view(1, 1, 1)` for the pelvis term.
  - Explicit fallback: when `reproj_depth_weighted=False`, behaviour matches design002.
- [x] Exact config values: `reproj_loss_weight=1.0`, `reproj_include_pelvis=True`, `reproj_depth_weighted=True`; all other values baseline-identical and tabulated.
- [x] Training/loss/inference changes: training-only; `forward()` and `predict()` unchanged.
- [x] Constraints and invariants section exhaustively enumerated (14 items), including the load-bearing `.detach()` on `w_i` and the explicit note that the depth-weighted loss magnitude is expected to be numerically smaller (no extra rescale).
- [x] Edge cases: `X>=0.01` clamp applied both in `project_joints_to_2d` and in the `w_i` computation; near-camera joints (hands) explicitly noted as possibly under-weighted but acceptable since hand MPJPE is not in the composite.

### Feasibility

- [x] All in-scope tensors are available at the insertion point.
- [x] `.detach()` on the depth weight is correctly justified: attaching the weight to the graph would create a pathological incentive to predict smaller `X` to shrink the loss without improving accuracy. The design correctly identifies and addresses this.
- [x] The reduction scheme `smooth_l1(reduction='none')` → `.mean()` per-sample → `torch.stack(...).mean()` over the batch is mathematically equivalent to `reduction='mean'` in the unweighted case, so the lambda scale remains comparable to design002.
- [x] Shape correctness: `err_i` is `(1, 22, 2)`, `w_i` is `(1, 22, 1)` — broadcasts correctly. `err_p_i` is `(1, 1, 2)`, `w_p` is `(1, 1, 1)` — also broadcasts correctly.
- [x] `fx_i` is a python float from `float(K[0, 0])` — scalar broadcast safe.
- [x] Per-sample K loop mirrors `compute_mpjpe_abs`.

### Invariant Compliance

- [x] No modifications to: evaluation metric, dataset, transforms, backbone, data preprocessor, `infra/*`, `train.py`, `tools/train.py`.
- [x] Loss still body-only (indices 0-21).
- [x] `persistent_workers=False` preserved.
- [x] No Python `import` statements in `config.py`; all three new kwargs are float/bool literals.
- [x] Head file uses absolute imports; extension of existing pelvis_utils import line.
- [x] `custom_imports` unchanged.
- [x] `predict()` unchanged.

### Implementation Readiness

The Builder can implement this without guessing. Every tensor shape is annotated, every reduction axis is named, the detach placement is explicit, and defaults preserve baseline behaviour.

---

## Issues

None. The design correctly handles the gradient-pathology concern via `.detach()` on the depth weight, and the numerical-magnitude consequence is documented with an explicit instruction not to re-scale.

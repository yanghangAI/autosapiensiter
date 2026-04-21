# Design Review — idea011/design002

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 002 (Two-pass coordinate-conditioned decoder with shared weights and intermediate supervision, weight=0.5) is complete, unambiguous, and implementation-ready. The design is identical to Design 001 in architecture but enables the intermediate body-joint supervision branch of `loss()`. Only allowed files are modified.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present and accurate.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to change: `pose3d_transformer_head.py`, `config.py`. `pelvis_utils.py` marked no-change.
- [x] Exact algorithmic changes specified with full code snippets:
  - Same three `__init__` kwargs with identical signature placement and defaults as Design 001.
  - Identical `coord_enc` module construction and zero-init.
  - Identical `forward()` body.
  - `loss()` body includes intermediate supervision branch: `loss/joints_init/train = intermediate_supervision_weight * loss_joints_module(joints_initial[:, 0-21], gt_joints[:, 0-21])`. Uses shared `self.loss_joints_module`.
  - Pelvis supervision remains on pass-2 outputs only (rationale given: coord_enc maps only joint coordinates, not pelvis, and the pelvis head reads pass-2 token 0 only).
  - `predict()` explicitly unchanged.
- [x] Exact config values: `num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.5` in head dict.
- [x] Training/loss/inference changes specified: four loss scalars emitted (`loss/joints/train`, `loss/joints_init/train`, `loss/depth/train`, `loss/uv/train`), predict unchanged.
- [x] Constraints and invariants exhaustively enumerated (18 items), including exact loss key name `'loss/joints_init/train'`, same `_BODY = list(range(0, 22))` index set for both joint loss terms, reuse of `self.loss_joints_module`.
- [x] Edge cases: init-time equality `joints_final ≈ joints_initial` leading to ~1.5x effective body-joint loss weight at init is documented.

### Feasibility

- [x] `self.loss_joints_module` is stateless (SoftWeightSmoothL1Loss) and safely callable multiple times per iteration.
- [x] `pred['joints_initial']` is guaranteed present in the `num_refine_passes > 1` path (matches `forward()` contract).
- [x] Defensive `'joints_initial' in pred` guard ensures safe fallback if `num_refine_passes=1`.
- [x] Intermediate loss adds negligible compute (~< 1 ms on `(B, 22, 3)` tensor).
- [x] Gradient flow from `loss/joints_init/train` back through pass-1 decoder layer is intact (no `.detach()` in forward path).

### Invariant Compliance

- [x] No modifications to: `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, `tools/train.py`, `pelvis_utils.py`.
- [x] Loss still restricted to body joints 0-21 (both main and intermediate).
- [x] `persistent_workers=False` invariant preserved.
- [x] No Python `import` statements added to `config.py` (all three new kwargs are int/bool/float literals).
- [x] Head file uses absolute imports (unchanged).
- [x] `custom_imports` list unchanged.
- [x] `predict()` body unchanged; `joints_initial` is training-only and is not passed into `InstanceData`.
- [x] `MetricsCSVHook`, `TrainMPJPEAveragingHook`, `BedlamMPJPEMetric` untouched.

### Implementation Readiness

The Builder can implement this without guessing. The only delta from Design 001 is the `intermediate_supervision_weight=0.5` config value, which activates the already-specified branch of `loss()`. The rationale for single-sided (pass-2-only) pelvis supervision is clearly stated.

---

## Issues

None.

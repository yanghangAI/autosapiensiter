# Design Review — idea011/design001

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 001 (Two-pass coordinate-conditioned decoder with shared weights, no intermediate supervision) is complete, unambiguous, and implementation-ready. All required changes are specified at the code level and confined to the two allowed files (`pose3d_transformer_head.py`, `config.py`). `pelvis_utils.py` is explicitly left unchanged.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present and accurate.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to change: `pose3d_transformer_head.py`, `config.py` — only allowed files. `pelvis_utils.py` marked no-change.
- [x] Exact algorithmic changes specified with full code snippets:
  - Three new `__init__` kwargs with exact signature placement (after `loss_weight_uv`, before `init_cfg`) and defaults (`num_refine_passes=1, shared_decoder=True, intermediate_supervision_weight=0.0`) that preserve baseline.
  - `self.coord_enc = nn.Sequential(Linear(3, hidden_dim), GELU, Linear(hidden_dim, hidden_dim))` placement after `decoder_layer`, before `joints_out`.
  - Explicit zero-init on `coord_enc[2]` weight AND bias in `_init_head_weights`, with `trunc_normal_(std=0.02)` on `coord_enc[0]`.
  - Full `forward()` body with pass-1 → coord_enc → pass-2 → residual output logic and short-circuit for `num_refine_passes <= 1`.
  - Full `loss()` body with defensive `intermediate_supervision_weight > 0.0 and 'joints_initial' in pred` guard.
  - `predict()` explicitly stated unchanged.
- [x] Exact config values: `num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.0` appended to `head=dict(...)`; all other config values baseline-identical and tabulated.
- [x] Training/loss/inference changes specified: extra forward compute only, no new loss keys (intermediate supervision disabled), predict unchanged.
- [x] Constraints and invariants exhaustively enumerated (16 items), including: shared `joints_out`, no `.detach()` on differentiable tensors, residual formulation `joints_1 + joints_residual`, `num_refine_passes=1` baseline-equivalence, default kwargs preserve baseline.
- [x] Edge cases: gradient flow through `joints_1` documented; init-time baseline-match property called out explicitly.

### Feasibility

- [x] `decoder_layer` signature `(queries, spatial) -> queries` matches baseline and is safely callable twice on the same module.
- [x] `coord_enc` output shape `(B, 70, hidden_dim)` matches `decoded_1` for residual addition.
- [x] Residual sum `joints_1 + joints_residual` is a differentiable torch op; gradient flow back through both paths.
- [x] Extra parameter cost (~66.8K for `coord_enc`) is negligible vs. backbone.
- [x] Shape of `joints_out(decoded_2)` = `(B, 70, 3)` matches `joints_1` for residual add.
- [x] `pred['joints']` shape `(B, 70, 3)` unchanged → downstream `BedlamMPJPEMetric` and `TrainMPJPEAveragingHook` unaffected.

### Invariant Compliance

- [x] No modifications to: `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, `tools/train.py`, `pelvis_utils.py`.
- [x] Loss still restricted to body joints 0-21.
- [x] `persistent_workers=False` invariant preserved (no dataloader change).
- [x] No Python `import` statements added to `config.py` (`num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.0` are int/bool/float literals).
- [x] Head file uses absolute imports (unchanged).
- [x] `custom_imports` list unchanged.
- [x] `predict()` body unchanged; evaluation pipeline sees `(B, 70, 3)` refined joints tensor.
- [x] LR schedule, optimizer, data pipeline, seed, batch size, accumulation all unchanged.

### Implementation Readiness

The Builder can implement this without guessing. Every line of the new `forward()`, `loss()`, and `_init_head_weights()` is written out; module placement order is specified; kwarg defaults preserve baseline; the `num_refine_passes=1` short-circuit is explicitly required for baseline-equivalent behaviour.

---

## Issues

None.

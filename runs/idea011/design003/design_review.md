# Design Review — idea011/design003

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 003 (Two-pass coordinate-conditioned decoder with INDEPENDENT pass-2 decoder weights and intermediate supervision, weight=0.5) is complete, unambiguous, and implementation-ready. The design extends Design 002 by conditionally constructing `self.decoder_layer_2` when `shared_decoder=False` and routing pass-2 through it. Only allowed files are modified.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present and accurate.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to change: `pose3d_transformer_head.py`, `config.py`. `pelvis_utils.py` marked no-change.
- [x] Exact algorithmic changes specified with full code snippets:
  - Same three `__init__` kwargs with identical signature placement and defaults as Designs 001/002.
  - Identical `coord_enc` module construction and zero-init.
  - Conditional `self.decoder_layer_2 = _DecoderLayer(hidden_dim, num_heads, dropout)` built only when `not self.shared_decoder and self.num_refine_passes >= 2`; placement AFTER `self.decoder_layer` and `self.coord_enc`, BEFORE `self.joints_out`.
  - Explicit note that `self.decoder_layer_2` is NOT built when `shared_decoder=True`, avoiding unused params in the optimizer.
  - `_init_head_weights` unchanged from Designs 001/002 (decoder layer internals use PyTorch defaults, matching baseline `self.decoder_layer`).
  - `forward()` body identical in structure to Designs 001/002 with the existing `if self.shared_decoder: ... else: decoded_next = self.decoder_layer_2(...)` branch taking the `else` path.
  - `loss()` body identical to Design 002 (intermediate supervision weight=0.5 enabled).
  - `predict()` explicitly unchanged.
- [x] Exact config values: `num_refine_passes=2`, `shared_decoder=False`, `intermediate_supervision_weight=0.5` in head dict.
- [x] Training/loss/inference changes specified: four loss scalars, predict unchanged, `self.decoder_layer_2` called once per forward.
- [x] Constraints and invariants exhaustively enumerated (20 items), including: `_DecoderLayer` constructor args (`hidden_dim=256, num_heads=8, dropout=0.1`), shared `joints_out` across passes, no `.detach()`, residual formulation, optimizer `paramwise_cfg` unchanged (coord_enc and decoder_layer_2 fall under default head LR 1e-4), and note that `decoder_layer_2` MUST NOT exist when `shared_decoder=True`.
- [x] Edge cases: explicit discussion of init-time behavior — `decoded_2 != decoded_1` because `decoder_layer_2` has fresh random weights, but `joints_residual` stays small because of std=0.02 init on shared `joints_out`; intermediate supervision on pass-1 prevents collapse. Parameter-count and overfitting risks are called out.

### Feasibility

- [x] `_DecoderLayer` constructor signature `(embed_dim, num_heads, dropout)` verified against baseline (lines 80-83 of `pose3d_transformer_head.py`).
- [x] Added parameter cost ~1.2M — well within 1080 Ti budget.
- [x] Conditional build guard `if (not self.shared_decoder) and self.num_refine_passes >= 2` correctly covers the only config path that exercises `decoder_layer_2`; defaults (`shared_decoder=True`) skip the branch.
- [x] MMEngine config `shared_decoder=False` is a bool literal → no import needed.
- [x] `forward()` `else` branch uses `self.decoder_layer_2` only when `shared_decoder=False` — when Python reaches `else`, the attribute exists by construction.
- [x] Output shapes unchanged: `pred['joints']` stays `(B, 70, 3)`.

### Invariant Compliance

- [x] No modifications to: `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, `tools/train.py`, `pelvis_utils.py`.
- [x] Loss still restricted to body joints 0-21.
- [x] `persistent_workers=False` invariant preserved.
- [x] No Python `import` statements added to `config.py` (all three new kwargs are int/bool/float literals).
- [x] Head file uses absolute imports (unchanged).
- [x] `custom_imports` list unchanged.
- [x] Optimizer `paramwise_cfg.custom_keys` unchanged — only `'backbone': dict(lr_mult=0.1)`. New params fall under default head LR 1e-4 (consistent with baseline treatment of `self.decoder_layer`).
- [x] `predict()` body unchanged.
- [x] `MetricsCSVHook`, `TrainMPJPEAveragingHook`, `BedlamMPJPEMetric` untouched.

### Implementation Readiness

The Builder can implement this without guessing. Every construction, init, forward, and loss detail is spelled out; the conditional guard for building `decoder_layer_2` is explicit; default kwargs preserve baseline. The only additional delta vs. Design 002 is the conditional attribute build and the `shared_decoder=False` config value.

---

## Issues

None.

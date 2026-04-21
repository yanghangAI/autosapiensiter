# Design Review — idea009/design003

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-16

---

## Summary

Design 003 (Structured Spatial Token Dropout with Linear Annealing, p=0.30 → 0.10) is complete, unambiguous, and implementation-ready. All required components — the dropout mechanism, annealing hook, DDP-safe model access, and config changes — are fully specified with exact code.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present and accurate.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to change: `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` explicitly excluded.
- [x] Exact algorithmic changes specified for every modified location:
  - New imports: `from mmengine.registry import HOOKS` and `from mmengine.hooks import Hook` added at top of `pose3d_transformer_head.py`.
  - `_DecoderLayer.forward` signature and `key_padding_mask` logic (identical to designs 001/002).
  - `Pose3dTransformerHead.__init__`: new parameters `spatial_drop_prob_start=0.30`, `spatial_drop_prob_end=0.10`; `self.spatial_drop_prob` initialised to `spatial_drop_prob_start`.
  - `set_drop_prob(p)` method: exact signature and body given.
  - `Pose3dTransformerHead.forward`: decoder call updated to pass `spatial_drop_prob=self.spatial_drop_prob`.
  - `SpatialDropAnnealHook`: full class body with exact formula, epoch indexing (0-indexed `runner.epoch` converted to 1-indexed), and DDP-safe model access pattern.
- [x] Exact config values: `spatial_drop_prob_start=0.30`, `spatial_drop_prob_end=0.10` in head dict; `custom_hooks` extended with `SpatialDropAnnealHook(num_epochs=20, start_prob=0.30, end_prob=0.10)`.
- [x] Annealing schedule: per-epoch table provided and formula stated: `p = 0.30 + (0.10 - 0.30) * (epoch - 1) / 19`.
- [x] Hook registration: `@HOOKS.register_module()` decorator on `SpatialDropAnnealHook`; hook fires via `before_train_epoch`.
- [x] Import order and registration path: hook defined in `pose3d_transformer_head.py`, which is already in `custom_imports`, so registration happens before MMEngine processes `custom_hooks`.
- [x] Invariants: all listed — `persistent_workers=False`, body-only loss, `custom_imports` unchanged, no `import` in config, device placement, mask shape, fresh mask, no buffer, absolute imports, DDP pattern.

### Feasibility

- [x] `@HOOKS.register_module()` + `from mmengine.registry import HOOKS` is standard MMEngine hook registration. Works correctly when the module is imported via `custom_imports` before the config is parsed.
- [x] `before_train_epoch` is a valid MMEngine `Hook` method. Called before each epoch's training loop begins.
- [x] `runner.epoch` is confirmed 0-indexed in MMEngine; the design correctly converts to 1-indexed for the interpolation formula (epoch 0 → p=0.30, epoch 19 → p=0.10).
- [x] DDP defensive pattern (`if hasattr(model, 'module'): model = model.module`) is standard and handles both single-GPU and DDP cases.
- [x] `self.spatial_drop_prob = spatial_drop_prob_start` in `__init__` ensures epoch 1 starts at p=0.30 even if the hook fires at the same time — consistent behaviour confirmed in constraint 13.

### Invariant Compliance

- [x] No modifications to invariant files or components.
- [x] No Python `import` statements in `config.py`.
- [x] No relative imports in head file (new imports are absolute: `from mmengine.registry import HOOKS`, `from mmengine.hooks import Hook`).

### Implementation Readiness

The Builder can implement this without guessing. The full `SpatialDropAnnealHook` class body is given, the DDP access pattern is shown, the formula and epoch indexing are unambiguous, and all config changes are explicitly listed.

---

## Issues

None.

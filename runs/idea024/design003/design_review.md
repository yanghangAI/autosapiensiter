**Verdict: APPROVED**

**Design:** idea024/design003 — EMA per-joint difficulty weighting (alpha=1.0, linear, group-normalised + 5-epoch warmup)

---

## Review Summary

The design is fully specified, self-consistent, and implementable without guessing. All required sections are present. The anatomical mislabeling of upper/lower groups is explicitly acknowledged and resolved by the design itself.

### Completeness Check

| Section | Status |
|---|---|
| Design Description | Present |
| Starting point | `baseline/` — explicit |
| Files to change | `pose3d_transformer_head.py`, `config.py` only; `pelvis_utils.py` unchanged |
| Algorithmic changes | Fully described with exact formulas for both groups and warmup ramp |
| Config values and defaults | 6 literals explicitly listed |
| Training/loss changes | Joint loss replacement block with group-EMA update and warmup provided |
| Invariant preservation | Explicitly enumerated |
| Edge cases | Explicitly enumerated |

### Algorithm Correctness

- Two EMA buffers: `upper_err_ema` (13,) and `lower_err_ema` (9,) — registered when `group_normalise=True`.
- Group normalisation: within upper group, weights sum to 13; within lower, sum to 9; concatenated total = 22 — gradient scale preserved. ✓
- Warmup ramp: linear from 0 to 1 over `5 * 328 = 1640` iters. At iter=0: `ramp=0.0` → `w = uniform` (baseline). At iter=1640: `ramp=1.0` → full difficulty weights. ✓
- `ITERS_PER_EPOCH = 328` hardcoded as specified. ✓
- EMA update slices `per_joint_err[_UPPER_IDX]` and `per_joint_err[_LOWER_IDX]` — correct Python-list indexing on a 1D tensor. ✓

### Joint Group Definitions

The design names the groups "upper" and "lower" but the index assignments are:
- "Upper" = indices 0–12: includes pelvis, hips, knees, ankles, feet, spine, neck — mostly lower body anatomically.
- "Lower" = indices 13–21: includes collar, head, shoulders, elbows, wrists — mostly upper body anatomically.

This is an anatomical mislabeling, but the design explicitly addresses it: *"The implementation must use EXACTLY these index ranges: upper = indices 0..12 inclusive, count=13; lower = indices 13..21 inclusive, count=9."* The Builder must follow the index ranges, not anatomy. This is unambiguous. ✓

### Implementation Readiness

- Module-level `_UPPER_IDX = list(range(0, 13))` and `_LOWER_IDX = list(range(13, 22))` added after imports — accessible in `loss()` without `self.` prefix. ✓
- 7 new `__init__` params listed with exact defaults. Config omits `weight_temperature` — it uses the default of 1.0, which is correct since `weight_norm='linear'` (temperature is unused in the linear branch). ✓
- Buffer registration: conditional on both `per_joint_difficulty_weighting` and `group_normalise`. `_train_iter` registered in all `per_joint_difficulty_weighting=True` cases. ✓
- `_get_adaptive_weights`: includes both `group_normalise=True` path and `group_normalise=False` fallback (compatible with designs 1/2 if called with those params). ✓
- `loss()` block: group-specific EMA update inside `torch.no_grad()`, then `_get_adaptive_weights()`, then manual smooth-L1 with `beta=0.05` and `w.view(1, 22, 1)`. ✓

### Invariant Check

- `_train_mpjpe` / `_train_mpjpe_abs`, depth/uv losses, `predict()` — unchanged.
- `per_joint_difficulty_weighting=False` → bit-identical to baseline.
- No invariant files modified.

### Config Check

- 6 literal kwargs: `per_joint_difficulty_weighting=True`, `ema_alpha=1.0`, `ema_momentum=0.99`, `weight_norm='linear'`, `group_normalise=True`, `ema_warmup_epochs=5` — MMEngine-compliant, no import statements.
- `weight_temperature` correctly omitted from config; uses default 1.0 in `__init__`. ✓

### Minor Notes (non-blocking)

- The `_get_adaptive_weights` in design003 includes `import torch.nn.functional as F` inside the `else` branch (for the softmax fallback path). Since `group_normalise=True` for design003, this branch is never reached. The import is harmless dead code in this configuration.
- Stage-2 warmup restart: the design correctly explains that `_train_iter` resets to 0 for stage-2 (new model object, stage-1 checkpoint deleted). The warmup ramp restarts correctly. ✓

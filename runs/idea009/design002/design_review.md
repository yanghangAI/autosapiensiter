# Design Review — idea009/design002

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-16

---

## Summary

Design 002 (Moderate Spatial Token Dropout, p=0.30) is complete, unambiguous, and implementation-ready. It is mechanically identical to design001 with the single difference of `spatial_drop_prob=0.30`. All required details are specified.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present and accurate.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to change: `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` explicitly excluded.
- [x] Exact algorithmic changes specified with code snippets for every modified location (identical to design001):
  - `_DecoderLayer.forward` signature extension.
  - Cross-attention call replacement with `key_padding_mask` logic.
  - `Pose3dTransformerHead.__init__` new parameter.
  - `Pose3dTransformerHead.forward` updated decoder call.
- [x] Exact config values: `spatial_drop_prob=0.30` in head dict; all other values table-confirmed as baseline.
- [x] Invariants: `persistent_workers=False`, body-only loss (indices 0–21), `custom_imports` unchanged — all listed.
- [x] Edge cases and constraints: device placement, mask shape `(B, N_spatial)`, fresh mask per call, training-only gate, inference `key_padding_mask=None`, safe token count at p=0.30 (~672 visible). All specified.

### Feasibility

- [x] Identical mechanism to design001. Standard PyTorch `key_padding_mask` usage confirmed correct.
- [x] With p=0.30 and N=960: expected ~288 masked, ~672 visible. No risk of all-masked. Explicitly confirmed in constraints.
- [x] No memory concern on 1080 Ti.

### Invariant Compliance

- [x] No modifications to invariant files or components.
- [x] No Python `import` statements in `config.py`.
- [x] No relative imports in head file.

### Implementation Readiness

The Builder can implement this without guessing. The only difference from design001 is the `spatial_drop_prob` value (0.30 vs 0.15), which is explicitly stated.

---

## Issues

None.

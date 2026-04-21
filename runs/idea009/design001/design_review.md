# Design Review — idea009/design001

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-16

---

## Summary

Design 001 (Uniform Spatial Token Dropout, p=0.15) is complete, unambiguous, and implementation-ready. All required details are specified at the exact code level.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present and accurate.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to change: `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` explicitly excluded.
- [x] Exact algorithmic changes specified with code snippets for every modified location:
  - `_DecoderLayer.forward` signature extension (new `spatial_drop_prob: float = 0.0` kwarg).
  - Cross-attention call replacement with `key_padding_mask` logic.
  - `Pose3dTransformerHead.__init__` new parameter and storage as `self.spatial_drop_prob`.
  - `Pose3dTransformerHead.forward` decoder call updated to pass `spatial_drop_prob`.
- [x] Exact config values: `spatial_drop_prob=0.15` in head dict; all other values table-confirmed as baseline.
- [x] Invariants: `persistent_workers=False`, body-only loss (indices 0–21), `custom_imports` unchanged — all listed.
- [x] Edge cases and constraints: device placement, mask shape `(B, N_spatial)`, fresh mask per call (not a buffer), training-only gate, inference passes `key_padding_mask=None`. All specified.

### Feasibility

- [x] Mechanism is standard PyTorch: `nn.MultiheadAttention(batch_first=True)` accepts `key_padding_mask` of shape `(B, S)` with `True` = ignore. Correct usage.
- [x] No memory concern: mask is `(B, 960)` bool — negligible on 1080 Ti.
- [x] No interaction with baseline loss, dataset, or metric code.
- [x] The baseline `_DecoderLayer.forward` cross-attention call (`self.cross_attn(q, spatial_tokens, spatial_tokens)[0]`) matches the design's "before" snippet exactly.
- [x] The baseline `forward` decoder call (`decoded = self.decoder_layer(queries, spatial)`) matches the design's "before" snippet exactly.

### Invariant Compliance

- [x] No modifications to: `pelvis_utils.py`, `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, data preprocessor, `infra/` files, `train.py`.
- [x] No Python `import` statements in `config.py`.
- [x] No relative imports in head file.

### Implementation Readiness

The Builder can implement this without guessing. Every changed line is specified. No open design decisions remain.

---

## Issues

None.

**Verdict: APPROVED**

**Design:** idea004/design001 — Scalar depth per spatial token (linear projection)
**Reviewer:** Reviewer agent
**Date:** 2026-04-16

---

## Summary

Design 001 is feasible, complete, and explicit. The Builder can implement it without guessing on any point. All required sections are present and unambiguous.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present and clear.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to modify explicitly enumerated: `pose3d_transformer_head.py`, `config.py`. `pelvis_utils.py` explicitly excluded.
- [x] Exact algorithmic change: `spatial = input_proj(feat) + 2d_sincos_pos_enc + depth_proj(depth_grid)` with `depth_grid` shaped `(B, H'*W', 1)`, clamped to [0,10]/10.
- [x] New module: `nn.Linear(1, hidden_dim)` with explicit zero-init on both weight and bias.
- [x] Depth extraction: `_extract_depth_map` method specified in full, including NPZ vs NPY handling, `img_shape` crop convention, `F.interpolate` bilinear resize, and zero-fill fallback.
- [x] `forward` signature change: optional `depth_map: torch.Tensor | None = None` kwarg.
- [x] `loss()` and `predict()` changes: exact insertion point specified; rest of each method unchanged.
- [x] `config.py` change: exact head dict with `depth_pos_enc_type='linear'` shown in full.
- [x] All invariant config values confirmed identical to baseline (LR, batch, accumulation, warmup, seed, etc.).

### Constraints and Invariants

- [x] Loss restricted to body joints 0–21. Unchanged.
- [x] Pelvis pathway `decoded[:, 0, :]`. Unchanged.
- [x] `depth_proj` not added to `_init_head_weights` — explicitly stated.
- [x] Zero-init ensures functional equivalence to baseline at epoch 0.
- [x] `persistent_workers=False`. Unchanged.
- [x] Seed `2026`. Unchanged.
- [x] No Python `import` statements in `config.py`. `depth_pos_enc_type='linear'` is a string literal.
- [x] Invariant files not modified (no changes to backbone, dataset, transforms, metric, infra, train.py).

### No Issues Requiring Rejection

- The `depth_pos_enc_type` parameter is stored but no branching on it occurs in design001 — this is correct; the parameter documents the design variant and is used by config.py to distinguish designs.
- The `torch.Tensor | None` union syntax is safe because baseline already has `from __future__ import annotations`.
- The `img_shape` crop assumption (`raw[:ch, :cw]`) may not correctly crop the full-scene depth map if the person is not top-left-aligned. However, this is a correctness risk acknowledged in the idea spec, not an implementation ambiguity; the Builder can implement exactly what is written. The try/except fallback provides graceful degradation.
- The design correctly notes that if the pipeline caches the loaded depth array under a different metainfo key, the Builder should use that instead — this is stated explicitly as a note rather than ambiguity; the default path (load from `depth_npy_path`) is the specified implementation.

---

## No Changes Required

The design is approved as written. The Builder should implement exactly as specified.

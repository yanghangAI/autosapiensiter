**Verdict: APPROVED**

**Design:** idea004/design002 — Depth sinusoidal encoding (1D sin/cos + 2D pos enc, projected via Linear(384, 256))
**Reviewer:** Reviewer agent
**Date:** 2026-04-16

---

## Summary

Design 002 is feasible, complete, and explicit. The Builder can implement it without guessing. All required sections are present and unambiguous. The sinusoidal depth encoding approach is more complex than Design A but every step is fully specified.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present and clear.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to modify explicitly enumerated: `pose3d_transformer_head.py`, `config.py`. `pelvis_utils.py` explicitly excluded.
- [x] New module-level helper function `_build_1d_sincos_enc` specified in full with exact signature, docstring, frequency formula (`1.0 / (10000.0 ** omega)`), and output shape `(B, N, embed_dim)`.
- [x] Placement of `_build_1d_sincos_enc`: "after the existing `_build_2d_sincos_pos_enc` function" — explicit.
- [x] New module in `__init__`: `self.depth_pos_proj = nn.Linear(depth_enc_in_dim, hidden_dim)` where `depth_enc_in_dim = hidden_dim + hidden_dim // 2`. Explicit formula, not hardcoded 384.
- [x] Initialization: `trunc_normal_(std=0.02)` for weight, `zeros_` for bias. Rationale explained.
- [x] `_extract_depth_map` method: explicitly stated to be identical to Design A, full code provided.
- [x] `forward` changes: complete replacement block shown; unified code path for both depth and no-depth cases (both go through `depth_pos_proj`). Constraint that fallback must not skip the projection is explicitly stated.
- [x] `_get_pos_enc` preserved and called — `_build_2d_sincos_pos_enc` not modified.
- [x] `loss()` and `predict()` changes: exact insertion point and call pattern specified.
- [x] `config.py` change: full head dict shown with `depth_pos_enc_type='sinusoidal'`.
- [x] All other config values confirmed identical to baseline.

### Constraints and Invariants

- [x] `hidden_dim // 2` used (not hardcoded 128) — explicitly required in constraint #4.
- [x] `depth_enc_in_dim = hidden_dim + hidden_dim // 2` — not hardcoded 384 — explicitly required in constraint #5.
- [x] Fallback path always goes through `depth_pos_proj` (not a skip branch) — explicitly required in constraint #6.
- [x] `_get_pos_enc` and `_build_2d_sincos_pos_enc` are NOT modified — explicitly required in constraint #3.
- [x] Loss restricted to body joints 0–21. Unchanged.
- [x] Pelvis pathway `decoded[:, 0, :]`. Unchanged.
- [x] `persistent_workers=False`. Unchanged.
- [x] Seed `2026`. Unchanged.
- [x] No Python `import` statements in `config.py`.
- [x] Invariant files not modified.

### No Issues Requiring Rejection

- The `trunc_normal_(std=0.02)` init for `depth_pos_proj` means the model does NOT start from a functional equivalent of baseline (unlike Design A's strict zero-init). The design explicitly acknowledges this: "The projection of `(2d_sincos || 0)` ... gives a noisy but reasonable starting positional signal." This is a design choice, not an ambiguity.
- The unified fallback code path (zero-padding depth dimension and projecting through `depth_pos_proj`) is unusual but fully specified and justified.
- Same `img_shape` crop assumption and double I/O considerations as Design A — acknowledged in idea spec, not an implementation ambiguity.

---

## No Changes Required

The design is approved as written. The Builder should implement exactly as specified, particularly: (1) use `hidden_dim // 2` not literal 128; (2) use `hidden_dim + hidden_dim // 2` not literal 384; (3) always run the fallback path through `depth_pos_proj` rather than skipping it.

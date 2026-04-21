**Verdict: APPROVED**

**Design:** idea004/design003 — Depth+2D MLP positional encoding (3-input learned MLP replacing fixed sinusoidal pos enc)
**Reviewer:** Reviewer agent
**Date:** 2026-04-16

---

## Summary

Design 003 is feasible, complete, and explicit. The Builder can implement it without guessing. All required sections are present. The MLP replacement of the fixed 2D sinusoidal positional encoding is fully specified, including the critical constraint that `_get_pos_enc` must NOT be called in `forward`.

---

## Checklist

### Completeness

- [x] `**Design Description:**` present and clear.
- [x] Starting point explicitly stated: `baseline/`.
- [x] Files to modify explicitly enumerated: `pose3d_transformer_head.py`, `config.py`. `pelvis_utils.py` explicitly excluded.
- [x] New module in `__init__`: `self.pos_mlp = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, hidden_dim))` — architecture fully specified.
- [x] `pos_mlp_hidden = 64`: hardcoded local variable in `__init__`, explicitly not exposed as a config kwarg.
- [x] Initialization loop: `for layer in self.pos_mlp: if isinstance(layer, nn.Linear): trunc_normal_(std=0.02), zeros_bias`. Explicit.
- [x] `_extract_depth_map`: stated as identical to Design A, full code provided.
- [x] New method `_build_3d_pos_grid`: full specification including (x, y, depth) ordering, `torch.linspace(-1,1,h/w)` for spatial axes, `clamp([0,10])/10` for depth, fallback `depth=0.5` (not 0.0), and return shape `(1, h*w, 3)` when depth_map is None.
- [x] `forward` changes: complete replacement block; `pos_grid.expand(B,-1,-1)` when depth_map is None; `pos_embed = self.pos_mlp(pos_grid)`; `spatial = spatial + pos_embed`.
- [x] Critical: `_get_pos_enc` NOT called in forward — explicitly stated as constraint #3 and #9.
- [x] `_build_2d_sincos_pos_enc` and `_get_pos_enc` remain in the file but are unused in forward — explicitly stated.
- [x] `loss()` and `predict()` changes: identical pattern to designs A and B, exact insertion point specified.
- [x] `config.py` change: full head dict with `depth_pos_enc_type='mlp'`.
- [x] All other config values confirmed identical to baseline.

### Constraints and Invariants

- [x] `pos_mlp_hidden = 64` must not be changed — explicitly required in constraint #4.
- [x] x/y normalisation `[-1, 1]` via `torch.linspace` — explicitly required in constraint #5.
- [x] Depth normalisation `[0, 1]` via `clamp([0,10])/10` — explicitly required in constraint #6.
- [x] Fallback depth = `0.5` (not 0.0) — explicitly required in constraint #7.
- [x] `_build_3d_pos_grid` returns `(1, h*w, 3)` for None depth; `forward` must expand — explicitly required in constraint #8.
- [x] No `spatial = spatial + pos_enc` line in forward — explicitly required in constraint #9.
- [x] Loss restricted to body joints 0–21. Unchanged.
- [x] Pelvis pathway `decoded[:, 0, :]`. Unchanged.
- [x] `persistent_workers=False`. Unchanged.
- [x] Seed `2026`. Unchanged.
- [x] No Python `import` statements in `config.py`.
- [x] Invariant files not modified.

### No Issues Requiring Rejection

- The near-zero initialization of `pos_mlp` means early training lacks a meaningful positional signal. The design acknowledges this convergence risk explicitly and notes the 3-epoch warmup partially mitigates it.
- The `depth_pos_enc_type` parameter stored but no branching — correct for this standalone design variant.
- The `_build_3d_pos_grid` returns B=1 when depth_map is None (a `full((1, h*w), 0.5, ...)`). The `forward` call `pos_grid.expand(B, -1, -1)` correctly broadcast to the actual batch size. This is unambiguous.
- Same `img_shape` crop and I/O concerns as other designs — acknowledged in idea spec, not an implementation ambiguity.

### High-Risk Design Note (for Orchestrator/Builder awareness)

Design 003 replaces the fixed sinusoidal positional encoding entirely. Training may underperform baseline in early epochs while `pos_mlp` learns a spatial layout. The design spec explicitly documents this risk. The Builder should not "fix" the design by re-adding the sinusoidal encoding — constraint #9 forbids it.

---

## No Changes Required

The design is approved as written. Key points for the Builder: (1) do NOT call `_get_pos_enc` or add `spatial + pos_enc` in forward; (2) fallback depth is 0.5, not 0.0; (3) `pos_mlp_hidden = 64` is hardcoded; (4) expand `pos_grid` to batch size when `depth_map is None`.

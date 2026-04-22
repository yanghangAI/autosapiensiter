## Design Review — idea027/design003

**Verdict: APPROVED**

---

### Feasibility

Two-layer stack with `num_layers=2`. The `_SpatialContextNet` class loop already handles this via `is_last = (i == num_layers - 1)`: layer 0 (`i=0`, `is_last=False`) gets `trunc_normal_(std=0.02)` on pointwise; layer 1 (`i=1`, `is_last=True`) gets `zeros_` on pointwise. The zero-init guarantee holds: `pw_1.weight = 0`, `pw_1.bias = 0` → net output = 0 for any input → delta = 0 → spatial unchanged at init. All modules are standard PyTorch and AMP-safe.

---

### Completeness and Explicitness

All required fields are present and unambiguous:

- **Design Description:** present and accurate.
- **Starting point:** `baseline/` — explicit.
- **Files changed:** `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` explicitly excluded.
- **Algorithmic change:** fully specified. The `_SpatialContextNet` class is provided verbatim (identical to design001/design002). The two-layer stack structure is itemized explicitly:
  1. `dw_0`: Conv2d(256,256,3,pad=1,groups=256,bias=False), kaiming_normal
  2. `GroupNorm(32,256)` — layer 0
  3. `GELU()` — layer 0
  4. `pw_0`: Conv2d(256,256,1,bias=True), **trunc_normal(0.02)** weight, zeros bias
  5. `dw_1`: Conv2d(256,256,3,pad=1,groups=256,bias=False), kaiming_normal
  6. `GroupNorm(32,256)` — layer 1
  7. `GELU()` — layer 1
  8. `pw_1`: Conv2d(256,256,1,bias=True), **zeros** weight and bias
- **No per-layer residual:** explicitly stated. Single outer residual `spatial + delta` only.
- **Insertion points:** identical to design001/design002.
- **Config values:** all six kwargs listed as literals (`True`, `3`, `2`, `'groupnorm'`, `32`, `'gelu'`). Complete.
- **Zero-init strategy for two-layer:** clearly differentiated — `zero_init_last=True` zeroes only layer 1 pointwise; layer 0 pointwise uses trunc_normal. Mechanically correct via `is_last` flag.

---

### Zero-Init Guarantee for Two-Layer Case

The guarantee is sound because:
- `pw_1.weight = 0`, `pw_1.bias = 0`
- `nn.Sequential` feeds the full output of layers 1–4 into layers 5–8
- Layer 8 (`pw_1`) applies a linear transform with zero weight and zero bias to any input → output is identically zero regardless of what layers 1–7 produced
- `delta = net(x) = 0` → `spatial + delta = spatial`

This is mechanically correct. The design explicitly states "the entire sequential output is 0" and explains the mechanism. No ambiguity.

---

### Invariants Audit

1. Zero-init guarantee: holds via zero weight+bias on `pw_1`. Correct.
2. No per-layer residual inside sequential: explicitly stated.
3. Shape invariant: `(B, H*W, hidden_dim)` preserved.
4. `H, W` from `feat.shape`. Correct.
5. GroupNorm divisibility: 256/32=8. Satisfied.
6. Config constraint: all literals. Satisfied.
7. Loss/output interfaces: unchanged.
8. `persistent_workers=False`: not touched.
9. AMP safety: all modules safe.

---

### No Invariant Violations

No invariant files touched.

---

### Minor Notes (non-blocking)

- The `zero_init_last=True` call-site hardcoding is correct and consistent: it zeroes only the last pointwise regardless of `num_layers`.
- The effective receptive field calculation (5×5 from two chained 3×3 convolutions) is correct.
- The design explicitly notes that the residual is at the `_SpatialContextNet` level, not per-layer. This is critical for two-layer correctness and is clearly documented.

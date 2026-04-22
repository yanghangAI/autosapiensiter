## Design Review — idea027/design002

**Verdict: APPROVED**

---

### Feasibility

Identical architecture to design001 with `norm='groupnorm'`. `nn.GroupNorm(32, 256)` is valid: 256 channels divisible by 32 groups (8 channels per group). All modules are standard PyTorch and AMP-safe. No new dependencies.

---

### Completeness and Explicitness

All required fields are present and unambiguous:

- **Design Description:** present and accurate.
- **Starting point:** `baseline/` — explicit.
- **Files changed:** `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` explicitly excluded.
- **Algorithmic change:** fully specified. The `_SpatialContextNet` class is provided verbatim (identical to design001, which already handles `norm='groupnorm'` via the conditional `layers.append(nn.GroupNorm(num_groups, hidden_dim))`). Layer ordering is: dw → GroupNorm(32, 256) → GELU → pw.
- **Insertion points:** identical to design001 — "after `self.loss_weight_uv`" and "after `self.decoder_layer = _DecoderLayer(...)`" in `__init__`; "after `spatial = spatial + pos_enc`" in `forward()`.
- **Config values:** all six kwargs listed as literals (`True`, `3`, `1`, `'groupnorm'`, `32`, `'gelu'`). Complete.
- **Init strategy:** zero-init on pointwise (weight and bias), kaiming_normal on depthwise. GroupNorm gamma=1, beta=0 (PyTorch default, correctly noted).
- **GroupNorm at init with zero-init downstream:** design explicitly addresses that the GroupNorm→GELU→pointwise(zeros) path outputs zero at init regardless of GroupNorm normalization. Mechanically correct.

---

### Invariants Audit

1. Zero-init guarantee: single layer, `is_last=True` at `i=0`, zeros on pw. Delta=0 at init. Correct.
2. Shape invariant: preserved.
3. `H, W` from `feat.shape`. Correct.
4. Config constraint: all literals. Satisfied.
5. GroupNorm divisibility: 256/32=8. Satisfied.
6. Loss/output interfaces: unchanged.
7. `persistent_workers=False`: not touched.
8. AMP safety: `Conv2d`, `GroupNorm`, `GELU` — all safe.

---

### No Invariant Violations

Same as design001 — no invariant files touched.

---

### Minor Notes (non-blocking)

- The class definition is explicitly stated to be identical to design001, referencing design001 for the full code. This is clear enough for the Builder.
- GroupNorm behavior section correctly explains that gamma=1/beta=0 defaults mean the GroupNorm path starts learning from the first gradient step (once depthwise filters become non-trivial), but the pointwise zero-init still collapses the output to zero at init. No issue.

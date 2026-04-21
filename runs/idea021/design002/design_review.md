# Design Review — idea021/design002

**Verdict: APPROVED**

---

## Summary

Design002 adds a factored cross-attention bias `u_i[h] + v_i[w]` (zero-initialized) with row parameter `(70, 40)` and col parameter `(70, 24)`. The design is complete, explicit, and implementable without guessing.

---

## Checklist

### Completeness and Explicitness

- **Design Description:** Present and clear. ✓
- **Starting point:** `baseline/` — explicit. ✓
- **Files to change:** `pose3d_transformer_head.py` and `config.py` only — both allowed. No `pelvis_utils.py` change. ✓
- **Algorithmic change:** Factored outer sum `u_i.unsqueeze(-1) + v_i.unsqueeze(-2)` → `(70,40,24)` → `.view(70, 960)` → `attn_mask`. Fully specified. ✓
- **Parameter shapes:** `cross_attn_bias_row (70, 40)` and `cross_attn_bias_col (70, 24)` — explicit. ✓
- **Initialization:** Both `torch.zeros(...)` — zero, exact baseline equivalence at epoch 0. ✓
- **New kwargs with defaults:** Same as design001 — all defaulted. ✓
- **`_init_head_weights()` unchanged:** No warm-start logic needed. ✓
- **AMP compatibility:** `.to(q.dtype)` cast in `_DecoderLayer.forward()` — same as design001. ✓
- **Config values:** `use_cross_attn_bias=True`, `cross_attn_bias_type='factored'`, `feat_h=40`, `feat_w=24` — all literals, MMEngine compliant. ✓
- **feat_h=40, feat_w=24:** Consistent with design001's verified resolution. ✓
- **Broadcasting correctness:** `(70,40,1) + (70,1,24) → (70,40,24)` — correct. Entry `[i,h,w] = u_i[h] + v_i[w]`. Flattened row-major matches `feat.flatten(2).transpose(1,2)` token ordering. ✓
- **`joint_row_prior` kwarg ignored:** Correctly — warm-start logic is gated on `cross_attn_bias_type == 'factored_warmstart'`. ✓
- **Invariants preserved:** Body joint loss indices 0–21 unchanged; `pelvis_token` unchanged; `persistent_workers=False` unchanged; no invariant files modified. ✓

### No Invariant File Modifications

No changes to evaluation metric, dataset, transforms, backbone, data preprocessor, or infra files. ✓

### Implementation Readiness

All method signatures, parameter names, shapes, broadcasting operations, and config key-value pairs are fully specified. The Builder can implement this exactly from the design without guessing.

---

## Notes

None. Design is approved as written.

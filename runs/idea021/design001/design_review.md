# Design Review — idea021/design001

**Verdict: APPROVED**

---

## Summary

Design001 adds a full `(70, 960)` learnable cross-attention logit bias (zero-initialized) via `attn_mask` to `nn.MultiheadAttention` in `_DecoderLayer`. The design is complete, explicit, and implementable without guessing.

---

## Checklist

### Completeness and Explicitness

- **Design Description:** Present and clear. ✓
- **Starting point:** `baseline/` — explicit. ✓
- **Files to change:** `pose3d_transformer_head.py` and `config.py` only — both are allowed experimentable files. `pelvis_utils.py` explicitly untouched. ✓
- **Algorithmic change:** Exactly specified. `attn_mask=cross_attn_bias.to(q.dtype)` passed to `self.cross_attn(...)` in `_DecoderLayer.forward()`. ✓
- **Parameter shape:** `(num_joints=70, feat_h * feat_w=960)` — explicit. ✓
- **Initialization:** `torch.zeros(...)` — zero, ensuring exact baseline equivalence at epoch 0. ✓
- **New kwargs with defaults:** `use_cross_attn_bias=False`, `cross_attn_bias_type='full'`, `feat_h=40`, `feat_w=24`, `joint_row_prior=None` — all defaulted for backward compatibility. ✓
- **AMP compatibility:** `.to(q.dtype)` cast on bias before passing to attention. ✓
- **Config values:** All bool/str/int literals — MMEngine compliant, no import statements. ✓
- **feat_h=40, feat_w=24:** Correctly resolved the ambiguity flagged in idea.md (640/16=40, 384/16=24). ✓
- **`attn_mask` shape semantics:** `(tgt_len=70, src_len=960)` regardless of `batch_first=True` — confirmed correct. ✓
- **`bias.view(self.num_joints, -1)`:** Relies on `feat_h * feat_w = 960`. Verified: 40×24=960. ✓
- **Forward routing block:** Exact replacement for `decoded = self.decoder_layer(queries, spatial)` specified with full if/else. ✓
- **Invariants preserved:** Body joint loss indices 0–21 unchanged; `pelvis_token = decoded[:, 0, :]` unchanged; `persistent_workers=False` unchanged; no modification to invariant files. ✓

### No Invariant File Modifications

No changes to `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, data preprocessor, `infra/` files, or `train.py`. ✓

### Implementation Readiness

The Builder can implement this without guessing. All method signatures, code snippets, parameter names, shapes, and config key-value pairs are fully specified. The ordering of operations (parameter allocation before `_init_head_weights()` call) is consistent with the baseline `__init__` structure.

---

## Notes

None. Design is approved as written.

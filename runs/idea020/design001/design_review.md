# Design Review — idea020/design001

**Verdict: APPROVED**

---

## Checklist

### Feasibility
- The mechanism is straightforward: add a module-level `_temp_scaled_attn` helper that replicates `nn.MultiheadAttention`'s forward with an inserted per-query temperature divide before softmax.
- Uses `mha_module.in_proj_weight`, `in_proj_bias`, `out_proj` — all standard attributes of `nn.MultiheadAttention` when `_qkv_same_embed_dim=True`. The design mandates an assertion for this guard.
- AMP safety is explicitly addressed: `.to(attn.dtype)` cast on `tau` before division. This is mandatory and the design states it.
- `tau.clamp(min=0.1)` prevents logit overflow. Sufficient for Design 001.

### Completeness
- Starting point: `baseline/` — explicit.
- Files changed: `pose3d_transformer_head.py` and `config.py` only. `pelvis_utils.py` explicitly not modified.
- Full function body of `_temp_scaled_attn` with exact signature provided.
- Exact changes to `_DecoderLayer.__init__` (new `cross_temp`, `self_temp` params, stored as attrs) specified.
- Exact before/after for `_DecoderLayer.forward` cross-attention block specified.
- Exact new kwargs for `Pose3dTransformerHead.__init__` with defaults specified.
- Parameter creation logic for `use_cross_temp=True, temp_log_space=False` path specified.
- Decoder layer construction replacement (before/after) specified.
- `_qkv_same_embed_dim` assertion placement specified.
- `loss()` change: explicitly no change needed (`temp_reg_weight=0.0`).
- `_init_head_weights()` change: explicitly no change needed.
- Config kwargs: exact block with literal values provided.

### Explicitness
- Insertion point for `_temp_scaled_attn`: "after `_build_2d_sincos_pos_enc`, before `class _DecoderLayer`" — unambiguous.
- `nn.Parameter` reference passing: explicitly described; Builder told not to copy the tensor.
- Dropout: uses `mha_module.dropout` float attr — explicitly called out.
- Temperature shape: `(num_joints,) = (70,)` — explicit.
- Init value: `torch.ones(num_joints)` → tau=1.0 at init → identical to baseline. Explicit.
- Self-attention: explicitly unchanged for Design 001.

### Invariants preserved
- Body-only joint loss `_BODY = list(range(0, 22))`: explicitly flagged as must-preserve.
- Pelvis token at index 0: explicitly flagged.
- Backward compatibility via defaults: all four new kwargs have defaults matching the off/disabled state.
- `persistent_workers=False`: explicitly flagged as unchanged.
- No MMEngine config imports: all config values are bool/float literals.
- Invariant files: no modifications to `pelvis_utils.py`, `bedlam_metric.py`, dataset, transforms, backbone, infra, `train.py`.

### Issues
None. Design is implementation-ready.

---

**APPROVED**

# Design Review — idea020/design002

**Verdict: APPROVED**

---

## Checklist

### Feasibility
- Core mechanism identical to Design 001 (`_temp_scaled_attn` helper), with the parameterisation difference: `log_cross_temp = nn.Parameter(torch.zeros(num_joints))`, and `tau = F.softplus(log_cross_temp)` computed at forward time.
- `softplus(0) ≈ 0.693` — slightly sharper attention than baseline at init. Acceptable and explicitly stated.
- Because `softplus` output is a dynamic tensor (not a static `nn.Parameter`), the design correctly identifies that it cannot be pre-stored in `_DecoderLayer` at init time. The recommended solution — add `cross_temp_override` argument to `_DecoderLayer.forward()` — is specified precisely and completely.
- The `_temp_scaled_attn` function for Design 002 uses `clamp(min=1e-6)` (instead of `min=0.1`), which is appropriate since `softplus` already guarantees strict positivity. The design explicitly instructs the Builder not to remove the clamp.
- L2 regularisation: `temp_reg_weight * log_cross_temp.pow(2).mean()` added to `loss()` under key `'loss/temp_reg/train'`.

### Completeness
- Starting point: `baseline/` — explicit.
- Files changed: `pose3d_transformer_head.py` and `config.py` only.
- `_temp_scaled_attn` function body provided in full with exact clamp value (`min=1e-6`).
- `_DecoderLayer.__init__` changes: identical to Design 001, fully specified.
- `_DecoderLayer.forward` changes: the `cross_temp_override` argument addition is fully specified with exact signature and conditional logic.
- `Pose3dTransformerHead.__init__` changes: log-space branch (`temp_log_space=True`) creates `self.log_cross_temp = nn.Parameter(torch.zeros(num_joints))` and passes `cross_temp=None` to decoder layer. This is fully specified.
- `Pose3dTransformerHead.forward()` change: compute `softplus(self.log_cross_temp)` and pass as `cross_temp_override` — fully specified.
- `loss()` L2 reg term: exact key name, condition, and formula specified.
- Config kwargs: `temp_log_space=True`, `temp_reg_weight=0.01` — exact block provided.
- `_qkv_same_embed_dim` assertion retained.
- `_init_head_weights`: no change needed — `torch.zeros` init for `log_cross_temp` is correct.

### Explicitness
- The two-path distinction (Design 001 `temp_log_space=False` vs Design 002 `temp_log_space=True`) in the same head class is clearly described and the Builder is explicitly warned about the name difference (`self.log_cross_temp` vs `self.cross_temp`).
- The `cross_temp_override=None` default in `_DecoderLayer.forward()` is explicitly required for backward compatibility.
- The `softplus` import path: `torch.nn.functional.softplus` — no new import needed. Explicitly stated.
- Loss key `'loss/temp_reg/train'` naming convention explained and enforced.
- `hasattr(self, 'log_cross_temp')` guard in `loss()` ensures no reg loss for non-log-space cases. Explicit.

### Invariants preserved
- Body-only joint loss: explicitly flagged.
- Pelvis token at index 0: explicitly flagged.
- Backward compatibility via defaults.
- `persistent_workers=False`: unchanged.
- No MMEngine config imports.
- No modifications to invariant files.

### Issues
None. The design handles the dynamic-tensor problem (softplus output cannot be stored as a static reference in _DecoderLayer) with a clean solution (`cross_temp_override` argument) and specifies it unambiguously.

---

**APPROVED**

# Design Review — idea020/design003

**Verdict: APPROVED**

---

## Checklist

### Feasibility
- Extends Design 001 by applying the same `_temp_scaled_attn` function to self-attention as well as cross-attention.
- Self-attention call: `_temp_scaled_attn(self.self_attn, q, q, q, self.self_temp, training=self.training)` — the function handles Ns=Nq without any structural issue since the attention matrix is `(B, Nh, Nq, Nq)` and the temperature `(Nq,)` is reshaped to `(1,1,Nq,1)`, applied per query row. Explicitly verified in the design.
- Two assertions added: one for `cross_attn._qkv_same_embed_dim`, one for `self_attn._qkv_same_embed_dim`. Both are True for standard `nn.MultiheadAttention` construction in baseline. Explicit and correct.
- `temp_log_space=False` for Design 003 — direct parameterisation with clamp, same as Design 001. Simpler implementation path (no `cross_temp_override` complication from Design 002).
- Total: 140 new scalar parameters (negligible).

### Completeness
- Starting point: `baseline/` — explicit.
- Files changed: `pose3d_transformer_head.py` and `config.py` only.
- `_temp_scaled_attn` function body: identical to Design 001, full body provided.
- `_DecoderLayer.__init__` changes: same as Designs 001/002 — fully specified.
- `_DecoderLayer.forward` changes: both self-attention AND cross-attention blocks replaced, with exact before/after provided.
- `Pose3dTransformerHead.__init__`: `use_self_temp=True` path creates `self.self_temp = nn.Parameter(torch.ones(num_joints))` and passes `self_temp_param` to decoder layer. Both parameters passed to `_DecoderLayer`. Fully specified.
- Both assertions present.
- `loss()`: no change (temp_reg_weight=0.0).
- `_init_head_weights`: no change needed.
- Config kwargs: `use_self_temp=True`, exact block provided with all four kwargs as literals.

### Explicitness
- Self-attention temperature dimension analysis: `B, Nq, D = query.shape` when called with `q, q, q` extracts `Nq = num_joints` correctly since `q = self.norm1(queries)` is `(B, num_joints, D)`. Explicitly verified.
- Both `nn.Parameter` references passed by reference (not copied) — explicitly flagged.
- Dropout in `_temp_scaled_attn` for self-attention calls uses `self.self_attn.dropout` — consistent with the function interface (`mha_module.dropout`). Explicit.
- Two assertions: Builder is told both are required.

### Invariants preserved
- Body-only joint loss: explicitly flagged.
- Pelvis token at index 0: explicitly flagged.
- Backward compatibility via defaults.
- `persistent_workers=False`: unchanged.
- No MMEngine config imports.
- No modifications to invariant files.

### Issues
None. Design 003 is a clean extension of Design 001 with all additional specification needed for the self-attention temperature path.

---

**APPROVED**

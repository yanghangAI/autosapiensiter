# Design Review — idea006/design003

**Verdict: APPROVED**

**Reviewer:** Reviewer agent
**Date:** 2026-04-16

---

## Checklist

### Feasibility
PASS. Per-head bias requires reading batch size `B` from `queries.shape[0]` inside `forward`, expanding `(num_heads, J, J)` to `(B * num_heads, J, J)`, and passing as `attn_mask`. All standard PyTorch operations. No new dependencies.

### Completeness
PASS. All required details are specified:
- Starting point: `baseline/` — explicit.
- Files to change: `pose3d_transformer_head.py` and `config.py` — explicit.
- Exact algorithmic change: `(8, 70, 70)` zero-initialized `nn.Parameter`; expand to `(B*8, 70, 70)` via `.unsqueeze(0).expand(B,-1,-1,-1).reshape(B*num_heads,J,J)`.
- `attn_bias_mode` string values: `'per_head'`, `'shared'`, `'none'` — all defined.
- `self.num_heads` stored as attribute — specified.
- Config change: `attn_bias_type='per_head'` — specified as string literal.
- 39200 new parameters documented.

### Explicitness
PASS. Exact code snippets for every change:
- New `_DecoderLayer.__init__` signature with `attn_bias_mode: str = 'none'`.
- Conditional parameter registration for all three modes.
- Exact expand/reshape sequence step-by-step.
- `Pose3dTransformerHead.__init__` changes with `attn_bias_type` → `attn_bias_mode` mapping explicitly noted.
- Full config.py head dict replacement provided.
- Builder has zero ambiguity.

### Expand/Reshape Correctness
PASS. `.unsqueeze(0)` → `(1, H, J, J)`; `.expand(B, -1, -1, -1)` → `(B, H, J, J)`; `.reshape(B*H, J, J)` → `(B*H, J, J)`. Constraint #9 correctly analyzes the batch-head interleaving for `batch_first=True`: the resulting layout `[b0h0, b0h1, ..., b0h(H-1), b1h0, ...]` matches PyTorch's internal expected order for per-head `attn_mask`. Correct.

### Contiguity
PASS. Constraint #3 explicitly requires `.contiguous().reshape(...)` or equivalent `.reshape()` (which handles non-contiguous tensors). Addressed.

### `self.attn_bias = None` for mode `'none'`
PASS. When `attn_bias_mode='none'`, `self.attn_bias` is set to `None` (not `nn.Parameter`), so it does not appear in `self.parameters()`. Correctly specified in constraint #6.

### `B` from `queries.shape[0]`
PASS. Constraint #2 explicitly states `B` must be read from `queries.shape[0]` inside `forward`, not passed as argument. The forward signature is unchanged. Correct.

### Invariant Compliance
PASS. Only `pose3d_transformer_head.py` and `config.py` change. `config.py` change is `attn_bias_type='per_head'` — a string literal, no imports. No invariant files touched.

### Naming Mismatch (attn_bias_type → attn_bias_mode)
PASS. The design explicitly notes and explains the naming difference: `attn_bias_type` is the MMEngine config key in `Pose3dTransformerHead`; `attn_bias_mode` is the internal `_DecoderLayer` API parameter. The mapping `attn_bias_mode=attn_bias_type` is shown in the replacement code. No ambiguity.

---

## Notes
- Design 003's `_DecoderLayer` also includes `'shared'` mode handling (single `(J,J)` bias). This is defensive inclusion, not strictly required for Design C, but does not violate any constraint. The Builder should implement it as written.
- No `_build_skeleton_attn_bias` helper is needed or present in this design — correct, as only zero-init is used.

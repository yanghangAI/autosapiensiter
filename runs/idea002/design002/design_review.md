**Verdict: APPROVED**

**Reviewer:** Reviewer agent
**Date:** 2026-04-16
**Design:** idea002/design002 — Decoupled pelvis query, independent decoder layer

---

## Review Summary

All required sections are present and fully specified. The design is unambiguous and implementation-ready.

---

## Checklist

### Feasibility
- PASS. Two new constructor parameters (`decouple_pelvis: bool = False`, `pelvis_decoder_type: str = 'shared'`) are straightforward.
- PASS. `_DecoderLayer(hidden_dim, num_heads, dropout)` as `self.pelvis_decoder` is a valid instantiation of the existing class.
- PASS. The cross-attention-only sub-call uses `self.pelvis_decoder.norm2`, `self.pelvis_decoder.cross_attn`, `self.pelvis_decoder.dropout2`, `self.pelvis_decoder.norm3`, `self.pelvis_decoder.ffn` — all confirmed as valid `_DecoderLayer` attributes in baseline.
- PASS. Unused `self_attn`, `norm1`, `dropout1` modules in `pelvis_decoder` are registered but not called — this is safe in PyTorch (they will get zero gradients but cause no errors).

### Completeness
- PASS. Starting point (`baseline/`) is explicitly stated.
- PASS. Files to change: `pose3d_transformer_head.py` and `config.py`. `pelvis_utils.py` explicitly excluded.
- PASS. Module creation is conditional and precisely specified (`if self.decouple_pelvis: ... if self.pelvis_decoder_type == 'independent': ...`).
- PASS. Weight init section specified (pelvis_query trunc_normal; pelvis_decoder Linear layers rely on PyTorch default xavier uniform — explicitly noted as not needing custom init).
- PASS. Exact forward code block and replacement are given verbatim.
- PASS. Config changes (`decouple_pelvis=True`, `pelvis_decoder_type='independent'`) given with all baseline parameters confirmed unchanged.
- PASS. Invariants explicitly enumerated.

### Explicitness / No Guessing Required
- PASS. The pelvis decoder uses its own weights (not shared with `decoder_layer`) — stated explicitly in critical implementation notes.
- PASS. The fallback (`decouple_pelvis=False` or `pelvis_decoder_type != 'independent'`) resolves to `decoded[:, 0, :]` — baseline behaviour.
- PASS. `joints` pathway completely unchanged — stated explicitly.

### Invariant Compliance
- PASS. No changes to invariant files.
- PASS. No Python `import` statements in `config.py`.
- PASS. Loss restriction, `predict()` keys, seed, persistent_workers all preserved.

### Edge Cases
- PASS. Single-token self-attention no-op explained and skipped explicitly.
- PASS. Unused modules (`self_attn`, `norm1`, `dropout1`) will not receive gradients but are harmless — noted.
- PASS. Fallback path is explicit.

### Minor Note (non-blocking)
- When `decouple_pelvis=True, pelvis_decoder_type='shared'`, the code creates `pelvis_query` but not `pelvis_decoder`, and the forward falls back to `decoded[:, 0, :]` (baseline). This internal inconsistency is irrelevant for design002 (which uses `pelvis_decoder_type='independent'`), but would matter if someone mixed configs. Since each design is a standalone config, this has no impact on this design's implementation.

---

## Issues

None.

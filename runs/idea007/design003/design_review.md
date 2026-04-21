# Design Review — idea007 / design003

**Verdict: APPROVED**

---

## Checklist

### Feasibility
PASS. The per-head bias approach is technically sound. PyTorch `nn.MultiheadAttention` with `batch_first=True` accepts `attn_mask` of shape `(B*num_heads, T_q, T_k)` for per-sample per-head masking — this is documented PyTorch behaviour. The `B` dimension is already available in `Pose3dTransformerHead.forward` via `B, C, H, W = feat.shape`. The `.expand().reshape()` pattern creates a non-copying view of the bias, which is correct.

### Completeness
PASS. All required sections are present:
- Design Description: stated clearly.
- Starting point: `baseline/` — confirmed.
- Files to change: `pose3d_transformer_head.py` and `config.py` only.
- Exact algorithmic changes: per-head bias shape `(num_heads, J, S)`, expansion to `(B*num_heads, J, S)`, `B` propagation from head forward, `_per_head` branching flag, `_num_heads` stored attribute.
- Exact config values: `num_spatial=960`, `cross_routing_type='per_head'` as plain literals.
- Training/loss/data/inference changes: none.
- Constraints and edge cases: all 12 constraints listed explicitly.

### Explicitness
PASS. The Builder is given:
- Final unified `_DecoderLayer.__init__` signature (the "Revised approach" block supersedes the initial partial signature — this is clearly indicated in the design with the "Revised approach" heading).
- Exact branching logic in `__init__` for `per_head_routing=True` vs `False`.
- Exact bias expansion code: `.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self._num_heads, ...)`.
- Exact `_DecoderLayer.forward` signature change: `B: int = 1` default.
- Exact branching in `forward` on `self._per_head`.
- `Pose3dTransformerHead.__init__` mapping: `_per_head = (cross_routing_type == 'per_head')`, then passed as `per_head_routing=_per_head` to `_DecoderLayer`.
- Exact `Pose3dTransformerHead.forward` change: `decoded = self.decoder_layer(queries, spatial, B=B)`.
- Complete config.py head dict.
- Explicit note that `B=1` default must not be relied on in production — `Pose3dTransformerHead.forward` must always pass `B=B` explicitly.

### Implementation Readiness
PASS. No ambiguity requiring guessing:
- The initial partial `_DecoderLayer.__init__` signature shown in section 1a is superseded by the "Revised approach" final signature — the design is internally consistent because the "Revised approach" explicitly replaces the earlier attempt and provides the complete final form.
- `_per_head` and `_num_heads` stored as instance attributes — stated.
- `expand` not `repeat` — stated with rationale.
- Shape of expanded bias `(B*8, 70, 960)` — stated in parameter table.
- Backward compatibility (`cross_routing_type='none'` → `_per_head=False`, zero init, `(70, 960)` shape) — stated in constraint 7.
- `_init_head_weights` exclusion — stated in constraint 8.

### Invariant Compliance
PASS. No invariant files touched. Config uses only string and integer literals.

### Cross-Design Consistency
PASS. design003 starts from `baseline/` independently. The "Revised approach" in section 1c introduces the combined `_DecoderLayer` signature with both `cross_attn_bias_init` (from design002) and `per_head_routing` — this is a standalone implementation that subsumes all three routing modes in one file, which is the right approach for a design starting from baseline.

---

## Notes
- The "Revised approach" section in 1c contains the authoritative final `_DecoderLayer.__init__` signature. The earlier partial signature in section 1a (without `cross_attn_bias_init` or `per_head_routing`) is superseded by it. Builder must use the Revised approach signature.
- When `per_head_routing=True`, `cross_attn_bias_init` is hardcoded to `'zero'` in the construction call — zero init only for Design C, as stated. The `cross_attn_bias_init` kwarg is accepted but only used in the `per_head_routing=False` branch; this is consistent.
- The assert `spatial_tokens.shape[1] == self.cross_attn_bias.shape[-1]` checks the last dimension — for both `(J, S)` and `(num_heads, J, S)` shapes, `shape[-1]` is `S = num_spatial`. This is correct.
- The `B=1` default in `_DecoderLayer.forward` is a safety default only. Since Design C always passes `B` explicitly from the head, the default will never be used in production. Builder must add a comment to this effect as noted in constraint 12.

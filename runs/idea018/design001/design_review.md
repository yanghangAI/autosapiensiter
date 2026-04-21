# Design Review — idea018/design001

**Verdict: APPROVED**

---

## Review Summary

Design 001 proposes a fixed-sigma (sigma=1.0) Gaussian depth gate on cross-attention logits via two zero-initialized linear probes. The design is complete, unambiguous, and implementation-ready.

---

## Checklist

### Feasibility
- The mechanism (additive logit bias to `nn.MultiheadAttention` via `attn_mask` with float dtype) is a standard and supported PyTorch API usage. The shape arithmetic `(B, N_spatial) → (B*num_heads, Nq, N_spatial)` is explicitly detailed and correct.
- Zero-init of both probes guarantees flat gate at step 0, recovering baseline behavior exactly. Safe training start.
- AMP compatibility is addressed: `attn_logit_bias` values are ≤ 0 (bounded negative), which is numerically safe under float16.
- The `depth_gate_sigma_buf` is correctly registered as a buffer (non-optimized), not a parameter.

### Completeness
- Starting point: `baseline/` — specified.
- Files changed: only `pose3d_transformer_head.py` and `config.py` — within the allowed set.
- `pelvis_utils.py`: explicitly no changes.
- All four modification points in `pose3d_transformer_head.py` are specified: `_DecoderLayer.forward()`, `__init__()`, `_init_head_weights()`, and `forward()`.
- `loss()`: explicitly no changes for Design 001 — correct (no auxiliary loss).
- Config additions: `depth_gate_type='gaussian'`, `depth_gate_sigma=1.0` — str/float literals, compliant with MMEngine no-imports constraint.

### Explicitness
- Exact new constructor kwargs and defaults are given.
- Exact code snippets for all modification points are provided.
- The `attn_mask` shape requirement `(B*num_heads, Nq, N_spatial)` is explicitly called out along with the required float dtype.
- The note that `_depth_probe_z_hat` caching is NOT needed for Design 001 (constraint 9) is explicit — removes ambiguity for the Builder.
- Output tensor shapes are confirmed unchanged: `joints (B, 70, 3)`, `pelvis_depth (B, 1)`, `pelvis_uv (B, 2)`.

### Invariant Compliance
- Invariant files not touched: backbone, dataset, transforms, metric, data preprocessor, train.py, pelvis_utils.py, infra files.
- `persistent_workers=False` preserved.
- MMEngine config no-import constraint satisfied.

### Issues Found
- None. The design is self-consistent with the baseline code and the idea.md specification.

---

## Notes for Builder

1. In `_DecoderLayer.forward()`, the `q` variable is the normalized `queries` (after `self.norm2`). The shape variable `B_nq` mentioned in idea.md is not used in Design 001's code snippet (the design uses `q.shape[1]` for `Nq` directly) — the design's own code snippet is clear and correct; use Design 001's snippet, not idea.md's.
2. Do not add `self._depth_probe_z_hat` caching — explicitly prohibited by constraint 9.
3. The `depth_gate_sigma_buf` buffer will appear in `state_dict` — this is correct and expected.

# Code Review — idea006 / design003

**Verdict: APPROVED**

**Reviewer:** Reviewer agent
**Date:** 2026-04-16

---

## review-check-implementation

PASSED.

---

## Files Changed Check

`implementation_summary.md` lists two files: `code/pose3d_transformer_head.py` and `code/config.py`. Both are required by the design. Confirmed correct.

---

## Fidelity to Design

### `_DecoderLayer.__init__`
- Signature: `def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_joints=70, attn_bias_mode: str = 'none')`: PRESENT.
- `self.num_heads = num_heads` stored as attribute: PRESENT.
- `self.attn_bias_mode = attn_bias_mode` stored as attribute: PRESENT.
- `if attn_bias_mode == 'per_head': self.attn_bias = nn.Parameter(torch.zeros(num_heads, num_joints, num_joints))`: PRESENT. Shape `(8, 70, 70)` confirmed via defaults.
- `elif attn_bias_mode == 'shared': self.attn_bias = nn.Parameter(torch.zeros(num_joints, num_joints))`: PRESENT.
- `else: self.attn_bias = None`: PRESENT. Not an `nn.Parameter` in this path — correct.

### `_DecoderLayer.forward`
- `if self.attn_bias_mode == 'per_head':` block: PRESENT.
- `B = queries.shape[0]`: PRESENT — read from `queries.shape[0]`, not passed as argument.
- Expand sequence: `.unsqueeze(0).expand(B, -1, -1, -1).contiguous().reshape(B * self.num_heads, queries.shape[1], queries.shape[1])`: PRESENT. `.contiguous()` called before `.reshape()` — satisfies constraint 3.
- `q2 = self.self_attn(q, q, q, attn_mask=_mask)[0]`: PRESENT.
- `elif self.attn_bias_mode == 'shared': q2 = self.self_attn(q, q, q, attn_mask=self.attn_bias)[0]`: PRESENT.
- `else: q2 = self.self_attn(q, q, q)[0]`: PRESENT.

### `Pose3dTransformerHead.__init__`
- `attn_bias_type: str = 'none'` parameter added: PRESENT.
- `_DecoderLayer(hidden_dim, num_heads, dropout, num_joints=num_joints, attn_bias_mode=attn_bias_type)`: PRESENT. Naming mapping `attn_bias_type` (config key) → `attn_bias_mode` (`_DecoderLayer` internal) confirmed.

### `config.py`
- `attn_bias_type='per_head'` added as string literal in the head dict: PRESENT.
- No Python import statements added: confirmed.

### Constraints
1. Zero-init: `torch.zeros(num_heads, num_joints, num_joints)` → confirmed.
2. `B` read from `queries.shape[0]`, not passed as argument: confirmed.
3. `.contiguous().reshape(...)` used (not `.view()`): confirmed.
4. `attn_mask` dtype: `nn.Parameter` (float32), matches model dtype: confirmed.
5. String values `'per_head'`, `'shared'`, `'none'` correctly dispatched: confirmed. `attn_bias_type='per_head'` maps to `attn_bias_mode='per_head'`: confirmed.
6. `self.attn_bias = None` for mode `'none'`: confirmed — not an `nn.Parameter`, excluded from `parameters()`.
7. No changes to loss, data pipeline, backbone, `pelvis_utils.py`, or invariant files: confirmed.
8. No Python imports in `config.py`: confirmed.
9. `batch_first=True` and `(B * num_heads, T, T)` expand/reshape interleaving: code uses `.unsqueeze(0).expand(B,-1,-1,-1).contiguous().reshape(B*num_heads,J,J)` which produces layout `[b0h0,b0h1,...,b0h7,b1h0,...]` — matches PyTorch's expected layout for per-head `attn_mask`: confirmed correct.

---

## Invariant File Check

- `pelvis_utils.py`: identical to baseline (no diff).
- `train.py`: identical to baseline.
- `config.py`: only `output_dir` and `attn_bias_type='per_head'` differ from baseline; no invariant components touched.

---

## Test Output

- Training completed without errors.
- Epoch 1 val metrics produced: composite/val=490.96, mpjpe/body/val=443.13, mpjpe/pelvis/val=588.07.
- `metrics.csv` populated correctly with all required CSV columns.
- No runtime errors or abnormal output observed.
- Loss values and grad_norm are in reasonable range (loss ~1.854, grad_norm ~8.2).
- Memory usage (10612 MB) is slightly higher than design001 (10611 MB), consistent with the larger `(8,70,70)` parameter — expected.

---

## Summary

All changes in `pose3d_transformer_head.py` and `config.py` precisely match the design spec. Per-head bias shape `(8,70,70)` is zero-initialized, `B` read from `queries.shape[0]`, `.contiguous().reshape()` used for safe expansion, `attn_bias=None` for mode `'none'`, correct naming mapping in `Pose3dTransformerHead`. Test run completed cleanly with valid metric output.

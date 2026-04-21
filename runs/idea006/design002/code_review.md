# Code Review — idea006 / design002

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

### `_build_skeleton_attn_bias` (module-level function)
- Placed at module level, before `_DecoderLayer`, after imports: PRESENT.
- Signature: `_build_skeleton_attn_bias(num_joints=70, adjacent_val=0.5, pelvis_diag_val=-0.5)`: PRESENT.
- All 21 body edges, 23 left-hand edges, 23 right-hand edges, 2 face edges hardcoded: PRESENT. Edge list is identical to the design spec.
- Bidirectionality: `bias[i,j] = bias[j,i] = adjacent_val` for all edges: PRESENT.
- `bias[0, 0] = pelvis_diag_val`: PRESENT.
- No external graph library imports: confirmed (only `torch` used).
- Returns `bias` (not an `nn.Parameter`): correct — wrapping happens in caller.

### `_DecoderLayer.__init__`
- Signature: `def __init__(self, embed_dim, num_heads=8, dropout=0.1, num_joints=70, attn_bias_init: torch.Tensor | None = None)`: PRESENT.
- `if attn_bias_init is not None: self.attn_bias = nn.Parameter(attn_bias_init.float().clone())`: PRESENT with `.float().clone()` as required.
- `else: self.attn_bias = nn.Parameter(torch.zeros(num_joints, num_joints))`: PRESENT.

### `_DecoderLayer.forward`
- `q2 = self.self_attn(q, q, q, attn_mask=self.attn_bias)[0]`: PRESENT.

### `Pose3dTransformerHead.__init__`
- `attn_bias_type: str = 'none'` parameter added: PRESENT.
- `if attn_bias_type == 'skeleton_init': _bias_init = _build_skeleton_attn_bias(num_joints, adjacent_val=0.5, pelvis_diag_val=-0.5)`: PRESENT.
- `elif attn_bias_type == 'zero_init': _bias_init = torch.zeros(num_joints, num_joints)`: PRESENT.
- `else: _bias_init = None`: PRESENT.
- `_DecoderLayer(hidden_dim, num_heads, dropout, num_joints=num_joints, attn_bias_init=_bias_init)`: PRESENT.
- Fallback path (`attn_bias_type='none'` → `_bias_init=None`) produces `torch.zeros`-initialized parameter in `_DecoderLayer`: confirmed correct.

### `config.py`
- `attn_bias_type='skeleton_init'` added as string literal in the head dict: PRESENT.
- No Python import statements added: confirmed.

### Constraints
1. `_build_skeleton_attn_bias` at module level, before `_DecoderLayer`: confirmed.
2. Bidirectionality for all edges: confirmed.
3. No external imports: confirmed.
4. `'none'` fallback to `attn_bias_init=None` → zeros: confirmed.
5. `attn_bias_type='skeleton_init'` is a string literal: confirmed.
6. `.float().clone()` called before `nn.Parameter`: confirmed.
7. `attn_mask` additive semantics, shape `(70,70)`: confirmed.
8. No changes to loss, data pipeline, backbone, `pelvis_utils.py`, or invariant files: confirmed.
9. All 69 kinematic edge pairs indices remain within [0, 69]: confirmed by inspection of edge list.

---

## Invariant File Check

- `pelvis_utils.py`: identical to baseline (no diff).
- `train.py`: identical to baseline.
- `config.py`: only `output_dir` and `attn_bias_type='skeleton_init'` differ from baseline; no invariant components touched.

---

## Test Output

- Training completed without errors.
- Epoch 1 val metrics produced: composite/val=490.65, mpjpe/body/val=442.97, mpjpe/pelvis/val=587.46.
- `metrics.csv` populated correctly with all required CSV columns.
- No runtime errors or abnormal output observed.
- Loss values and grad_norm are in reasonable range (loss ~1.856, grad_norm ~8.2).

---

## Summary

All changes in `pose3d_transformer_head.py` and `config.py` precisely match the design spec. The `_build_skeleton_attn_bias` function matches the design's edge list exactly, bidirectionality and pelvis diagonal are correct, `.float().clone()` applied, fallback paths implemented correctly. Test run completed cleanly with valid metric output.

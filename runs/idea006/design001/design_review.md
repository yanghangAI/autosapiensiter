# Design Review — idea006/design001

**Verdict: APPROVED**

**Reviewer:** Reviewer agent
**Date:** 2026-04-16

---

## Checklist

### Feasibility
PASS. The change is minimal: one new `nn.Parameter` in `_DecoderLayer.__init__`, one modified call in `_DecoderLayer.forward`, and one updated construction call in `Pose3dTransformerHead.__init__`. No new dependencies. No new imports required.

### Completeness
PASS. All required details are specified:
- Starting point: `baseline/` — explicit.
- Files to change: `pose3d_transformer_head.py` only — explicit. `config.py` correctly noted as requiring no changes.
- Exact algorithmic change: zero-initialized `(70, 70)` `nn.Parameter` added as `attn_mask` to `self.self_attn(q, q, q)[0]`.
- Exact parameter values: shape `(70, 70)`, `torch.zeros`, 4900 parameters.
- Constraints: zero-init invariant, additive semantics, no other init changes.

### Explicitness
PASS. Exact current signatures and exact replacement code are provided. Line numbers match the actual baseline file (line 80 for `_DecoderLayer.__init__`, line 113 for `q2 = self.self_attn(q, q, q)[0]`, line 185 for `self.decoder_layer`). Builder has no ambiguity.

### Implementation Readiness
PASS. The design provides exact code snippets for every required change. The `num_joints=70` is passed to `_DecoderLayer` from `Pose3dTransformerHead.__init__` which already holds `self.num_joints`. Correct.

### Invariant Compliance
PASS. Only `pose3d_transformer_head.py` changes. No invariant files touched. No Python imports added to `config.py`.

### attn_mask Semantics
PASS. PyTorch `nn.MultiheadAttention` with `attn_mask` of shape `(T_q, T_k)` = `(70, 70)` broadcasts additively to attention logits before softmax across batch and heads when `batch_first=True`. Baseline confirms `batch_first=True` (line 83). Correct.

---

## Notes
- `config.py` requires no change — the `attn_bias` parameter is always registered in this design (unconditional). This is a valid design choice and correctly stated.
- The design explicitly instructs the Builder NOT to initialize `attn_bias` in `_init_head_weights`, which is correct.

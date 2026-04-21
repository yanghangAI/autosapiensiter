# Design Review — idea007 / design001

**Verdict: APPROVED**

---

## Checklist

### Feasibility
PASS. The change is minimal: one new `nn.Parameter`, one modified forward call, one assert. PyTorch `nn.MultiheadAttention` with `batch_first=True` accepts `attn_mask` of shape `(T_q, T_k) = (num_joints, num_spatial)` and broadcasts over batch and head dimensions — this is correct and documented behaviour since PyTorch 1.9.

### Completeness
PASS. All required sections are present:
- Design Description: stated clearly.
- Starting point: `baseline/` — confirmed.
- Files to change: `pose3d_transformer_head.py` and `config.py` only — both are experimentable files.
- Exact algorithmic changes: new `nn.Parameter`, assert in forward, `attn_mask` pass-through.
- Exact config values: `num_spatial=960` as plain integer literal.
- Training/loss/data/inference changes: none (correct).
- Constraints and edge cases: all 10 constraints listed explicitly.

### Explicitness
PASS. The Builder is given:
- Exact new `_DecoderLayer.__init__` signature with default values.
- Exact location in `__init__` where the parameter is registered (after `self.dropout2`).
- Exact replacement code block for the cross-attention section in `forward`.
- Exact new `Pose3dTransformerHead.__init__` signature.
- Exact location and code for decoder layer construction.
- Explicit exclusion of `cross_attn_bias` from `_init_head_weights`.

### Implementation Readiness
PASS. No ambiguity requiring guessing:
- Shape `(70, 960)` is fully derived and stated.
- Zero-init via `nn.Parameter(torch.zeros(num_joints, num_spatial))` is stated.
- `attn_mask` semantics (additive, positive = stronger, no `key_padding_mask`, no `is_causal`) stated.
- `batch_first=True` compatibility confirmed.
- `num_spatial` propagation path from config → `Pose3dTransformerHead.__init__` → `_DecoderLayer.__init__` fully specified.

### Invariant Compliance
PASS. No invariant files touched. No changes to `pelvis_utils.py`, evaluation metric, dataset, transforms, backbone, data preprocessor, or training infrastructure. Config changes use only plain literals.

### Cross-Design Consistency
PASS. design001 starts from `baseline/` independently. The `_DecoderLayer` signature introduced here (`num_joints`, `num_spatial` kwargs) is a compatible subset of design002's signature. Since all three designs start from `baseline/`, there is no dependency issue.

---

## Notes
- The assert `spatial_tokens.shape[1] == self.cross_attn_bias.shape[-1]` must be placed before the cross-attn call, not after — design spec correctly shows it before.
- The design explicitly notes that `self.num_spatial = num_spatial` must be stored on the head. Builder must not forget this line.
- Design 001 does NOT add `cross_attn_bias_init` kwarg to `_DecoderLayer` (that is introduced in design002). Since both start from `baseline/`, this is correct and creates no conflict.

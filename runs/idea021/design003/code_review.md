# Code Review — idea021/design003

**Verdict: APPROVED**

## Automated Check
`python scripts/cli.py review-check-implementation runs/idea021/design003` — PASSED.

## Files Changed vs. Design Specification
`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design. PRESENT.
- `code/config.py` — required by design. PRESENT.

No files outside the allowed set were modified. `pelvis_utils.py` and `train.py` are byte-for-byte identical to baseline.

## Code vs. Design Fidelity

### `_DecoderLayer.forward()`
- Identical to design001 and design002. CONFIRMED by diff.

### `Pose3dTransformerHead.__init__()`
- Same kwargs and factored parameter allocation as design002. PRESENT.
- `self.joint_row_prior = joint_row_prior` stored as instance attribute. PRESENT (required for warm-start access in `_init_head_weights()`).

### `_init_head_weights()` — Gaussian warm-start (design003 key change)
- Guard condition: `self.use_cross_attn_bias and self.cross_attn_bias_type == 'factored_warmstart' and self.joint_row_prior is not None`. PRESENT and correct.
- `h_coords = torch.arange(self.feat_h, dtype=torch.float32)` — `(40,)`, matching `feat_h=40`. CORRECT.
- `sigma = 4.0`, `alpha = 1.0`. MATCHES design specification exactly.
- Loop: `for i, mu in enumerate(self.joint_row_prior[:22])` — sliced to exactly 22 body joints. MATCHES design.
- `gauss = alpha * torch.exp(-(h_coords - mu) ** 2 / (2.0 * sigma ** 2))` — Gaussian formula. MATCHES design exactly.
- `self.cross_attn_bias_row.data[i] = gauss` — direct `.data` assignment for initialization. CORRECT.
- Hand joints (22–69) remain zero-initialized. CORRECT.
- Column biases (`cross_attn_bias_col`) remain zero throughout. CORRECT.

### `forward()`
- Identical factored bias computation as design002. CONFIRMED by diff.

### `config.py`
- Head dict contains: `use_cross_attn_bias=True`, `cross_attn_bias_type='factored_warmstart'`, `feat_h=40`, `feat_w=24`. PRESENT.
- `joint_row_prior=[12.0, 10.0, 14.0, 12.0, 9.0, 15.0, 7.0, 19.0, 21.0, 5.0, 3.0, 2.0, 11.0, 13.0, 11.0, 13.0, 9.0, 9.0, 15.0, 15.0, 12.0, 12.0]` — exactly 22 float entries. MATCHES design specification value-for-value.
- All values are bool/str/int/float/list literals. No import statements. MMEngine-compliant.

## Invariant Preservation
- All invariants identical to design001 and design002. CONFIRMED.

## Test Output
- Job completed: `[test] Finished.` with no errors.
- 72 iterations of epoch 1 logged with finite losses (joints ~0.19–0.22, depth ~1.4–3.7, uv ~0.05–0.3).
- `grad_norm: 8.516533` at iter 50 — well within normal range; clipped by `max_norm=1.0`.
- Loss values at epoch 1 are consistent with warm-started biases (slightly different from design002 as expected, due to non-zero initial routing).
- Checkpoint saved at epoch 1. Training completed normally.

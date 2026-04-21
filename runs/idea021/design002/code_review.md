# Code Review — idea021/design002

**Verdict: APPROVED**

## Automated Check
`python scripts/cli.py review-check-implementation runs/idea021/design002` — PASSED.

## Files Changed vs. Design Specification
`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design. PRESENT.
- `code/config.py` — required by design. PRESENT.

No files outside the allowed set were modified. `pelvis_utils.py` and `train.py` are byte-for-byte identical to baseline.

## Code vs. Design Fidelity

### `_DecoderLayer.forward()`
- Identical to design001 (design002 specifies "identical to design001"). CONFIRMED by diff: only comment lines differ.

### `Pose3dTransformerHead.__init__()`
- Same kwargs as design001. PRESENT.
- Parameter allocation for `cross_attn_bias_type='factored'`: `self.cross_attn_bias_row = nn.Parameter(torch.zeros(num_joints, feat_h))` → shape `(70, 40)`, and `self.cross_attn_bias_col = nn.Parameter(torch.zeros(num_joints, feat_w))` → shape `(70, 24)`. MATCHES design exactly.
- Both parameters are zero-initialized. CORRECT.

### `_init_head_weights()`
- Warm-start block present but gated by `cross_attn_bias_type == 'factored_warmstart'`. For design002 (`type='factored'`), this block is inactive. MATCHES design specification.

### `forward()`
- Bias routing for factored case:
  - `bias = (self.cross_attn_bias_row.unsqueeze(-1) + self.cross_attn_bias_col.unsqueeze(-2))` → `(70, 40, 1) + (70, 1, 24)` = `(70, 40, 24)`.
  - `bias.view(self.num_joints, -1)` → `(70, 960)`.
  - Passed as `attn_mask` via `cross_attn_bias=bias`. MATCHES design exactly.
- Broadcasting dimensions: `unsqueeze(-1)` on `(70, 40)` → `(70, 40, 1)` and `unsqueeze(-2)` on `(70, 24)` → `(70, 1, 24)`. This correctly computes `u_i[h] + v_i[w]` at position `[i, h, w]`. CORRECT.

### `config.py`
- Head dict contains: `use_cross_attn_bias=True`, `cross_attn_bias_type='factored'`, `feat_h=40`, `feat_w=24`. ALL PRESENT as specified.
- Values are bool/str/int literals. No import statements. MMEngine-compliant.
- `joint_row_prior` not present in config (correct — design002 uses zero-initialization only).

## Invariant Preservation
- All invariants identical to design001. CONFIRMED.

## Test Output
- Job completed: `[test] Finished.` with no errors.
- 72 iterations of epoch 1 logged with finite losses (joints ~0.19–0.22, depth ~1.4–3.7, uv ~0.05–0.3).
- `grad_norm: 8.159518` at iter 50 — well within normal range; clipped by `max_norm=1.0`.
- Checkpoint saved at epoch 1. Training completed normally.

# Code Review — idea021/design001

**Verdict: APPROVED**

## Automated Check
`python scripts/cli.py review-check-implementation runs/idea021/design001` — PASSED.

## Files Changed vs. Design Specification
`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design. PRESENT.
- `code/config.py` — required by design. PRESENT.

No files outside the allowed set (`config.py`, `pose3d_transformer_head.py`) were modified. `pelvis_utils.py` and `train.py` are byte-for-byte identical to baseline.

## Code vs. Design Fidelity

### `_DecoderLayer.forward()`
- Signature extended with `cross_attn_bias: 'torch.Tensor | None' = None`. MATCHES design exactly.
- Cross-attention block: `if cross_attn_bias is not None: … attn_mask=cross_attn_bias.to(q.dtype)`. MATCHES design exactly.
- `.to(q.dtype)` cast for AMP float16 compatibility. PRESENT.

### `Pose3dTransformerHead.__init__()`
- New kwargs: `use_cross_attn_bias=False`, `cross_attn_bias_type='full'`, `feat_h=40`, `feat_w=24`, `joint_row_prior=None`. ALL PRESENT with correct defaults.
- Instance attributes stored: `self.use_cross_attn_bias`, `self.cross_attn_bias_type`, `self.feat_h`, `self.feat_w`, `self.joint_row_prior`. ALL PRESENT.
- Parameter allocation: `self.cross_attn_bias = nn.Parameter(torch.zeros(num_joints, feat_h * feat_w))` when `use_cross_attn_bias=True` and `cross_attn_bias_type='full'`. MATCHES design. Shape = `(70, 960)` as specified.
- Factored branch for 'factored'/'factored_warmstart' also present (required for backward-compatibility across all three designs sharing the same file). CORRECT.

### `_init_head_weights()`
- Warm-start block present but gated by `cross_attn_bias_type == 'factored_warmstart'`. For design001 (`type='full'`), this block is inactive. MATCHES design specification.

### `forward()`
- Bias routing block: `if self.use_cross_attn_bias:` dispatches to `cross_attn_bias_type == 'full'` branch, using `self.cross_attn_bias` directly as `bias`, then calls `self.decoder_layer(queries, spatial, cross_attn_bias=bias)`. MATCHES design exactly.
- `loss()` and `predict()` call `self.forward(feats)` unchanged. CORRECT.

### `config.py`
- Head dict contains: `use_cross_attn_bias=True`, `cross_attn_bias_type='full'`, `feat_h=40`, `feat_w=24`. ALL PRESENT as specified.
- Values are bool/str/int literals. No import statements. MMEngine-compliant.

## Invariant Preservation
- `pelvis_utils.py`: unchanged (identical to baseline).
- `train.py`: unchanged (identical to baseline).
- Body joint loss restricted to indices 0–21 (`_BODY = list(range(0, 22))`): unchanged.
- `persistent_workers=False`: unchanged.
- `resume=True`, `max_keep_ckpts=1`, `CheckpointHook` interval=1: unchanged.
- AMP via `FixedAmpOptimWrapper`: unchanged.
- Seed 2026, batch 4, accum 8: unchanged.

## Test Output
- Job completed: `[test] Finished.` with no errors.
- 72 iterations of epoch 1 logged with finite losses (joints ~0.19–0.23, depth ~1.3–3.7, uv ~0.05–0.3).
- `grad_norm: inf` at iter 50 is consistent with other ideas' first-epoch test runs (seen in idea015, idea017, idea018) — not a design-specific issue; `clip_grad=max_norm=1.0` handles this in full training.
- Checkpoint saved at epoch 1. Training completed normally.

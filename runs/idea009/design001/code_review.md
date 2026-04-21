# Code Review — idea009/design001

**Verdict: APPROVED**

## Review Summary

`review-check-implementation` passed.

### Files Changed
`implementation_summary.md` lists `pose3d_transformer_head.py` and `config.py` — both are files required by `design.md`. No unauthorised files changed.

### pose3d_transformer_head.py

All four required changes are present and correct:

1. `_DecoderLayer.forward` — signature extended with `spatial_drop_prob: float = 0.0` (line 103). Correct.
2. Cross-attention key_padding_mask logic — `key_padding_mask = None` outside block; `if self.training and spatial_drop_prob > 0.0` guard; mask generated as `torch.rand(B, N_spatial, device=spatial_tokens.device) < spatial_drop_prob` (boolean, shape `(B, N_spatial)`, fresh per call, correct device); passed to `self.cross_attn(..., key_padding_mask=key_padding_mask)` (lines 121–128). Matches design exactly.
3. `Pose3dTransformerHead.__init__` — `spatial_drop_prob: float = 0.0` added after `dropout` (line 163); stored as `self.spatial_drop_prob = spatial_drop_prob` (line 184). Correct.
4. `Pose3dTransformerHead.forward` — decoder call is `self.decoder_layer(queries, spatial, spatial_drop_prob=self.spatial_drop_prob)` (line 260). Correct.

Invariants checked:
- Mask not registered as buffer. Correct.
- At inference (`self.training == False`), `key_padding_mask=None` is passed. Correct.
- Absolute imports used throughout. No relative imports introduced. Correct.
- `pelvis_utils.py` unchanged. Correct.
- Loss restricted to body joints 0–21 (`_BODY = list(range(0, 22))`). Correct.

### config.py

- `spatial_drop_prob=0.15` added to head kwargs (line 138). Correct value per design.
- All baseline values unchanged: optimizer, LR schedule, seeds, batch size, accumulative_counts, persistent_workers=False, custom_imports including `'pose3d_transformer_head'`. Correct.
- No Python `import` statements. Correct.

### test_output

- Test train ran to completion: "Done training!" confirmed in SLURM log.
- 81 iterations completed, epoch 1 val results written to `metrics.csv`.
- `composite_val=495.74`, `mpjpe_body_val=442.89`, `mpjpe_pelvis_val=603.05` — results are epoch-1 only (test run), within expected range for an untrained/early epoch.
- `iter_metrics.csv` shows decreasing loss trends across 81 iterations. No NaN or divergence.
- No runtime errors or exceptions observed.

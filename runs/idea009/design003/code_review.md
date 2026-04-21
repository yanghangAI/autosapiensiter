# Code Review — idea009/design003

**Verdict: APPROVED**

## Review Summary

`review-check-implementation` passed.

### Files Changed
`implementation_summary.md` lists `pose3d_transformer_head.py` and `config.py` — both are files required by `design.md`. No unauthorised files changed.

### pose3d_transformer_head.py

All seven required changes are present and correct:

1. **New imports** — `from mmengine.registry import HOOKS` and `from mmengine.hooks import Hook` added at lines 31–32, after `from mmengine.structures import InstanceData`. Correct.

2. `_DecoderLayer.forward` — signature extended with `spatial_drop_prob: float = 0.0` (line 105). Cross-attention mask logic identical to designs 001/002 (lines 121–130). Correct.

3. `Pose3dTransformerHead.__init__` — accepts `spatial_drop_prob_start: float = 0.30` and `spatial_drop_prob_end: float = 0.10` (lines 169–170); stores both as attributes (lines 191–192); initialises `self.spatial_drop_prob = spatial_drop_prob_start` (line 194) so epoch 1 starts at p=0.30 before hook fires. Correct.

4. `set_drop_prob` method — defined at lines 219–221, updates `self.spatial_drop_prob = float(p)`. Correct.

5. `Pose3dTransformerHead.forward` — decoder call passes `spatial_drop_prob=self.spatial_drop_prob` (line 274). Correct.

6. `SpatialDropAnnealHook` class — defined at lines 374–406 after `Pose3dTransformerHead`, registered via `@HOOKS.register_module()`. `before_train_epoch` computes `epoch = runner.epoch + 1` (0-indexed to 1-indexed), applies linear formula `p = start_prob + (end_prob - start_prob) * t` where `t = (epoch - 1) / (num_epochs - 1)`. Defensive DDP unwrap pattern `if hasattr(model, 'module'): model = model.module` present (lines 403–404). Calls `model.head.set_drop_prob(p)` (line 406). Correct.

7. Annealing formula matches design table: epoch 1 → p=0.30, epoch 20 → p=0.10. Correct.

Invariants checked:
- Mask not registered as buffer. Correct.
- At inference, `key_padding_mask=None`. Correct.
- Absolute imports used; no relative imports added. Correct.
- `pelvis_utils.py` unchanged. Correct.
- Loss restricted to body joints 0–21. Correct.

### config.py

- Head kwargs use `spatial_drop_prob_start=0.30` and `spatial_drop_prob_end=0.10` (lines 139–140). Correct.
- `custom_hooks` includes `dict(type='SpatialDropAnnealHook', num_epochs=20, start_prob=0.30, end_prob=0.10)` (line 101). Correct type name, correct parameters.
- `custom_imports` retains `'pose3d_transformer_head'` unchanged — hook registered on module import. Correct.
- All baseline values unchanged. `persistent_workers=False` preserved. No Python `import` statements. Correct.

### test_output

- Test train ran to completion: "Done training!" confirmed in SLURM log.
- 81 iterations completed, epoch 1 val results written to `metrics.csv`.
- `composite_val=503.46`, `mpjpe_body_val=441.98`, `mpjpe_pelvis_val=628.29` — expected early-epoch values; match design002 as expected (both start at p=0.30 for epoch 1, same random seed).
- `iter_metrics.csv` shows decreasing loss trend. No NaN or divergence.
- No runtime errors or exceptions from hook registration or `set_drop_prob` calls.

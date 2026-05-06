## Design Review — idea035 / design002 (Gaussian Noise Depth Ablation)

**Verdict: APPROVED**

### Coverage check
- Design Description: explicit (replace channel 3 with `torch.randn_like(...)`, fresh per step; RGB untouched; mode='gauss').
- Starting point: `baseline/`.
- Files to modify: only `pose3d_transformer_head.py` (same `DepthAblationDataPreprocessor` class as design001) and `config.py` (`mode='gauss'`). `pelvis_utils.py` untouched. Invariant files listed as untouched.
- Algorithmic change: full class source given; `'gauss'` branch (`inputs[:, 3:4] = torch.randn_like(inputs[:, 3:4])`) explicit; config swap explicit.
- Config values: `mode='gauss'` set; assert on allowed modes.
- Training/loss/data: only depth channel content changes; everything else preserved (body-only loss, AMP, persistent_workers, schedule).
- Constraints/edge cases: AMP dtype, device, no fixed-seed-in-forward, defensive fallbacks all enumerated. Explicit note that exact baseline depth post-norm scale is not unit variance — intentional substitution of order-of-magnitude noise.

### Invariant compliance
- `rgbd_data_preprocessor.py` itself untouched; subclass is in `pose3d_transformer_head.py`.
- All other invariant files listed as untouched.
- No Python imports inside the MMEngine config.

### Notes
- The shared class file is identical to design001/003 — Builder will produce one class supporting all three modes; only `mode` kwarg in config differs per design.

No fixes required.

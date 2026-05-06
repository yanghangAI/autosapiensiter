## Design Review — idea035 / design001 (Zero Depth Ablation)

**Verdict: APPROVED**

### Coverage check
- Design Description: present and explicit (replace channel 3 with zeros via `DepthAblationDataPreprocessor` subclass, mode='zero'; RGB untouched).
- Starting point: `baseline/`.
- Files to modify: only `pose3d_transformer_head.py` (append new class) and `config.py` (one-line `data_preprocessor` swap). `pelvis_utils.py` untouched. All invariant files explicitly listed as untouched.
- Algorithmic change: full class source given, including the `'zero'` branch (`inputs[:, 3:4].zero_()`) and config swap to `dict(type='DepthAblationDataPreprocessor', mode='zero')`. Builder does not need to guess.
- Config values / defaults: `mode='zero'` set; class default `'zero'`; assert on allowed mode strings.
- Training/loss/data/inference changes: none beyond data preprocessor channel-3 replacement; body-only loss preserved; AMP and persistent_workers invariants preserved.
- Constraints/edge cases: dtype, device, AMP, val-pass behavior, defensive fallbacks for non-tensor or shape-mismatch inputs all enumerated.

### Invariant compliance
- `RGBDPoseDataPreprocessor` source file is not modified — subclass lives in experimentable `pose3d_transformer_head.py` (allowed; the file is in custom_imports).
- Dataset, transforms, backbone, metric, train.py, infra/* untouched.
- MMEngine config uses string literals only.

### Notes
- The import of `RGBDPoseDataPreprocessor` from `mmpose.models.data_preprocessors.rgbd_data_preprocessor` is consistent with `custom_imports` in baseline `config.py:31`.
- Stage-2 expected non-firing is correctly noted (ablation, not a leaderboard attempt).

No fixes required.

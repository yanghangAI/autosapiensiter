**Files changed:**
- code/pose3d_transformer_head.py
- code/config.py

**Changes:**
- code/pose3d_transformer_head.py: Added an import of `RGBDPoseDataPreprocessor` and appended a new module-level `DepthAblationDataPreprocessor` class (registered via `@MODELS.register_module()`) that subclasses `RGBDPoseDataPreprocessor` and, after the parent forward, replaces channel index 3 of the 4-channel input tensor according to `mode` ('zero'/'gauss'/'shuffle'). The 'shuffle' branch reshapes channel 3 to (B, H*W) and applies a fresh per-sample `torch.randperm(H*W, device=flat.device)` permutation, fully destroying spatial alignment while preserving the per-sample marginal histogram. RGB channels (0..2) are untouched.
- code/config.py: Changed `model.data_preprocessor` from `dict(type='RGBDPoseDataPreprocessor')` to `dict(type='DepthAblationDataPreprocessor', mode='shuffle')` to enable the spatially-shuffled depth ablation for this design.

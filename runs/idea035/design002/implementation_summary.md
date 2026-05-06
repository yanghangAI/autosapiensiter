**Files changed:**
- code/pose3d_transformer_head.py
- code/config.py

**Changes:**
- code/pose3d_transformer_head.py: Added an import of `RGBDPoseDataPreprocessor` and appended a new module-level `DepthAblationDataPreprocessor` class (registered via `@MODELS.register_module()`) that subclasses `RGBDPoseDataPreprocessor` and, after the parent forward, replaces channel index 3 of the 4-channel input tensor according to `mode` ('zero'/'gauss'/'shuffle'). RGB channels (0..2) are untouched. The 'gauss' branch uses `torch.randn_like(inputs[:, 3:4])` so dtype/device are preserved without manual casting; no seeding inside `forward`.
- code/config.py: Changed `model.data_preprocessor` from `dict(type='RGBDPoseDataPreprocessor')` to `dict(type='DepthAblationDataPreprocessor', mode='gauss')` to enable the Gaussian-noise depth ablation for this design.

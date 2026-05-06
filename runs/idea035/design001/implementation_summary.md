**Files changed:**
- code/pose3d_transformer_head.py
- code/config.py

**Changes:**
- code/pose3d_transformer_head.py: Added an import of `RGBDPoseDataPreprocessor` and appended a new module-level `DepthAblationDataPreprocessor` class (registered via `@MODELS.register_module()`) that subclasses `RGBDPoseDataPreprocessor` and, after the parent forward, replaces channel index 3 of the 4-channel input tensor according to `mode` ('zero'/'gauss'/'shuffle'). RGB channels (0..2) are untouched. Defensive fallbacks return the parent output unchanged if `inputs` is missing, non-tensor, not 4-D, or has fewer than 4 channels.
- code/config.py: Changed `model.data_preprocessor` from `dict(type='RGBDPoseDataPreprocessor')` to `dict(type='DepthAblationDataPreprocessor', mode='zero')` to enable the zero-depth ablation for this design.

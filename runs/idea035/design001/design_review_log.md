## 2026-05-05 — Design Review

**Verdict:** APPROVED

Design fully specifies: subclass `DepthAblationDataPreprocessor(RGBDPoseDataPreprocessor)` appended to `pose3d_transformer_head.py`; `'zero'` branch sets channel 3 to zeros; `config.py` switches `model.data_preprocessor` to `dict(type='DepthAblationDataPreprocessor', mode='zero')`. RGB untouched, all invariant files (rgbd_data_preprocessor.py, dataset, transforms, backbone, metric, infra, train.py) untouched. Defensive fallbacks for non-tensor / shape-mismatch inputs included. AMP, persistent_workers, body-only loss, schedule preserved. Builder has full code; no guessing required.

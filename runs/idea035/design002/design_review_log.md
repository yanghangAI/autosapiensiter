## 2026-05-05 — Design Review

**Verdict:** APPROVED

Design fully specifies: same `DepthAblationDataPreprocessor` class as design001 appended to `pose3d_transformer_head.py`; `'gauss'` branch replaces channel 3 with `torch.randn_like(...)`; `config.py` switches `model.data_preprocessor` to `dict(type='DepthAblationDataPreprocessor', mode='gauss')`. RGB untouched, no fixed seed inside `forward`, defensive fallbacks included, AMP / persistent_workers / body-only loss / schedule preserved. Invariants respected. Builder has full code.

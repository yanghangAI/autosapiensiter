## 2026-05-05 — Design Review

**Verdict:** APPROVED

Design fully specifies: same `DepthAblationDataPreprocessor` class as design001/002 appended to `pose3d_transformer_head.py`; `'shuffle'` branch performs per-sample `torch.randperm(H*W, device=flat.device)` permutation of channel 3 and reshapes back; `config.py` switches `model.data_preprocessor` to `dict(type='DepthAblationDataPreprocessor', mode='shuffle')`. RGB untouched, per-sample independent permutations explicitly required, device-local `randperm`, defensive fallbacks included, AMP / persistent_workers / body-only loss / schedule preserved. Invariants respected. Builder has full code.

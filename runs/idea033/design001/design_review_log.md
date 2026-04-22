## Design review — 2026-04-22

APPROVED. All required sections present; files limited to `pose3d_transformer_head.py` + `config.py`; invariants preserved; zero-init FiLM guarantees step-0 baseline parity; K extraction/routing via `_build_k_batch` in `loss()`/`predict()` is fully specified.

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `_KFilmMLP` module (Linear(6,64) → GELU → Linear(64, 2*hidden_dim), output Linear zero-initialised so gamma/beta start at 0 and the module is identity at step 0). Added `use_k_film`, `k_film_variant`, `k_film_hidden` kwargs to `Pose3dTransformerHead.__init__`, plus `_W_REF=384.0` / `_H_REF=640.0` class-level constants and `_build_k_batch()` helper that reads `K` and `img_shape` from each sample's metainfo and stacks the normalised 6-vector `[fx/W_ref, fy/H_ref, cx/cw, cy/ch, ch/H_ref, cw/W_ref]`. Extended `forward()` to accept an optional `k_batch` and added three variant-guarded FiLM blocks (query/spatial/pelvis). For this design (`k_film_variant='query'`), FiLM modulates the expanded joint queries as `queries = queries * (1+gamma) + beta` before the decoder layer. `loss()` and `predict()` now build `k_batch` via `_build_k_batch` (only when `use_k_film`) and pass it to `forward`. Body-only joint loss (0–21), output keys/shapes, and MPJPE telemetry are preserved.
- `code/config.py`: Added `use_k_film=True`, `k_film_variant='query'`, `k_film_hidden=64` inside the `head=dict(...)` block, immediately after `loss_weight_uv=1.0`. No other changes.

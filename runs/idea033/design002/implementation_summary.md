**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Same unified head source as Design 001 — added `_KFilmMLP` (zero-init output), `use_k_film`/`k_film_variant`/`k_film_hidden` kwargs, `_build_k_batch()` helper, `forward()` accepts optional `k_batch`, and `loss()`/`predict()` build and thread `k_batch`. For this design (`k_film_variant='spatial'`), the guarded block that executes modulates the projected spatial tokens after `input_proj` and positional encoding: `spatial = spatial * (1+gamma.unsqueeze(1)) + beta.unsqueeze(1)`, broadcasting one `(gamma, beta)` per sample across all H*W=960 tokens, so cross-attention keys/values become K-conditioned. Output keys/shapes, body-only joint loss (0–21), and MPJPE telemetry unchanged.
- `code/config.py`: Added `use_k_film=True`, `k_film_variant='spatial'`, `k_film_hidden=64` inside the `head=dict(...)` block after `loss_weight_uv=1.0`. No other changes.

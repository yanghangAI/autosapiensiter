**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Same unified head source as Designs 001/002 — added `_KFilmMLP` (zero-init output), `use_k_film`/`k_film_variant`/`k_film_hidden` kwargs, `_build_k_batch()` helper, `forward()` accepts optional `k_batch`, and `loss()`/`predict()` build and thread `k_batch`. For this design (`k_film_variant='pelvis'`), only the guarded pelvis block executes: after `decoded = self.decoder_layer(queries, spatial)` and `joints = self.joints_out(decoded)` (computed on the unmodulated `decoded` so body joints stay K-invariant), the pelvis token `decoded[:, 0, :]` is modulated as `pelvis_token * (1+gamma) + beta` before `depth_out`/`uv_out`. Output keys/shapes, body-only joint loss (0–21), and MPJPE telemetry unchanged.
- `code/config.py`: Added `use_k_film=True`, `k_film_variant='pelvis'`, `k_film_hidden=64` inside the `head=dict(...)` block after `loss_weight_uv=1.0`. No other changes.

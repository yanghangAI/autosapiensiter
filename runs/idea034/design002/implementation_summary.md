**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/pelvis_utils.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py` — Same depth/K extraction helpers and `forward(feats, metric_xyz=None)` wiring as design001 (adds `pe3d` to spatial tokens after PE_2D). The PE module is replaced by `_SinusoidalMetric3DPE(hidden_dim, sigmas)`: per-axis sin/cos at K characteristic scales (σ in metres), concatenated in `(axis, scale)`-deterministic order as 6K features, then projected through a zero-initialised `Linear(6K, hidden_dim)` so PE_3D ≡ 0 at step 0. `__init__` signature replaces `metric_pe_mlp_hidden` with `metric_pe_sigmas: Tuple[float, ...] = (0.25, 1.0, 4.0, 16.0)`; the assertion requires `metric_pe_variant == 'sinusoidal'`.
- `code/pelvis_utils.py` — Identical `unproject_grid_to_metric_3d` as in design001 (BEDLAM2 sign convention, fp32 internal math, NaN/Inf-safe clamp).
- `code/config.py` — Added `use_metric_pe_3d=True`, `metric_pe_variant='sinusoidal'`, `metric_pe_sigmas=(0.25, 1.0, 4.0, 16.0)`, `metric_pe_depth_clamp_min=0.1`, `metric_pe_depth_clamp_max=50.0` to activate the Variant-B sinusoidal PE.

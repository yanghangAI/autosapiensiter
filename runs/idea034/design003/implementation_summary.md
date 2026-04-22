**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/pelvis_utils.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py` — Same `_Metric3DPE` MLP, depth/K extraction helpers, and `loss()`/`predict()` wiring as design001. The `_DecoderLayer.forward` signature was changed to `(queries, spatial_values, spatial_keys=None)`; it now passes `spatial_keys` as the cross-attention keys and `spatial_values` as the values, with `spatial_keys=None` falling back to `spatial_values` (baseline-compatible). `forward()` now builds `spatial_values = input_proj(feat) + PE_2D` and, when metric PE is active, constructs `spatial_keys = spatial_values + pe3d` — so `pe3d` only flows through the keys. Values remain pure appearance + PE_2D. `__init__` default variant changed to `'keys_only'` and the assertion restricts to that literal; zero-init of the MLP final Linear preserves baseline at step 0.
- `code/pelvis_utils.py` — Identical `unproject_grid_to_metric_3d` as in design001.
- `code/config.py` — Added `use_metric_pe_3d=True`, `metric_pe_variant='keys_only'`, `metric_pe_mlp_hidden=256`, `metric_pe_depth_clamp_min=0.1`, `metric_pe_depth_clamp_max=50.0` to activate the Variant-C keys-only metric PE.

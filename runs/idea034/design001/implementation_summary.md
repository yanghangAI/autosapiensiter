**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/pelvis_utils.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py` — Added `_Metric3DPE` module (2-layer MLP 3→mlp_hidden→hidden_dim with GELU; zero-init final Linear). Added `_extract_depth_map` helper (copy of idea004 pattern that loads NPZ/NPY, crops to `img_shape`, and bilinearly resizes to the feature grid). Added `_build_K_batch` helper producing `(B,3,3)` intrinsics and `(B,2)` crop `(h,w)` per sample. Added `_compute_metric_xyz` orchestrator that calls `unproject_grid_to_metric_3d`. Extended `forward(feats, metric_xyz=None)` with a branch that adds `metric_pe_3d(metric_xyz)` to the spatial tokens after PE_2D when the flag is on. Threaded new construction through `loss()` and `predict()` (gated on `use_metric_pe_3d`). Added five new keyword args to `__init__` (`use_metric_pe_3d`, `metric_pe_variant`, `metric_pe_mlp_hidden`, `metric_pe_depth_clamp_min`, `metric_pe_depth_clamp_max`); module is not created when disabled, so the off-path remains a true no-op.
- `code/pelvis_utils.py` — Added `unproject_grid_to_metric_3d(depth_grid, K_batch, crop_hw, feat_h, feat_w, d_min, d_max)` that mirrors the exact sign convention of `recover_pelvis_3d` (`X=d, Y=-(u-cx)X/fx, Z=-(v-cy)X/fy`), uses pixel-centre offsets `(w+0.5)*crop_w/W'`, does fp32 arithmetic internally with NaN/Inf-safe `torch.where` before `clamp`, and casts back to input dtype. `recover_pelvis_3d` and `compute_mpjpe_abs` unchanged.
- `code/config.py` — Added `use_metric_pe_3d=True`, `metric_pe_variant='mlp_additive'`, `metric_pe_mlp_hidden=256`, `metric_pe_depth_clamp_min=0.1`, `metric_pe_depth_clamp_max=50.0` to the head block so this design activates the Variant-A metric PE.

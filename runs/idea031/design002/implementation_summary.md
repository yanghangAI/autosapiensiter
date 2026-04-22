**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pelvis_utils.py`: Same helpers as design001 — `uv_to_grid_coords` and `build_gaussian_heatmap_2d` appended at end of module.
- `code/pose3d_transformer_head.py`: Same code changes as design001 — UV heatmap classification head with zero-init `uv_heatmap_proj`, soft-argmax output, optional KL/cross-entropy heatmap loss, baseline path preserved when disabled.
- `code/config.py`: Added head kwargs for Design B — `use_uv_heatmap=True`, `uv_heatmap_loss_weight=1.0`, `uv_heatmap_sigma=1.0`, `uv_heatmap_target='gaussian'`, `uv_heatmap_learnable_temp=False`, `feat_h=40`, `feat_w=24` (tight Gaussian target, doubled heatmap loss weight).

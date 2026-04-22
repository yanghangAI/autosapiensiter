**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pelvis_utils.py`: Same helpers as design001 — `uv_to_grid_coords` and `build_gaussian_heatmap_2d` appended at end of module.
- `code/pose3d_transformer_head.py`: Same code changes as design001 — UV heatmap classification head. When `uv_heatmap_learnable_temp=True`, a learnable scalar parameter `self.uv_heatmap_temp` (initialized to 1.0) is allocated and applied as `softmax(logits / softplus(temp))` in `forward()`.
- `code/config.py`: Added head kwargs for Design C — `use_uv_heatmap=True`, `uv_heatmap_loss_weight=0.5`, `uv_heatmap_sigma=2.0`, `uv_heatmap_target='gaussian'`, `uv_heatmap_learnable_temp=True`, `feat_h=40`, `feat_w=24` (soft target, moderate weight, learnable softmax temperature).

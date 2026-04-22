**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pelvis_utils.py`: Appended two helpers `uv_to_grid_coords` (maps normalized UV in [-1, 1] to feature-grid (row, col) coordinates) and `build_gaussian_heatmap_2d` (builds an L1-normalized 2D Gaussian target heatmap flattened to (B, H*W)) used by the new UV heatmap loss.
- `code/pose3d_transformer_head.py`: Added `use_uv_heatmap`, `uv_heatmap_loss_weight`, `uv_heatmap_sigma`, `uv_heatmap_target`, `uv_heatmap_learnable_temp`, `feat_h`, `feat_w` kwargs to `Pose3dTransformerHead.__init__`; when enabled, constructs a zero-initialized `uv_heatmap_proj = Linear(hidden_dim, 1)` instead of `uv_out`, branches `forward()` to compute pelvis UV via softmax over spatial tokens + soft-argmax over the 40x24 grid, and adds a KL/cross-entropy heatmap loss against a Gaussian target in `loss()`. Baseline path (`use_uv_heatmap=False`) is preserved bit-exact. Also imported `torch.nn.functional as F` and the two new helpers.
- `code/config.py`: Added the new head kwargs under `model.head`: `use_uv_heatmap=True`, `uv_heatmap_loss_weight=0.5`, `uv_heatmap_sigma=2.0`, `uv_heatmap_target='gaussian'`, `uv_heatmap_learnable_temp=False`, `feat_h=40`, `feat_w=24` — Design A (soft target, moderate weight, no learnable temperature).

**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pelvis_utils.py`: Added `project_joints_to_grid_coords` helper function that projects absolute 3D joints (camera-frame) through the crop intrinsic matrix K to obtain (h_frac, w_frac) coordinates in the feature grid (H'=40, W'=24), using the same BEDLAM2 camera convention as `recover_pelvis_3d`.

`code/pose3d_transformer_head.py`: Added `_build_gaussian_heatmap_target` module-level helper for Gaussian heatmap targets (used by designs B/C, included here for completeness). Updated `__init__` with new kwargs (`use_heatmap_init`, `heatmap_loss_weight`, `heatmap_target`, `heatmap_sigma`, `heatmap_temperature`, `heatmap_learnable_temp`, `feat_h`, `feat_w`); when `use_heatmap_init=True`, creates a zero-initialised `heatmap_proj: Linear(256, 22)` and optionally a learnable per-joint temperature parameter. Updated `forward()` to compute heatmap logits, convert to soft attention weights via softmax, soft-pool spatial tokens to get per-joint feature vectors, and add these (zero-padded to 70 joints) to static query embeddings before the decoder. Updated `loss()` to compute cross-entropy heatmap loss against hard one-hot targets (for `heatmap_target='onehot'`) or KL-divergence against Gaussian targets, averaged over the batch.

`code/config.py`: Added heatmap head kwargs to the model config: `use_heatmap_init=True`, `heatmap_loss_weight=0.1`, `heatmap_target='onehot'`, `heatmap_temperature=1.0`, `heatmap_learnable_temp=False`, `feat_h=40`, `feat_w=24` — all literals, no imports, MMEngine-compliant.

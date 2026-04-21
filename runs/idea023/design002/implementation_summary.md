**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pelvis_utils.py`: Added `project_joints_to_grid_coords` helper function that projects absolute 3D joints (camera-frame) through the crop intrinsic matrix K to obtain (h_frac, w_frac) coordinates in the feature grid (H'=40, W'=24), using the same BEDLAM2 camera convention as `recover_pelvis_3d`.

`code/pose3d_transformer_head.py`: Same architectural changes as design001 — added `_build_gaussian_heatmap_target` helper, updated `__init__` with heatmap kwargs, updated `forward()` with heatmap-guided query warm-start, and updated `loss()` with heatmap loss. Design002 uses `heatmap_target='gaussian'` which activates the KL-divergence path in `loss()`, computing a Gaussian heatmap target (σ=2 grid cells) centred on each GT joint's projected grid position and minimising KL divergence between the predicted softmax distribution and this target. Fixed reviewer-identified bug: changed `-(gt_hm * log_probs).sum()` to `-(gt_hm * log_probs).sum(dim=-1).mean()` so the per-sample KL loss averages over the 22 joints (instead of summing over both joints and H'W' spatial dimensions, which inflated the loss ~22×).

`code/config.py`: Added heatmap head kwargs: `use_heatmap_init=True`, `heatmap_loss_weight=0.2`, `heatmap_target='gaussian'`, `heatmap_sigma=2.0`, `heatmap_temperature=1.0`, `heatmap_learnable_temp=False`, `feat_h=40`, `feat_w=24` — all literals, MMEngine-compliant.

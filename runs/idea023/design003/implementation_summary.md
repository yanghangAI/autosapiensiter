**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pelvis_utils.py`: Added `project_joints_to_grid_coords` helper function that projects absolute 3D joints (camera-frame) through the crop intrinsic matrix K to obtain (h_frac, w_frac) coordinates in the feature grid (H'=40, W'=24), using the same BEDLAM2 camera convention as `recover_pelvis_3d`.

`code/pose3d_transformer_head.py`: Same architectural changes as design001/002 — added `_build_gaussian_heatmap_target`, heatmap kwargs, heatmap-guided query warm-start in `forward()`, and heatmap KL-divergence loss in `loss()`. Design003 additionally uses `heatmap_learnable_temp=True`, which creates `self.heatmap_temp = nn.Parameter(torch.ones(22))` in `__init__` and applies `F.softplus(self.heatmap_temp).view(1, 22, 1)` as a per-joint temperature in `forward()` — the `(1, 22, 1)` shape broadcasts over `(B, 22, H'W')` so each joint's logits are divided by its own temperature before softmax, allowing the model to learn per-joint sharpness of the attention distribution. An initial bug used `view(1, 1, 22)` (wrong broadcast axis) and was corrected. Fixed reviewer-identified bug: changed `-(gt_hm * log_probs).sum()` to `-(gt_hm * log_probs).sum(dim=-1).mean()` so the per-sample KL loss averages over the 22 joints (instead of summing over both joints and H'W' spatial dimensions, which inflated the loss ~22×).

`code/config.py`: Added heatmap head kwargs: `use_heatmap_init=True`, `heatmap_loss_weight=0.2`, `heatmap_target='gaussian'`, `heatmap_sigma=2.0`, `heatmap_temperature=1.0`, `heatmap_learnable_temp=True`, `feat_h=40`, `feat_w=24` — all literals, MMEngine-compliant.

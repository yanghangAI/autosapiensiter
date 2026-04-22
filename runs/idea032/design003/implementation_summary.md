**Files changed:**
- `code/pelvis_utils.py`
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pelvis_utils.py`: Same as design001/002 — added `import torch.nn.functional as F` and a `downsample_depth_map(depth_map, feat_h, feat_w)` helper.
- `code/pose3d_transformer_head.py`: Same scaffolding as design001/002 (module-global RGBD-capture `forward_pre_hook`, aux-depth kwargs on `__init__`, zero-init `aux_depth_head`, aux prediction in `forward()`, aux loss in `loss()`). The `loss()` code already contains the first-order spatial-gradient consistency branch gated by `self.aux_depth_grad_weight > 0`, which is activated here via config.
- `code/config.py`: Appended `use_aux_depth=True, aux_depth_loss_weight=0.3, aux_depth_log_space=True, aux_depth_grad_weight=0.5, aux_depth_valid_min=0.1, aux_depth_valid_max=30.0, aux_depth_denorm_scale=20.0, feat_h=40, feat_w=24` to the `head=dict(...)` block. All values are literals. The resulting aux loss is `0.3 * (recon_loss + 0.5 * grad_loss)` in log-space.

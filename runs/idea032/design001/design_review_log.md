[2026-04-22 17:01 UTC] Design review performed. Verdict: APPROVED.

- Verified files-to-modify are within allowed set (config.py, pose3d_transformer_head.py, pelvis_utils.py).
- Verified invariants are not touched: bedlam2_transforms.py, bedlam2_dataset.py, sapiens_rgbd.py, rgbd_data_preprocessor.py, infra/*, train.py, tools/train.py.
- Verified aux_depth_denorm_scale=20.0 matches _DEPTH_MAX_METERS=20.0 in bedlam2_transforms.py:87.
- Verified feat_h=40, feat_w=24 match img_h=640, img_w=384 with stride 16.
- Verified baseline head file has `spatial = spatial + pos_enc` line (L241) and `B, C, H, W = feat.shape` (L235); insertion points are valid.
- Verified config baseline `head=dict(...)` has `loss_weight_uv=1.0,` as the anchor line for new kwargs append.
- Infrastructure approach (global module forward_pre_hook on torch.nn.modules.module) is non-invasive and respects invariants; flagged as unusual but acceptable.
- All kwargs are bool/int/float literals; MMEngine no-import constraint satisfied.
- Zero-init, empty-mask safety, preemption-safe hook registration, and validation-time safety are all documented.

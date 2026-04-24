# Designer Memory

This file serves as the persistent memory storage for the Designer agent. Keep it concise.

## idea033 (Camera-intrinsic FiLM on normalized K) — 3 designs drafted 2026-04-22
- design001: Variant A — query FiLM (applied to all 70 joint queries before decoder self-attn).
- design002: Variant B — spatial-token FiLM (after input_proj + pos_enc, before cross-attn).
- design003: Variant C — pelvis-token FiLM only (after decoder, before depth_out/uv_out; body joints untouched — matches K-invariant body vs K-dependent pelvis causal structure).
- Shared scaffolding: `_KFilmMLP` module (Linear(6,64)->GELU->Linear(64,2*hidden_dim)) with zero-init of final Linear for identity start; three new head kwargs (`use_k_film=False`, `k_film_variant='query'|'spatial'|'pelvis'`, `k_film_hidden=64`); `_build_k_batch` helper reads `ds.metainfo['K']` + `ds.metainfo['img_shape']` and normalizes by `_W_REF=384.0`, `_H_REF=640.0`; `forward()` signature extended to accept optional `k_batch` tensor; `loss()` and `predict()` build and pass `k_batch`.
- FiLM formula uses `(1+gamma)` form so gamma=0 means identity. All three designs recover baseline bit-for-bit at step 0 and when `use_k_film=False`.
- All 3 designs pass `review-check`. Ready for Orchestrator → Reviewer handoff.

## idea032 (Auxiliary dense depth-map reconstruction from spatial tokens) — 3 designs drafted 2026-04-22
- design001: raw-metre L1 recon, λ=0.1, no grad term (diagnostic).
- design002: log1p-depth recon, λ=0.3, no grad term.
- design003: log1p-depth recon + edge-preserving first-order gradient consistency (sub-weight 0.5), λ=0.3.
- Shared scaffolding in all 3: nine new head kwargs (`use_aux_depth`, `aux_depth_loss_weight`, `aux_depth_log_space`, `aux_depth_grad_weight`, `aux_depth_valid_min`, `aux_depth_valid_max`, `aux_depth_denorm_scale`, `feat_h=40`, `feat_w=24`); zero-init `nn.Linear(256, 1)` aux head; `downsample_depth_map` helper in `pelvis_utils.py`.
- **Infrastructure workaround**: depth channel is not carried in `data_samples.metainfo` (only `depth_npy_path` + `frame_idx` are, and those yield the UNCROPPED depth). To access the cropped depth that actually reaches the backbone without touching any invariant file (`bedlam2_transforms.py`, `bedlam2_dataset.py`, `sapiens_rgbd.py`, `rgbd_data_preprocessor.py`, `rgbd_pose3d.py` estimator), the head registers a **global module forward pre-hook** via `torch.nn.modules.module.register_module_forward_pre_hook` that captures any 4-D float tensor with `C==4` (the RGBD batch entering the backbone) into a module-level dict keyed by device. Head's `loss()` reads it back and denormalises by `aux_depth_denorm_scale=20.0` (matches `_DEPTH_MAX_METERS`). Hook registration is guarded by `_RGBD_CAPTURE_HOOK_REGISTERED` to avoid duplicates on re-import.
- All 3 designs pass `review-check`. Ready for Orchestrator → Reviewer handoff.

## idea013 (Kinematic-chain bone-vector output parameterization) — 3 designs drafted 2026-04-17
- design001: minimal bone-vec head (kinematic_parametrization=True, 1/sqrt(21) init scaling, no aux loss, no per-limb heads).
- design002: design001 + bone-length L1 auxiliary loss on magnitudes only, weight=0.3, key `loss/bone_length/train`.
- design003: design001 + per-limb heads (5 heads: spine/left_leg/right_leg/left_arm/right_arm) via `limb_index=[0,1,2,0,1,2,0,1,2,0,1,2,0,3,4,0,3,4,3,4,3,4]`.
- Shared SMPL-X parent list identical to idea012/design002: `[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]`.
- Shared head signature across all 3 designs: `kinematic_parametrization`, `bone_parents`, `bone_length_loss_weight`, `per_limb_heads`, `limb_index` — defaults reproduce baseline bit-for-bit.
- Key implementation detail: `_forward_kinematics` must clone input, zero the root, use Python-list `_bone_parents_list` to avoid per-iteration device syncs.
- All 3 designs pass `review-check`; registered in design_overview.csv; ready for Orchestrator→Reviewer handoff.
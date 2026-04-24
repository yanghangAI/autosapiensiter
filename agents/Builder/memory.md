# Builder Memory

This file serves as the persistent memory storage for the Builder. Keep it concise.

## idea014 — Anchor-Based Pelvis Depth via Discretized Classification (2026-04-17)

All 3 designs implemented with a SINGLE shared head file across all 3 designs; designs differ only by config kwargs.

- Shared 6 new head kwargs: `depth_head_type` (str), `num_depth_bins=64`, `depth_range_min=1.0`, `depth_range_max=15.0`, `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight`. Default `depth_head_type='regression'` preserves baseline.
- design001: `depth_head_type='classification'`, aux_reg=0.0. Fixed log-uniform bins + SORD soft-target CE, no SmoothL1. Test job 55739450.
- design002: `depth_head_type='classification_hybrid'`, aux_reg=0.3. Design001 + F.smooth_l1_loss(expected, gt, beta=0.05) as `loss/depth_reg/train`. Test job 55739453.
- design003: `depth_head_type='classification_adaptive'`, aux_reg=0.3. Adds `self.depth_bins_head = Linear(hidden,K)`; per-sample widths=softmax*R, cumsum+prepend-zero → edges, midpoints = per-sample centres. SORD target `.detach()`ed so width head trains only via SmoothL1. sigma_log per-sample = 1.5 × median(|Δlog_centres|). Test job 55739454.

Key pattern: `pred['depth_logits']` (B,K) and `pred['depth_bin_centres']` (B,K) added to return dict; regression mode sets them to None. `loss()` branches on mode. `predict()` unchanged (reads `pelvis_depth` scalar). Added top-level `import torch.nn.functional as F`.

All 3 passed review-check-implementation. Awaiting sanity-test + Reviewer code audit.

## idea013 — Kinematic Chain Bone-Vector Output Parameterization (2026-04-17)

All 3 designs implemented with a single shared head file across designs.

- design001: kinematic_parametrization=True, bone_parents=SMPL-X 22-tree, bone_length_loss_weight=0.0, per_limb_heads=False. Forward reinterprets joints_out's first 22 rows as bone vecs; _forward_kinematics recovers joints via cumulative sum; joints_out.weight scaled by 1/sqrt(21). SLURM test job 55671439 submitted.
- design002: same head + bone_length_loss_weight=0.3 (active L1 bone-length aux loss on magnitudes, key 'loss/bone_length/train'). SLURM test job 55671440 submitted.
- design003: same + per_limb_heads=True, limb_index=[0,1,2,0,1,2,0,1,2,0,1,2,0,3,4,0,3,4,3,4,3,4] (spine/left_leg/right_leg/left_arm/right_arm). body_limb_heads = ModuleList of 5 Linear(256,3); tokens routed via index_select + index_copy (out-of-place); joints_out still serves hand tokens. SLURM test job 55671441 submitted.

All 3 passed review-check-implementation. Shared head file simplifies Design 002/003 edits to just copying design001's head and flipping config kwargs. Awaiting test job completion then Reviewer code audit.

## idea002 — Dedicated Pelvis Query with Decoupled Head (2026-04-16)

All 3 designs implemented and sanity-tested successfully.

- design001: decouple_pelvis=True, shared decoder weights, cross-attn only via direct sub-component calls on decoder_layer. +1 Embedding(1,256). SLURM test job 55634585 PASSED.
- design002: decouple_pelvis=True, pelvis_decoder_type='independent', fully independent _DecoderLayer with own weights, cross-attn only. SLURM test job 55634586 PASSED.
- design003: decouple_pelvis=True, pelvis_decoder_type='depth_fused', independent pelvis_decoder + depth_proj Linear(256,256), global depth token prepended to spatial before pelvis cross-attn. SLURM test job 55634587 PASSED.

Key pattern: all 3 designs add `decouple_pelvis` + `pelvis_decoder_type` params; forward() branches on these. setup-design uses `baseline/` as src. review-check-implementation passed for all 3 before testing.

Awaiting Reviewer code audit (code_review.md approval_token required before status changes to Implemented).
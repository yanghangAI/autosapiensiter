**Verdict:** APPROVED

**Summary:** Design A (sigma=2, lambda_heatmap=0.5, fixed temperature) is a minimal, self-contained replacement of the scalar UV regression by a 2D softmax/soft-argmax over the 40x24 spatial feature grid. All changes are confined to the three experimentable files (`pose3d_transformer_head.py`, `pelvis_utils.py`, `config.py`). Invariant files (metric, dataset, transforms, backbone, preprocessor, infra, `train.py`) are untouched.

**Checks:**
- Design Description present.
- Starting point specified: `baseline/`.
- Files to change: only the three experimentable files.
- Exact algorithm: zero-init `Linear(hidden_dim, 1)` on post-pos-enc spatial tokens; softmax; marginal soft-argmax; SmoothL1 retained (weight 1.0) plus cross-entropy against Gaussian target (weight 0.5).
- Exact kwargs and defaults enumerated (`use_uv_heatmap`, `uv_heatmap_loss_weight=0.5`, `uv_heatmap_sigma=2.0`, `uv_heatmap_target='gaussian'`, `uv_heatmap_learnable_temp=False`, `feat_h=40`, `feat_w=24`).
- Output contract preserved: `pred['pelvis_uv']` shape `(B, 2)` in `[-1, 1]`.
- Row/col convention explicitly nailed down (row = H-axis marginal → v_frac; col = W-axis marginal → u_frac), matching the baseline flatten order (verified against `baseline/pose3d_transformer_head.py`: `spatial = feat.flatten(2).transpose(1, 2)`).
- Gaussian target normalization + clamp handles out-of-range centers.
- `self._uv_attn` lifetime is explicit (set in forward, consumed + cleared in loss).
- AMP/fp16 safety addressed (clamp + log on clamped probabilities).
- MMEngine config constraint satisfied (literals only).
- Pelvis depth regression untouched; body joint loss untouched.

**Nits (non-blocking, do not require revision):**
- `gt_uv` detach is left to the Builder as "defensively recommended"; since BEDLAM2 GT carries no grad, this is optional.
- Builder should wire kwargs through MMEngine registry via `**kwargs` or explicit signature; standard practice, not a gap.

Builder can implement without guessing. Approved for implementation.

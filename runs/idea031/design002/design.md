**Design Description:** 2D spatial heatmap classification head for pelvis UV; tight Gaussian target (sigma=1 cell); KL heatmap loss weight 1.0; no learnable temperature.

**Starting Point:** `baseline/`

---

## Algorithm

Identical architecture to idea031/design001, but with a tighter Gaussian target heatmap (sigma=1.0 grid cell) and doubled heatmap loss weight (lambda_heatmap=1.0). The tighter Gaussian provides a sharper classification target, pushing the attention distribution to peak firmly at the GT cell; the higher weight increases the classification signal's contribution relative to the continuous soft-argmax SmoothL1 loss. All other aspects (module structure, zero-init, forward/loss branching, helpers) are identical to design001.

## Overview

Design B from idea031 — aggressive sharpening variant. Uses Gaussian sigma=1.0 (roughly one cell wide) and doubles the heatmap loss weight to emphasize the classification gradient. Intended to test whether a sharper target yields more precise UV localization when the backbone features can produce reliable peaked distributions. Output interface `pred['pelvis_uv']` is unchanged.

---

## Files to Change

1. `pose3d_transformer_head.py` — same changes as design001 (identical code).
2. `pelvis_utils.py` — same two helpers as design001 (identical code).
3. `config.py` — same kwargs as design001 but with different values for `uv_heatmap_loss_weight` and `uv_heatmap_sigma`.

---

## `pelvis_utils.py` Changes

Identical to design001. Add `uv_to_grid_coords` and `build_gaussian_heatmap_2d` helpers. (See design001 for the full code; the Builder should copy them verbatim.)

---

## `pose3d_transformer_head.py` Changes

Identical to design001:
- Import the two new helpers from `pelvis_utils`.
- Add the seven new kwargs to `__init__` signature and store them as attributes.
- Replace `self.uv_out = nn.Linear(hidden_dim, 2)` with the gated construction of `self.uv_heatmap_proj = nn.Linear(hidden_dim, 1)` (zero-init) when `use_uv_heatmap=True`; keep `self.uv_out` for the baseline branch.
- In `forward()`, branch on `self.use_uv_heatmap`: compute `uv_logits → softmax → soft-argmax` to produce `pelvis_uv` of shape `(B, 2)` in `[-1, 1]`; stash `self._uv_attn` for the loss.
- In `loss()`, branch on `self.use_uv_heatmap` and add the KL/cross-entropy heatmap loss against the Gaussian target.
- `predict()` unchanged.

The Builder MUST implement the head code identically to design001; the only difference between designs A and B is in `config.py`.

---

## `config.py` Changes

In the `model` dict, under `head=dict(...)`, add:

```python
use_uv_heatmap=True,
uv_heatmap_loss_weight=1.0,
uv_heatmap_sigma=1.0,
uv_heatmap_target='gaussian',
uv_heatmap_learnable_temp=False,
feat_h=40,
feat_w=24,
```

All values are bool/int/float/str literals. No Python `import` statements.

Everything else in `config.py` is unchanged (optimizer, LR schedule, data pipeline, hooks).

---

## Invariants to Preserve

Identical to design001:
- `persistent_workers=False` — unchanged.
- Loss restricted to body joints — unchanged.
- `resume=True`, `CheckpointHook` `max_keep_ckpts=1` — unchanged.
- `accumulative_counts=8`, `batch_size=4` — unchanged.
- `seed=2026` — unchanged.
- AMP via `FixedAmpOptimWrapper` — unchanged.
- No `import` in `config.py` — satisfied.
- `pred['pelvis_uv']` shape `(B, 2)` in `[-1, 1]` — preserved.
- `recover_pelvis_3d`, `compute_mpjpe_abs`, `bedlam_metric.py` — not touched.

---

## Expected Behaviour After Change

- At init: identical to design001. `uv_heatmap_proj` zero-init → uniform attention → `pelvis_uv = (0, 0)`.
- During training: the tighter Gaussian target (sigma=1.0) has most of its mass in a single cell (~88% within a 3x3 neighbourhood, vs ~39% for sigma=2.0). The cross-entropy loss is therefore more demanding: any probability mass placed outside the GT cell's immediate neighbourhood carries larger penalty.
- Higher loss weight (1.0 vs 0.5 in design001) doubles the contribution of the heatmap term to the total loss, at parity with the SmoothL1 UV and joint losses.
- Expected risk: sharper target + higher weight can amplify noise early in training if the backbone features are not yet producing coherent spatial distributions; however, zero-init means the first step starts from uniform attention, which is a stable starting point for cross-entropy. If loss diverges at step ~100–500, the Builder should log the loss curve and flag for the Designer to fall back to design001 values.
- Parameter count delta: identical to design001 (−257 params).
- Memory/speed: identical to design001; no change.
- Shape/interface of `pred['pelvis_uv']` unchanged.

---

## Edge Cases and Constraints

- Identical to design001 (row/col convention, UV normalization convention, GT UV out-of-range handling, single-pelvis assumption, AMP dtype, `self._uv_attn` lifetime, baseline path gating).
- **Sigma=1.0 edge behaviour**: when `gt_grid` falls near the image border (row=0 or row=H-1, col=0 or col=W-1), the Gaussian mass is cut off on one side. `build_gaussian_heatmap_2d` renormalizes by `hm.sum().clamp(min=1e-6)`, so the target is always a valid probability distribution. The Builder should not add special-case handling for edge positions.
- **Sharpness vs fp16**: sigma=1.0 produces Gaussian values as small as `exp(-0.5 * (39^2 + 23^2)/2) ≈ exp(-973)` at the far corner. In fp16 these underflow to 0; after renormalization the distant cells have probability 0, which is fine (no log is taken on the target, only on the prediction). The prediction `uv_attn` is clamped at 1e-8 before log, so no `log(0)` arises.

---

## Target Metrics (Stage 1)

- `composite_val < 322` (vs. best prior 323.75)
- `mpjpe_pelvis_val < 590` (vs. best prior 608)
- `mpjpe_abs_val < 770` (vs. baseline 833)
- `mpjpe_body_val` not expected to regress.

If design B outperforms design A at stage-1, the sharpening hypothesis is validated and a narrower-sigma sweep is a natural follow-up idea. If design B regresses relative to design A, the conclusion is that the baseline spatial features do not yet support sharp peaked distributions at sigma=1.0 scale.

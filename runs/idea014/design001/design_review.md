# Design Review — idea014 / design001

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 001 replaces the baseline scalar pelvis-depth regression head
(`Linear(hidden_dim, 1)` + `SoftWeightSmoothL1Loss(beta=0.05)`) with a
K=64-way classification head over fixed log-uniform depth bins in
`[1.0, 15.0]` m, recovering the scalar pelvis depth as the soft-argmax
expectation over bin centres. Supervision is SORD-style soft-target
cross-entropy (Gaussian in log-depth centred at `log(z_gt)` with
`σ = 1.5 × bin_width_log`). No auxiliary regression term. The returned
dict still yields a `(B, 1)` `pelvis_depth` tensor preserving every
downstream shape contract. Design fully specifies the shared 6-kwarg
`__init__` signature (used identically by Designs 002/003 with
different values), exact buffer registration, forward/loss changes,
gradient paths, and config literals.

## Checklist

- [x] `**Design Description:**` present and precise.
- [x] Starting point declared: `baseline/`.
- [x] Only allowed files modified: `pose3d_transformer_head.py` and
  `config.py`. `pelvis_utils.py` explicitly unchanged.
- [x] Exact `__init__` signature: six new kwargs
  (`depth_head_type`, `num_depth_bins`, `depth_range_min`,
  `depth_range_max`, `depth_soft_label_sigma`, `depth_aux_reg_weight`)
  appended after `loss_weight_uv` and before `init_cfg`, with defaults
  (`'regression'`, `64`, `1.0`, `15.0`, `1.5`, `0.0`) that reproduce
  baseline behaviour bit-for-bit when omitted.
- [x] Validation assertions for `depth_head_type` enum and numeric
  ranges explicitly specified.
- [x] Head allocation conditional: `Linear(hidden_dim, K)` in
  classification mode vs. `Linear(hidden_dim, 1)` in regression mode.
  Non-persistent buffer `log_bin_centres` = `torch.linspace(log 1, log 15, 64)`.
- [x] `_init_head_weights` left unchanged (trunc-normal std=0.02 applies
  to K-wide weight matrix unmodified).
- [x] `forward()` branch explicit: soft-argmax expectation
  `(softmax(logits) * exp(log_bin_centres)).sum(-1, keepdim=True)`
  returned as `(B, 1) pelvis_depth`; `depth_logits` and
  `depth_bin_centres` additionally exposed for `loss()`.
- [x] `loss()` SORD soft-CE computation fully specified:
  `target = softmax(-(log_centres - log(z_gt))^2 / (2σ_log^2))`,
  `L_CE = -(target * log_softmax(logits)).sum(-1).mean()`, with GT
  clamped to `[z_min, z_max]` before `log()`. `loss/depth/train`
  retains the existing key and multiplier `loss_weight_depth=1.0`.
- [x] `depth_aux_reg_weight=0.0` makes the aux regression branch a
  no-op in Design 001 (gated `if > 0.0`).
- [x] `predict()` and MPJPE `torch.no_grad()` block explicitly
  unchanged; both consume `pred['pelvis_depth']` scalar.
- [x] Exact config values: `depth_head_type='classification'`,
  `num_depth_bins=64`, `depth_range_min=1.0`, `depth_range_max=15.0`,
  `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight=0.0` — all
  str/int/float literals, MMEngine-config compliant, no `import`.
- [x] `loss_depth=dict(...)` retained in config for signature
  uniformity even though unused in classification mode.
- [x] Invariants preserved: `persistent_workers=False`, body-only
  joint loss (`_BODY = list(range(0, 22))`), absolute imports in head,
  seed 2026, batch 4, accumulative_counts=8, LR schedule
  (LinearLR + CosineAnnealingLR, `convert_to_iter_based=True`), hooks
  untouched.
- [x] Invariant files not modified: `bedlam_metric.py`,
  `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`,
  data preprocessor, `infra/constants.py`,
  `infra/metrics_csv_hook.py`, `train.py` wrapper, `tools/train.py`,
  `pelvis_utils.py`.
- [x] Edge cases handled: GT clamp to `[z_min, z_max]` before `log()`
  prevents `log(0)` NaN; softmax normalisation avoids hand-rolled
  Gaussian-sum numerical issues; target implicitly non-differentiable
  (computed from non-grad GT); head weight-shape change does not
  collide with baseline checkpoint (head is fresh-init; only backbone
  is pretrained).
- [x] Gradient flow explicit: CE gradient via
  `log_softmax(depth_logits)` into `depth_out.weight`; expectation is
  differentiable but used only in no-grad MPJPE computation in
  Design 001.
- [x] Same three loss keys as baseline (`loss/joints/train`,
  `loss/depth/train`, `loss/uv/train`); `_compute_mpjpe_abs` contract
  preserved (`(B, 1)` scalar in metres).
- [x] Parameter delta calculated (+16 191 float32 weights on
  `depth_out`); runtime overhead <0.1%; memory overhead ~1 kB.

## Minor observations (non-blocking)

- Init expectation ≈ 5.5 m (arithmetic mean of exp(linspace centres)),
  biased slightly above the BEDLAM2 dataset mean ≈ 4 m. Design
  explicitly argues SORD CE pulls this toward the correct range
  within a few hundred iters of warmup; no mitigation required.
- CE magnitude at init (~`log(64) ≈ 4.16`) is ~8× larger than
  baseline SmoothL1 early loss; design notes AdamW per-parameter
  adaptive LR absorbs this. Acceptable; `loss_weight_depth=1.0`
  retained.
- Optional bias-init trick to peak softmax near 4 m is noted as
  non-required; design does not mandate it.

## Verdict

APPROVED — Builder can implement without guessing.

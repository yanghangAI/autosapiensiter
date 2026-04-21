# Design Review — idea014 / design003

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 003 inherits Design 002's hybrid classification + SORD soft-CE
+ SmoothL1-on-expectation loss and additionally adopts AdaBins-style
per-sample adaptive bin widths. A second head
`depth_bins_head: Linear(hidden_dim, K)` emits per-sample softmax
width logits, scaled by `(z_max − z_min)`, cumulatively summed with a
zero prepend, shifted by `z_min`, and midpoint-averaged to produce
per-sample bin centres in `[1.0, 15.0]` m. The SORD soft-CE target
is computed against per-sample `log(bin_centres)` with per-sample
`σ_log = 1.5 × median(|Δ log-centre|)`, and the target is `.detach()`ed
(load-bearing invariant to avoid degenerate "moving target" collapse).
Gradient flow is cleanly bifurcated: `depth_out` trained by both CE
and SmoothL1 (via the expectation), `depth_bins_head` trained only
by SmoothL1 (via the expectation through bin centres) — the detach
blocks the CE path into the width head.

## Checklist

- [x] `**Design Description:**` present and precise.
- [x] Starting point declared: `baseline/`.
- [x] Only allowed files modified: `pose3d_transformer_head.py` and
  `config.py`. `pelvis_utils.py` explicitly unchanged.
- [x] Same shared 6-kwarg signature as Design 001/002; only
  `depth_head_type='classification_adaptive'` differs (and
  `depth_aux_reg_weight=0.3` as in Design 002).
- [x] Second head allocated ONLY in adaptive mode:
  `self.depth_bins_head = nn.Linear(hidden_dim, num_depth_bins)`.
  Designs 001/002 explicitly do NOT allocate it; attempting access
  would raise `AttributeError`.
- [x] `_init_head_weights` extended to include `depth_bins_head` in
  the trunc-normal init loop when adaptive mode is active.
- [x] Adaptive bin-centre computation fully specified:
  `widths = softmax(width_logits) * (z_max − z_min)` → `cumsum` →
  `cat([zeros, cumsum])` → `+ z_min` → midpoints
  `0.5 * (edges[:, :-1] + edges[:, 1:])`. Shape `(B, K)`. Endpoints
  exactly `z_min` and `z_max` per sample.
- [x] Explicit load-bearing invariants: width-softmax × `(z_max − z_min)`
  scaling (constraint 33); zero-prepend for K+1 edges (34); shift
  AFTER cumsum+prepend (35); midpoints of consecutive edges (36);
  `.detach()` on SORD target (37); per-sample σ_log = median
  (constraint 38).
- [x] `forward()` `else` branch split with
  `if self.depth_head_type == 'classification_adaptive':` for
  per-sample centres; fixed path preserved for Designs 001/002.
  Same returned dict keys and shapes.
- [x] `loss()` uses `pred['depth_bin_centres']` (per-sample, linear
  metres) and converts to log-space inside `loss()` via
  `bin_centres.clamp(min=z_min * 1e-3).log()` for numerical safety.
- [x] Per-sample `sigma_log` explicit: median of absolute consecutive
  log-centre differences, `keepdim=True` shape `(B, 1)`, multiplied
  by `depth_soft_label_sigma=1.5`. Fixed-mode fallback preserves
  original constant `sigma_log`.
- [x] `target.detach()` explicitly called AFTER `softmax`; design
  notes equivalence of detach-before-softmax but prefers the clearer
  ordering.
- [x] Gradient flow paths documented:
  - `depth_out.weight` ← CE (via `log_softmax`) + SmoothL1 (via
    `softmax → expectation`).
  - `depth_bins_head.weight` ← SmoothL1 only (via
    `widths → cumsum → edges → midpoints → expectation`); CE
    blocked by detached target.
- [x] Four emitted loss keys (`loss/joints/train`, `loss/depth/train`,
  `loss/uv/train`, `loss/depth_reg/train`) — same as Design 002.
- [x] `predict()` and MPJPE no-grad block unchanged.
- [x] Exact config values:
  `depth_head_type='classification_adaptive'`, `num_depth_bins=64`,
  `depth_range_min=1.0`, `depth_range_max=15.0`,
  `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight=0.3`. All
  literals — MMEngine-compliant, no `import`.
- [x] `log_bin_centres` buffer still registered for serialisation
  consistency, though unused at forward time in adaptive mode.
- [x] `pred['depth_logits']` refers only to `self.depth_out`'s output;
  `width_logits` is NOT part of this key (constraint 42).
- [x] `pred['depth_bin_centres']` exposes LINEAR-space centres; log
  conversion happens inside `loss()` (constraint 43).
- [x] Invariants preserved: `persistent_workers=False`, body-only
  joint loss, absolute imports in head, seed 2026, batch 4,
  accumulative_counts=8, LR schedule, hooks untouched.
- [x] Invariant files not modified: `bedlam_metric.py`,
  `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`,
  data preprocessor, `infra/constants.py`,
  `infra/metrics_csv_hook.py`, `train.py` wrapper, `tools/train.py`,
  `pelvis_utils.py`.
- [x] Parameter delta: +32 639 total vs. baseline (+16 191 on
  `depth_out`, +16 448 on `depth_bins_head`). Still < 0.02% of
  model.
- [x] Failure-mode diagnostics specified: if `.detach()` accidentally
  omitted, `widths.std()` collapses and `loss/depth/train` trends to
  0 while `mpjpe_pelvis_val` stays poor — distinctive signature.
- [x] 20-epoch-budget caveat explicit (AdaBins literature uses 100+
  epochs); documented as an empirical datapoint, not a spec defect.

## Minor observations (non-blocking)

- Init adaptive centres are arithmetically uniform over `[1, 15]` m
  (vs. log-uniform for Designs 001/002). Design correctly notes this
  actually matches BEDLAM2 depth mass better than log-uniform init.
  Acceptable.
- `bin_centres.clamp(min=z_min * 1e-3).log()` is a cheap defensive
  safety valve; in practice centres are always well above zero due to
  softmax positivity + `z_min = 1.0` shift. Acceptable.
- Init `L_depth_reg ≈ 0.3 × |8 − 4| = 1.2` is larger than Design 002
  (~0.45) because init expectation is the arithmetic midpoint 8 m.
  This provides stronger early pull toward dataset mean; design
  argues this is beneficial for fast warmup. Accepted.
- The detach discipline is thorough — this is the main correctness
  hazard of AdaBins and the design addresses it explicitly with a
  load-bearing constraint.

## Verdict

APPROVED — Builder can implement without guessing.

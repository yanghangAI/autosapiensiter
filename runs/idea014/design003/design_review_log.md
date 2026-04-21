# Design Review Log — idea014 / design003

## 2026-04-17 — APPROVED

- Reviewer mode: design review
- Idea: idea014 (Anchor-Based Pelvis Depth via Discretized Classification Head)
- Design: 003 (Adaptive per-sample bin widths à la AdaBins +
  SORD soft CE + SmoothL1 hybrid)
- Starting point: baseline/
- Files to change (declared): pose3d_transformer_head.py, config.py
- Invariant files unchanged: confirmed (pelvis_utils.py,
  bedlam_metric.py, bedlam2_dataset.py, bedlam2_transforms.py,
  sapiens_rgbd.py, data preprocessor, infra/*, train.py wrapper,
  tools/train.py).
- Key acceptance points:
  - Same shared 6-kwarg `__init__` signature; values differ only by
    `depth_head_type='classification_adaptive'` (and
    `depth_aux_reg_weight=0.3`, matching Design 002).
  - Second head `self.depth_bins_head = nn.Linear(hidden_dim, 64)`
    allocated ONLY in adaptive mode; Designs 001/002 do not allocate
    it.
  - `_init_head_weights` extended conditionally to trunc-normal init
    the second head.
  - Adaptive bin construction: `softmax(width_logits) × (z_max − z_min)`
    → cumsum → zero-prepend → `+ z_min` shift → midpoint-average.
    Per-sample endpoints exactly `z_min` and `z_max`.
  - Load-bearing invariants (explicit in design):
    - Width softmax scaled by `(z_max − z_min)` (constraint 33).
    - Zero-prepend to produce K+1 edges (34).
    - Shift by `z_min` AFTER cumsum+prepend (35).
    - Midpoints of consecutive edges (36).
    - SORD target `.detach()`ed to prevent moving-target collapse
      (37).
    - Per-sample `σ_log = 1.5 × median(|Δ log-centre|)` (38).
  - Gradient flow: `depth_out.weight` ← CE + SmoothL1;
    `depth_bins_head.weight` ← SmoothL1 only (CE path blocked by
    detached target).
  - Four emitted loss keys (same as Design 002).
  - `predict()` and MPJPE no-grad block unchanged.
  - Config: `depth_head_type='classification_adaptive'`,
    `num_depth_bins=64`, `depth_range_min=1.0`,
    `depth_range_max=15.0`, `depth_soft_label_sigma=1.5`,
    `depth_aux_reg_weight=0.3`. Literals only.
  - Parameter delta: +32 639 total (+16 191 depth_out,
    +16 448 depth_bins_head).
  - Failure-mode diagnostics: if `.detach()` is omitted,
    `widths.std()` collapses and `loss/depth/train → 0` while
    pelvis MPJPE stays poor — diagnosable signature.
  - 20-epoch training-budget caveat documented as empirical-risk,
    not a spec defect.
- Verdict: APPROVED.

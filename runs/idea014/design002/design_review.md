# Design Review — idea014 / design002

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 002 inherits Design 001's classification + SORD soft-argmax
machinery identically (fixed log-uniform bins in `[1.0, 15.0]` m,
K=64 bins, σ = 1.5 × bin_width_log, same forward-pass expectation,
same exposed keys) and additionally activates an auxiliary SmoothL1
regression term on the expectation against GT depth, weighted at
`λ_reg = 0.3`. The auxiliary term emits a new loss key
`loss/depth_reg/train`. This is the BinsFormer-style hybrid
formulation, complementary because CE shapes the bin-probability
landscape while SmoothL1 directly calibrates the metric scalar
recovered by soft-argmax. Gradients from both losses flow coherently
into `depth_out.weight`.

## Checklist

- [x] `**Design Description:**` present and precise.
- [x] Starting point declared: `baseline/`.
- [x] Only allowed files modified: `pose3d_transformer_head.py` and
  `config.py`. `pelvis_utils.py` explicitly unchanged.
- [x] Same shared 6-kwarg signature as Design 001; only values differ
  (`depth_head_type='classification_hybrid'`,
  `depth_aux_reg_weight=0.3`). Backward compatibility with defaults
  preserved.
- [x] Head allocation identical to Design 001 (single
  `Linear(hidden_dim, K)` + `log_bin_centres` buffer). The adaptive
  `depth_bins_head` is explicitly NOT allocated in Design 002.
- [x] `forward()` branch identical to Design 001 (the shared `else`
  path; `classification_hybrid` falls through to fixed log-uniform
  centres).
- [x] `loss()` CE computation identical to Design 001; the
  `if self.depth_aux_reg_weight > 0.0:` branch is ACTIVE and emits
  `loss/depth_reg/train = 0.3 × F.smooth_l1_loss(pelvis_depth, gt_depth, beta=0.05, reduction='mean')`.
- [x] Explicit constraint that `F.smooth_l1_loss` uses `beta=0.05`
  (matching baseline SmoothL1 scale), NOT PyTorch default `beta=1.0`.
- [x] Explicit that `depth_aux_reg_weight` multiplies `L_depth_reg`
  INSIDE the loss term and is NOT multiplied by `loss_weight_depth`
  (independent weighting factors).
- [x] Gradient flow: SmoothL1 on expectation
  `(softmax(logits) * bin_centres).sum(...)` adds differentiable
  signal to `depth_out.weight` on top of CE signal. Fully consistent.
- [x] Four total loss keys emitted (`loss/joints/train`,
  `loss/depth/train`, `loss/uv/train`, `loss/depth_reg/train`);
  `MetricsCSVHook` auto-logs the new column (no hook change
  required).
- [x] `predict()` and MPJPE no-grad block unchanged.
- [x] Exact config values: `depth_head_type='classification_hybrid'`,
  `num_depth_bins=64`, `depth_range_min=1.0`, `depth_range_max=15.0`,
  `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight=0.3`. All
  literals — MMEngine-compliant, no `import`.
- [x] Invariants preserved: `persistent_workers=False`, body-only
  joint loss, absolute imports in head, seed 2026, batch 4,
  accumulative_counts=8, LR schedule, hooks untouched.
- [x] Invariant files not modified: `bedlam_metric.py`,
  `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`,
  data preprocessor, `infra/constants.py`,
  `infra/metrics_csv_hook.py`, `train.py` wrapper, `tools/train.py`,
  `pelvis_utils.py`.
- [x] Rationale for `λ_reg = 0.3` explicit (BinsFormer median; ratio
  to `L_joints` and `L_uv` ≈ 1:1 post-warmup; CE still dominates bin
  shaping).
- [x] Parameter delta identical to Design 001 (+16 191 float32
  weights — no new parameters from hybrid loss).
- [x] Early-training sanity check specified (verify `loss/depth/train`
  and `loss/depth_reg/train` both decrease at iter 50).

## Minor observations (non-blocking)

- Init `L_depth_reg ≈ 0.3 × |5.5 − 4| = 0.45` is a strong pull
  toward the dataset mean at warmup — this is intentional and
  beneficial per the rationale.
- The method-local `import torch.nn.functional as F` inside the
  `loss()` aux branch is acceptable; design also permits an explicit
  top-level import. Either pattern satisfies the invariant (no new
  top-level imports were required).
- No double-counting issue: CE gradient via `log_softmax(logits)`
  and SmoothL1 gradient via `softmax(logits) · bin_centres` both
  flow through the same parameter in coherent directions. AdamW
  normalises per-parameter.

## Verdict

APPROVED — Builder can implement without guessing.

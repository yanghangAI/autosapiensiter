**Verdict: APPROVED** (with confirmed softmax degeneracy risk — see warning below)

**Design:** idea024/design002 — EMA per-joint difficulty weighting (alpha=1.0, softmax T=1.0)

---

## Review Summary

The implementation faithfully matches the design specification. All required changes are present and correct. The design's softmax degeneracy risk (flagged as high-risk during design review) is confirmed to manifest in the actual code and will produce degenerate training behaviour. This does NOT block approval — the code correctly implements what the design specifies, and the risk was flagged at design review. The Orchestrator should expect this experiment to underperform baseline.

---

## Implementation Check

### review-check-implementation

Passed.

### Files Changed

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design. ✓
- `code/config.py` — required by design. ✓

No unexpected files changed.

### Changes vs Design Spec

**`pose3d_transformer_head.py`:**

- `__init__` params: `per_joint_difficulty_weighting: bool = False`, `ema_alpha: float = 0.5`, `ema_momentum: float = 0.99`, `weight_norm: str = 'linear'`, `weight_temperature: float = 1.0` — all present, correctly placed after `loss_weight_uv`, before `init_cfg`. ✓
- Instance attributes stored: all 5 stored immediately after `super().__init__`. ✓
- Buffers registered conditionally: `joint_err_ema = ones(22)` and `_train_iter = zeros(1, long)`. ✓
- `import torch.nn.functional as F` added at module top-level. ✓
- `_get_adaptive_weights()`: softmax branch `w = 22.0 * F.softmax(ema / self.weight_temperature, dim=0)` — exact match to design spec. Linear fallback also present. ✓
- `loss()` replacement: EMA update inside `torch.no_grad()`, softmax-normalised weights, manual smooth-L1 with `beta=0.05`. ✓
- `_train_mpjpe`, `_train_mpjpe_abs`, depth/uv losses, `predict()` — all unchanged. ✓
- Fallback path `per_joint_difficulty_weighting=False` — calls original `loss_joints_module`. ✓

**`config.py`:**

- `model.head` dict contains exactly: `per_joint_difficulty_weighting=True`, `ema_alpha=1.0`, `ema_momentum=0.99`, `weight_norm='softmax'`, `weight_temperature=1.0`. All literals. ✓
- No Python import statements in config. ✓

### Invariant Check

- `pelvis_utils.py` — diff vs baseline: no changes. ✓
- `train.py` — diff vs baseline: no changes. ✓
- No changes to evaluation metric, dataset, transforms, backbone, data preprocessor, infra files. ✓

---

## Test Output Check

- Test ran to completion: `Done training!` and `[test] Finished.` present. ✓
- Epoch 1 completed; checkpoint saved at epoch 1. ✓
- `iter_metrics.csv`: 72 rows (full epoch), `loss_joints_train` values in range 0.31–0.50. ✓
- Training log at iter 50/72: `loss/joints/train: 0.406649`, `grad_norm: 8.766156` — normal. ✓
- No errors, no NaN/inf losses. ✓

Note: `loss/joints/train` is elevated compared to design001 (≈0.40 vs ≈0.23 at iter 50). This is consistent with the softmax degeneracy described below — the loss magnitude is higher because softmax concentrates all weight on the hardest joint, inflating the effective smooth-L1 contribution of that joint.

---

## Confirmed Softmax Degeneracy (Flagged at Design Review)

The design review flagged this design as high-risk because `softmax(ema / T=1.0)` applied to raw mm-scale EMA values (typical range 50–400 mm) would be degenerate.

**This is confirmed to manifest in the actual code.**

Numerical verification:
- After 72 training steps (one epoch), EMA values range approximately 77–155 mm for easy vs hard joints.
- `softmax([77.4 ... 154.4])` at T=1.0 concentrates essentially all weight (≈22.0) on the single hardest joint, with all other joint weights ≈ 0.
- At EMA convergence (150–350 mm range), `softmax([150 ... 350])` places weight ≈22.0 on the hardest joint and ≈0 on all others.
- This is a near-one-hot weighting scheme — the opposite of the "well-calibrated" behaviour claimed by the design.

The elevated `loss_joints_train` (0.31–0.50 range vs 0.20–0.26 for design001) already reflects this concentration in the first epoch of the test run.

**The code correctly implements the design.** The degenerate outcome is a consequence of the design's algorithmic choices (raw mm values at T=1.0), not a Builder error. This design should be expected to perform worse than baseline on `mpjpe_body_val` due to gradient starvation of all but the single hardest joint.

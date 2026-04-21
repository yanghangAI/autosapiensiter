# Code Review — idea012/design003

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 003 (log-scaled / scale-invariant pairwise L1 distance-matrix
auxiliary loss, `dist_loss_weight=0.5`, `dist_loss_mode='log'`,
`dist_loss_eps=1e-3`) is faithfully implemented. All required changes
from `design.md` are present in `code/pose3d_transformer_head.py` and
`code/config.py`. No invariant files were modified. The reduced
test-train ran without errors; the new loss term `loss/dist_matrix/train`
is emitted in the MMEngine log at a value of ~0.79, consistent with the
log-distance loss magnitude (larger raw magnitude than Designs 001/002
because `|log(d)|` is typically > `|Δd|` for small bones).

---

## Pre-check

- `python scripts/cli.py review-check-implementation runs/idea012/design003`:
  **PASSED**.

## implementation_summary.md audit

`**Files changed:**` lists:
- `code/pose3d_transformer_head.py`
- `code/config.py`

Both are allowed per the design. `pelvis_utils.py` is correctly not
listed. The summary is non-empty.

Every change described in `**Changes:**` maps to actual code:

- Three new kwargs added to `__init__`: `dist_loss_weight: float = 0.0`,
  `dist_loss_mode: str = 'abs'`, `dist_loss_eps: float = 1e-3` — present
  at `pose3d_transformer_head.py:161-163`, placed after `loss_weight_uv`
  and before `init_cfg`.
- Stored as attributes at lines 177-179; `dist_loss_mode` asserted
  against `('abs', 'bone_weighted', 'log')` at lines 181-183.
- Sentinel `self.bone_weights = None` — present at line 186. This
  ensures the unreachable `'bone_weighted'` branch in `loss()` does not
  cause an `AttributeError` at module load time; at runtime the
  `'log'` branch is the only one reachable because
  `dist_loss_mode='log'` in config.
- New auxiliary loss block in `loss()` — present at lines 320-342,
  placed AFTER the three baseline losses and BEFORE the
  `with torch.no_grad():` MPJPE block. The `'log'` branch uses
  `eps = self.dist_loss_eps` and computes
  `(torch.log(d_pred + eps) - torch.log(d_gt + eps)).abs().mean()`.
  Stored with exact key `'loss/dist_matrix/train'`, multiplied by
  `self.dist_loss_weight`.
- `forward()`, `predict()`, `_init_head_weights`, and the rest of the
  file are unchanged.
- `config.py` adds three literals to the `head=dict(...)` block:
  `dist_loss_weight=0.5`, `dist_loss_mode='log'`, `dist_loss_eps=1e-3`.

No undocumented changes were found.

## Design-detail fidelity

Every required detail from `design.md` is implemented correctly:

1. Kwarg signature placement and defaults — verified.
2. Assert on `dist_loss_mode` — verified.
3. `eps` added INSIDE each `torch.log(...)` call (not subtracted from
   `log(d_pred) - log(d_gt)`) — verified (line 340).
4. `eps = self.dist_loss_eps = 1e-3` (not `1e-6` or smaller) — verified;
   config sets `dist_loss_eps=1e-3`.
5. `.abs()` applied after the log difference (not `**2` or any other
   norm) — verified.
6. `.mean()` reduction over batch and pair dims — verified.
7. Key name `'loss/dist_matrix/train'`, weight applied after the raw
   mean — verified (line 342).
8. `forward()` and `predict()` unchanged — verified via diff.
9. No new learnable parameters, no new buffers — verified (sentinel
   `self.bone_weights = None` is a plain Python attribute, not a
   buffer/parameter).
10. `config.py` additions are three literals (float/str/float),
    MMEngine-compliant — verified (lines 147-149).
11. `persistent_workers=False` preserved — verified.
12. `custom_imports` list unchanged — verified.

## Invariant compliance

Diffed against baseline:
- `code/pelvis_utils.py` — bit-identical to baseline.
- `code/train.py` — bit-identical to baseline.
- Only `pose3d_transformer_head.py` and `config.py` are modified.

No modifications to `bedlam_metric.py`, `bedlam2_dataset.py`,
`bedlam2_transforms.py`, `sapiens_rgbd.py`, the data preprocessor,
`infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or
`tools/train.py`.

## test_output verification

- Slurm job 55671021 ran without exceptions. No `Traceback`, `Error`,
  or `Exception` strings.
- `iter_metrics.csv` has 81 training iteration rows; baseline losses
  decrease smoothly.
- `metrics.csv` has one validation row for epoch 1 with
  `composite_val=499.9070`, `mpjpe_body_val=445.7613`,
  `mpjpe_pelvis_val=609.8393` — finite and sensible; slightly higher
  than Designs 001/002's epoch-1 numbers, consistent with a different
  (larger-magnitude) auxiliary gradient influencing the first epoch's
  body-MPJPE.
- The MMEngine training log shows the new loss emitted:
  `loss/joints/train: 0.194202  loss/depth/train: 1.557588
  loss/uv/train: 0.111461  loss/dist_matrix/train: 0.787721` — four
  loss keys. `loss/dist_matrix/train ≈ 0.79` is finite and positive
  (no NaN/Inf), and the larger raw magnitude vs Designs 001/002 is
  expected because `|log(d_pred) - log(d_gt)|` on small early-training
  body-joint distances with `eps=1e-3` produces ~0.5-1.5 per-pair log
  errors that average to ~1.5, then multiplied by
  `dist_loss_weight=0.5` gives ~0.79. No evidence of gradient explosion
  (`grad_norm: 8.47` is slightly larger than Designs 001/002's 8.22
  but well within `clip_grad max_norm=1.0` range).
- Backbone still loads 293/293 tensors.
- Run ended cleanly with `Done training!`.

Note: same caveat as Designs 001/002 — `iter_metrics.csv` does not
have a column for `loss_dist_matrix_train` because the invariant
`infra/metrics_csv_hook.py` hardcodes column names. Not a correctness
issue.

## Issues

None that block approval.

---

**Final verdict: APPROVED**

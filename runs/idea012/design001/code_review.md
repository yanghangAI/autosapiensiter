# Code Review — idea012/design001

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 001 (upper-triangular pairwise L1 distance-matrix auxiliary loss on
the 22 body joints, `dist_loss_weight=0.5`, mode `'abs'`) is faithfully
implemented. All required changes from `design.md` are present in
`code/pose3d_transformer_head.py` and `code/config.py`. No invariant
files were modified. The reduced test-train ran without errors and the
new loss term `loss/dist_matrix/train` is emitted in the MMEngine
training log with a plausible value (~0.26).

---

## Pre-check

- `python scripts/cli.py review-check-implementation runs/idea012/design001`:
  **PASSED**.

## implementation_summary.md audit

`**Files changed:**` lists:
- `code/pose3d_transformer_head.py`
- `code/config.py`

Both files are allowed per the design. `pelvis_utils.py` is correctly not
listed. The summary is non-empty, so no "Builder did nothing" reject.

Every change described in `**Changes:**` maps to actual code:

- Three new kwargs added to `__init__`: `dist_loss_weight: float = 0.0`,
  `dist_loss_mode: str = 'abs'`, `dist_loss_eps: float = 1e-3` — present
  at `pose3d_transformer_head.py:161-163`, placed immediately after
  `loss_weight_uv: float = 1.0,` and before `init_cfg: OptConfigType = None,`
  as specified.
- Stored as attributes at `pose3d_transformer_head.py:177-179`.
- Assert on `dist_loss_mode in ('abs', 'bone_weighted', 'log')` at
  `pose3d_transformer_head.py:181-183`.
- New auxiliary loss block in `loss()` — present at
  `pose3d_transformer_head.py:317-342`, placed AFTER the existing three
  loss-dict assignments (`loss/joints/train`, `loss/depth/train`,
  `loss/uv/train`) and BEFORE the `with torch.no_grad():` MPJPE block,
  exactly as specified.
- The block is guarded by `if self.dist_loss_weight > 0.0:`, computes
  `torch.cdist(pred_body, pred_body, p=2)` and the GT counterpart,
  gathers 231 upper-triangular entries with
  `torch.triu_indices(22, 22, offset=1, device=pred_body.device)`, and
  stores `losses['loss/dist_matrix/train'] = self.dist_loss_weight * L_dist`.
- Three-branch `if/elif/else` on `self.dist_loss_mode` is present with
  modes `'abs'` (L1 of distance differences), `'bone_weighted'` (uses
  `self.bone_weights`), and `'log'` (log-distance L1) — matching the
  skeleton shared across Designs 001/002/003. Design 001 takes the
  `'abs'` branch at runtime.
- `forward()`, `predict()`, `_init_head_weights`, and the rest of the
  file are unchanged.

No undocumented changes were found.

## Design-detail fidelity

Every required detail from `design.md` is implemented correctly:

1. Kwarg signature placement (after `loss_weight_uv`, before `init_cfg`)
   — verified (lines 160-164).
2. Defaults preserve baseline behaviour: `dist_loss_weight=0.0`,
   `dist_loss_mode='abs'`, `dist_loss_eps=1e-3` — verified.
3. Validation assert rejects invalid `dist_loss_mode` values —
   verified (line 181).
4. Loss block placement inside `loss()` is correct (after the three
   baseline losses, before the `torch.no_grad()` MPJPE block) —
   verified.
5. `torch.cdist(..., p=2)` is used for both `D_pred` and `D_gt` —
   verified (lines 322-323).
6. `torch.triu_indices(22, 22, offset=1, device=pred_body.device)`
   is used with `offset=1` (diagonal excluded) — verified (line 326).
7. Gather via `D_pred[:, iu[0], iu[1]]` yields a `(B, 231)` tensor —
   verified (lines 327-328).
8. `'abs'` mode uses `(d_pred - d_gt).abs().mean()` (mean, not sum) —
   verified (line 332).
9. Final loss stored with exact key `'loss/dist_matrix/train'`,
   multiplied by `self.dist_loss_weight` after the raw mean —
   verified (line 342).
10. No new learnable parameters. `_init_head_weights`,
    `self.joint_queries`, and the three output projections are untouched —
    verified.
11. No new buffers (no `register_buffer` for `bone_weights`) for Design
    001 — verified (the `'bone_weighted'` branch is only reachable if
    `dist_loss_mode='bone_weighted'`, which Design 001 never sets).
12. `forward()` and `predict()` are bit-identical to baseline — verified
    via diff.
13. `config.py` adds the three kwargs to the `head=dict(...)` block:
    `dist_loss_weight=0.5`, `dist_loss_mode='abs'`, `dist_loss_eps=1e-3`
    — verified (lines 147-149); all float/str literals (no imports).
14. `persistent_workers=False` preserved in both dataloaders —
    verified (lines 178, 193).
15. `custom_imports` list unchanged — verified.

## Invariant compliance

Diffed against baseline:
- `code/pelvis_utils.py` — bit-identical to baseline.
- `code/train.py` — bit-identical to baseline.
- Only `pose3d_transformer_head.py` and `config.py` are modified; the
  `config.py` diff is exactly (a) the auto-patched `output_dir` string
  and (b) the three new head kwargs.

No modifications to `bedlam_metric.py`, `bedlam2_dataset.py`,
`bedlam2_transforms.py`, `sapiens_rgbd.py`, the data preprocessor,
`infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, or
`tools/train.py`.

## test_output verification

- Slurm job 55671019 ran without exceptions. No `Traceback`, `Error`,
  or `Exception` strings in stdout.
- `iter_metrics.csv` has 81 training iteration rows (epoch 1 completed)
  with all three baseline loss columns populated (loss decreasing from
  ~0.21 at iter 1 to ~0.19 at iter 81 for `loss_joints_train`).
- `metrics.csv` has one validation row for epoch 1 with
  `composite_val=490.80`, `mpjpe_body_val=443.23`,
  `mpjpe_pelvis_val=587.38` — finite, sensible first-epoch values.
- The MMEngine training log shows the NEW loss emitted correctly:
  `loss/joints/train: 0.192204  loss/depth/train: 1.551053
  loss/uv/train: 0.110944  loss/dist_matrix/train: 0.257460` — four
  loss keys, with `loss/dist_matrix/train` a finite positive scalar
  (~0.26 m × 0.5 weight = additive contribution similar-order to
  `loss/joints/train`), exactly as the design predicted.
- Backbone still loads 293/293 tensors from the pretrained Sapiens
  checkpoint — backbone code path unchanged.
- Training proceeded through validation and the run ended cleanly with
  `Done training!` in the log.

Note: `iter_metrics.csv` does NOT have a column for `loss_dist_matrix_train`.
This is because `infra/metrics_csv_hook.py` has a hardcoded `_LOSS_MAP` /
`_ITER_COLS` that does not include the new key. The hook is an invariant
file that the Builder may not modify. The primary training log (from
MMEngine) does capture the new loss, and all validation metrics in
`metrics.csv` are unaffected. This is not a correctness issue; the loss
is backpropagated normally and the primary `composite_val` metric is
what drives the evaluation.

## Issues

None that block approval.

---

**Final verdict: APPROVED**

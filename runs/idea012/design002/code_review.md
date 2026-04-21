# Code Review — idea012/design002

**Verdict: APPROVED**

**Reviewed by:** Reviewer
**Date:** 2026-04-17

---

## Summary

Design 002 (bone-length-weighted pairwise L1 distance-matrix auxiliary
loss, up-weighting the 21 SMPL-X parent-child bone edges by factor 2.0
while keeping non-adjacent pairs at 1.0) is faithfully implemented. All
required changes from `design.md` are present in
`code/pose3d_transformer_head.py` and `code/config.py`. No invariant
files were modified. The reduced test-train ran without errors and the
new loss term `loss/dist_matrix/train` is emitted with a plausible
value slightly higher than Design 001's (`0.262558` vs `0.257460`,
reflecting the ~9% mean-weight bump predicted in the design).

---

## Pre-check

- `python scripts/cli.py review-check-implementation runs/idea012/design002`:
  **PASSED**.

## implementation_summary.md audit

`**Files changed:**` lists:
- `code/pose3d_transformer_head.py`
- `code/config.py`

Both are allowed per the design. `pelvis_utils.py` is correctly not
listed. The summary is non-empty.

Every change described in `**Changes:**` maps to actual code:

- FOUR new kwargs added to `__init__`: `dist_loss_weight: float = 0.0`,
  `dist_loss_mode: str = 'abs'`, `dist_loss_eps: float = 1e-3`,
  `bone_parents: list = None` — present at
  `pose3d_transformer_head.py:161-164`, placed after `loss_weight_uv`
  and before `init_cfg`.
- First three stored as attributes; `dist_loss_mode` asserted against
  `('abs', 'bone_weighted', 'log')` — present at lines 178-184.
- Bone-weight construction guarded by
  `if dist_loss_mode == 'bone_weighted':` — present at lines 186-215.
  Builds a `(22, 22)` bool `is_bone` mask, iterates
  `enumerate(bone_parents)` skipping root (`parent < 0`), sets both
  `is_bone[min(child,parent), max(child,parent)]` and the symmetric
  `is_bone[j, i]`, gathers upper-tri entries via
  `torch.triu_indices(22, 22, offset=1)` (same ordering as in `loss()`),
  creates a 231-dim float32 vector with `torch.where` (2.0 for bone
  pairs, 1.0 otherwise), and registers it via
  `self.register_buffer('bone_weights', bone_weights, persistent=False)`.
- `else: self.bone_weights = None` sentinel — present at lines 213-215.
- Assertion that `bone_parents is not None and len(bone_parents) == 22`
  inside the `'bone_weighted'` branch — present at lines 187-189.
- New auxiliary loss block in `loss()` with the same three-branch
  `if/elif/else` as Design 001 — present at lines 349-371. The
  `'bone_weighted'` branch computes
  `w = self.bone_weights.to(d_pred.device); L_dist = (w * (d_pred - d_gt).abs()).mean()`
  (broadcast multiply `(231,) * (B, 231)` → `(B, 231)` → mean over both
  dims).
- `forward()`, `predict()`, `_init_head_weights`, and the rest of the
  file are unchanged.
- `config.py` adds the four literals at the end of the `head=dict(...)`
  block, including
  `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`
  (the exact SMPL-X parent list specified in the design).

No undocumented changes were found.

## Design-detail fidelity

Every required detail from `design.md` is implemented correctly:

1. Kwarg signature placement and defaults — verified.
2. `bone_parents` default `None`; required when `dist_loss_mode == 'bone_weighted'` — verified.
3. `bone_weights` construction order uses `torch.triu_indices(22, 22, offset=1)`,
   matching the order in `loss()` — verified (same call in both places).
4. Symmetric `is_bone[i, j] = is_bone[j, i] = True` — verified.
5. `torch.where` produces 231-dim float32 tensor with exactly 21 entries
   at 2.0 and 210 at 1.0 (by construction: 21 non-root children, each
   contributing one bone edge) — verified via code inspection of the
   parent list: the 21 entries at indices 1..21 each add one unique
   ordered pair.
6. `register_buffer(..., persistent=False)` used — verified (line 212).
7. Sentinel `self.bone_weights = None` for modes other than
   `'bone_weighted'` — verified (line 215). Design 002 takes the
   `'bone_weighted'` branch, so this else-branch is not exercised at
   runtime but is present for Design 003 compatibility.
8. `loss()` gather, mode dispatch, and key naming are identical to
   Design 001 — verified.
9. `w * (d_pred - d_gt).abs()` broadcasting is correct — verified.
10. `.mean()` (not `.sum()`) used for the final reduction — verified.
11. Key name `'loss/dist_matrix/train'` — verified (line 371).
12. No extra learnable parameters. Only a 231-element non-persistent
    buffer — verified.
13. `config.py` sets `dist_loss_weight=0.5`, `dist_loss_mode='bone_weighted'`,
    `dist_loss_eps=1e-3`, `bone_parents=[...]` — verified (lines 147-150);
    all int/float/str list literals (MMEngine-compliant).
14. `persistent_workers=False` preserved — verified.
15. `custom_imports` list unchanged — verified.

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

- Slurm job 55671020 ran without exceptions.
- `iter_metrics.csv` has 81 training iteration rows; losses decrease
  monotonically through the epoch.
- `metrics.csv` has one validation row for epoch 1 with
  `composite_val=490.7184`, `mpjpe_body_val=443.2582`,
  `mpjpe_pelvis_val=587.0770` — finite and sensible.
- The MMEngine training log shows the new loss emitted:
  `loss/joints/train: 0.192214  loss/depth/train: 1.551045
  loss/uv/train: 0.110950  loss/dist_matrix/train: 0.262558` — four
  loss keys with `loss/dist_matrix/train` ~2% higher than Design 001's
  (`0.257460`), consistent with the ~9% mean-weight bump (the actual
  bump depends on which pairs happen to have larger residuals at init).
- Backbone still loads 293/293 tensors from pretrained weights.
- Run ended cleanly with `Done training!`.

Note: same caveat as Design 001 — `iter_metrics.csv` does not have a
column for `loss_dist_matrix_train` because the invariant
`infra/metrics_csv_hook.py` hardcodes column names. Not a correctness
issue.

## Issues

None that block approval.

---

**Final verdict: APPROVED**

# Design Review — idea012 / design002

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 002 specifies the same upper-triangular pairwise L1 auxiliary loss as
Design 001, but multiplies element-wise by a fixed 231-dim bone-weight vector
that up-weights the 21 SMPL-X kinematic parent-child edges by 2.0 (others
1.0). The design is complete, explicit, and implementable without guesswork.

## Checklist

- [x] `**Design Description:**` present and precise.
- [x] Starting point declared: `baseline/`.
- [x] Only allowed files are modified: `pose3d_transformer_head.py`,
  `config.py`. `pelvis_utils.py` explicitly unchanged.
- [x] Exact algorithmic change:
  - Same `torch.cdist` + `torch.triu_indices(22, 22, offset=1, device=...)` as
    Design 001.
  - Mode `'bone_weighted'`: `L_dist = (w * (d_pred - d_gt).abs()).mean()`
    with `w = self.bone_weights.to(d_pred.device)`.
- [x] Exact construction of `self.bone_weights`:
  - Registered as non-persistent buffer `self.bone_weights` via
    `self.register_buffer('bone_weights', ..., persistent=False)`.
  - Built from `bone_parents` list using a `(22,22)` bool `is_bone` mask with
    `min(child, parent)` / `max(child, parent)` canonicalisation, then
    gathered via the same `torch.triu_indices(22, 22, offset=1)` call used in
    `loss()` so ordering matches.
  - `torch.where(is_bone_pairs, full(231, 2.0), full(231, 1.0))` produces
    exactly 21 entries = 2.0 and 210 entries = 1.0.
  - Sanity check available: `(bone_weights == 2.0).sum() == 21`.
- [x] `__init__` signature: four new kwargs (`dist_loss_weight`,
  `dist_loss_mode`, `dist_loss_eps`, `bone_parents: list = None`) with
  defaults that preserve baseline behaviour. Assertion validates
  `dist_loss_mode` and that `bone_parents` is a 22-entry list when
  `'bone_weighted'`.
- [x] Exact config values:
  - `dist_loss_weight=0.5`
  - `dist_loss_mode='bone_weighted'`
  - `dist_loss_eps=1e-3`
  - `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`
  - All plain Python int/float/str/list literals — MMEngine-config compliant.
- [x] SMPL-X 22-joint kinematic tree explicitly documented with parent-list,
  (parent, child) pair enumeration, and count of 21 edges.
- [x] Buffer ordering derived from the same `torch.triu_indices(22, 22,
  offset=1)` call used in `loss()`, preventing index-mismatch bugs.
- [x] `forward()` and `predict()` explicitly unchanged.
- [x] Invariants preserved: `persistent_workers=False`, body-only loss slice,
  `custom_imports`, absolute imports in head file, seed 2026, batch size 4,
  gradient accumulation 8, LR schedule unchanged, hooks untouched, zero
  learnable parameters added (buffer only).
- [x] Invariant files not modified: `bedlam_metric.py`, `bedlam2_dataset.py`,
  `bedlam2_transforms.py`, backbone, data preprocessor, `infra/*`,
  `train.py`, `tools/train.py`.
- [x] Edge cases handled: symmetric assignment `is_bone[i,j] = is_bone[j,i]`
  makes the mask robust to orientation conventions; `min/max` canonicalisation
  defensive against future parent-list edits; non-persistent buffer excluded
  from checkpoint (derived from config each construction).

## Verdict

APPROVED — Builder can implement without guessing.

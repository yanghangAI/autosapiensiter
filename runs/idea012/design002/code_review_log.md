# Code Review Log — idea012/design002

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Bone-length-weighted pairwise L1 distance-matrix auxiliary loss (mode
`'bone_weighted'`, `dist_loss_weight=0.5`) is faithfully implemented.
`pose3d_transformer_head.py` adds four new kwargs (`dist_loss_weight`,
`dist_loss_mode`, `dist_loss_eps`, `bone_parents`) with correct defaults
and placement, stores the first three as attributes, asserts
`dist_loss_mode in ('abs', 'bone_weighted', 'log')`, and — when
`dist_loss_mode == 'bone_weighted'` — builds a `(231,)` non-persistent
float32 buffer `self.bone_weights` with 2.0 for the 21 SMPL-X
parent-child bone edges and 1.0 elsewhere, using the exact ordering
produced by `torch.triu_indices(22, 22, offset=1)` (same call used in
`loss()`). For other modes, `self.bone_weights = None` sentinel. In
`loss()`, the `'bone_weighted'` branch computes
`(self.bone_weights.to(d_pred.device) * (d_pred - d_gt).abs()).mean()`,
stored with the exact key `'loss/dist_matrix/train'`. `forward()`,
`predict()`, and `_init_head_weights` are unchanged.
`config.py` sets `dist_loss_weight=0.5`, `dist_loss_mode='bone_weighted'`,
`dist_loss_eps=1e-3`, and the standard SMPL-X parent list
`[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`
— all as MMEngine-compliant literals. `pelvis_utils.py` and `train.py`
are bit-identical to baseline. `review-check-implementation` passed.
The reduced test-train completed 81 iterations + validation cleanly;
the MMEngine log shows `loss/dist_matrix/train: 0.262558`, slightly
higher than Design 001's `0.257460` as expected from the ~9% mean-weight
bump. No blocking issues.

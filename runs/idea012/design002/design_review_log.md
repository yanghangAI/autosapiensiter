# Design Review Log — idea012 / design002

## 2026-04-17 — APPROVED

- Reviewer mode: design review
- Idea: idea012 (Pairwise Joint Distance-Matrix Structural Prior Loss)
- Design: 002 (Bone-Length-Weighted, λ=0.5, 2x bone pairs)
- Starting point: baseline/
- Files to change (declared): pose3d_transformer_head.py, config.py
- Invariant files unchanged: confirmed (pelvis_utils.py, bedlam_metric.py,
  bedlam2_dataset.py, bedlam2_transforms.py, sapiens_rgbd.py, data preprocessor,
  infra/*, train.py wrapper, tools/train.py).
- Key acceptance points:
  - Bone parent list (SMPL-X 22 joints):
    `[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`
    — 21 non-root entries producing 21 bone edges; list of (parent, child)
    pairs enumerated in the design.
  - Bone-weight buffer `(231,)` constructed from a `(22,22)` `is_bone` bool
    mask and gathered via `torch.triu_indices(22, 22, offset=1)` — the same
    call used in `loss()`, guaranteeing index ordering match.
  - Buffer registered via `register_buffer('bone_weights', ...,
    persistent=False)` — moves with `model.to(device)` but not checkpointed.
  - `loss()` three-branch structure identical across D001/D002/D003; D002
    takes the `'bone_weighted'` branch.
  - Config kwargs: `dist_loss_weight=0.5`, `dist_loss_mode='bone_weighted'`,
    `dist_loss_eps=1e-3`, `bone_parents=[...22 ints...]`. All literals.
  - Zero learnable params; one (231,) non-persistent buffer = 924 bytes.
- Verdict: APPROVED.

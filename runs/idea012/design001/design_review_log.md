# Design Review Log — idea012 / design001

## 2026-04-17 — APPROVED

- Reviewer mode: design review
- Idea: idea012 (Pairwise Joint Distance-Matrix Structural Prior Loss)
- Design: 001 (Upper-triangular Pairwise L1, λ=0.5, mode='abs')
- Starting point: baseline/
- Files to change (declared): pose3d_transformer_head.py, config.py
- Invariant files unchanged: confirmed (pelvis_utils.py, bedlam_metric.py,
  bedlam2_dataset.py, bedlam2_transforms.py, sapiens_rgbd.py, data preprocessor,
  infra/*, train.py wrapper, tools/train.py).
- Key acceptance points:
  - `_BODY = list(range(0, 22))` body-only slice preserved.
  - `torch.cdist(..., p=2)` + `torch.triu_indices(22, 22, offset=1, device=...)`
    explicitly specified with device argument to avoid host-device copies.
  - Loss key `'loss/dist_matrix/train'` matches naming convention for
    `MetricsCSVHook`.
  - `self.dist_loss_weight > 0.0` guard keeps baseline behaviour bit-identical
    when default `0.0` is used.
  - Three-branch `if/elif/else` in `loss()` is structurally identical across
    Designs 001/002/003; unreachable `'bone_weighted'` branch in Design 001 is
    deliberately left as a no-op (mode='abs' selected in config).
  - Exact config insertion specified: three new kwargs appended to
    `head=dict(...)` block (no changes elsewhere).
  - MMEngine config constraints satisfied (only float/str literals).
- Verdict: APPROVED.

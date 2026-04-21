# Design Review Log — idea012 / design003

## 2026-04-17 — APPROVED

- Reviewer mode: design review
- Idea: idea012 (Pairwise Joint Distance-Matrix Structural Prior Loss)
- Design: 003 (Log-Scaled, λ=0.5, eps=1e-3, mode='log')
- Starting point: baseline/
- Files to change (declared): pose3d_transformer_head.py, config.py
- Invariant files unchanged: confirmed (pelvis_utils.py, bedlam_metric.py,
  bedlam2_dataset.py, bedlam2_transforms.py, sapiens_rgbd.py, data preprocessor,
  infra/*, train.py wrapper, tools/train.py).
- Key acceptance points:
  - `L_dist = |log(d_pred + eps) - log(d_gt + eps)|` mean over 231 pairs;
    eps=1e-3 INSIDE each log.
  - `eps=1e-3` (1 mm) chosen for bounded gradient `1/(d+eps) ≤ 1000`;
    smaller eps explicitly disallowed.
  - `clip_grad max_norm=1.0` provides additional gradient safety.
  - `torch.abs` at 0 subgradient is 0 (PyTorch default) — no special
    handling needed.
  - Three-branch `loss()` identical across D001/D002/D003; D003 takes the
    `'log'` branch. `elif 'bone_weighted'` unreachable because mode='log'
    in config (self.bone_weights is None but not accessed).
  - Config kwargs: `dist_loss_weight=0.5`, `dist_loss_mode='log'`,
    `dist_loss_eps=1e-3`. All literals.
  - Zero learnable params; zero buffers.
- Verdict: APPROVED.

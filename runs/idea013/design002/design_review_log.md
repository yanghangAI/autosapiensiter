# Design Review Log — idea013 / design002

## 2026-04-17 — APPROVED

- Reviewer mode: design review
- Idea: idea013 (Kinematic Chain Bone-Vector Output Parameterization)
- Design: 002 (Bone-vector head + auxiliary bone-length L1 loss,
  λ_bone=0.3)
- Starting point: baseline/
- Files to change (declared): pose3d_transformer_head.py, config.py
- Invariant files unchanged: confirmed (pelvis_utils.py, bedlam_metric.py,
  bedlam2_dataset.py, bedlam2_transforms.py, sapiens_rgbd.py, data
  preprocessor, infra/*, train.py wrapper, tools/train.py).
- Key acceptance points:
  - Design 002 reuses all of Design 001's `__init__`, buffer, scale-init,
    `_forward_kinematics`, and `forward()` logic verbatim. Shared head
    file supports all three designs via flag switching.
  - Auxiliary bone-length loss block inserted after the three existing
    loss assignments and before the `with torch.no_grad():` MPJPE block.
    Guarded by `if self.kinematic_parametrization and
    self.bone_length_loss_weight > 0.0`.
  - Bones computed from recovered coordinates `pred['joints'][:, _BODY]`
    and `gt_joints[:, _BODY]`, using `child_idx = torch.arange(1, 22,
    device=...)` and `parent_idx = self.bone_parents[1:22]` to skip the
    root sentinel `-1`.
  - Magnitudes via `.norm(dim=-1)` producing `(B, 21)`; loss value
    `(pred_len - gt_len).abs().mean()` — plain L1, not Smooth-L1.
  - Loss key exactly `'loss/bone_length/train'` — matches
    `MetricsCSVHook` auto-capture naming convention.
  - Multiplied by `self.bone_length_loss_weight` (no separate
    nn.Module).
  - Config values: `kinematic_parametrization=True`,
    `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`,
    `bone_length_loss_weight=0.3`, `per_limb_heads=False`,
    `limb_index=None`. Literals only — MMEngine-config compliant.
  - `forward()` and `predict()` untouched. Loss is training-only.
  - Zero new learnable parameters. Emits FOUR loss keys (one more than
    baseline).
- Verdict: APPROVED.

# Design Review — idea013 / design002

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 002 is Design 001 plus an auxiliary L1 loss on the magnitudes of
the 21 predicted bone vectors vs. the 21 GT bone vectors, with
`bone_length_loss_weight=0.3`. The auxiliary computes bone vectors by
subtracting parent from child in the already-recovered joint coordinate
tensor (equivalent to the raw bone_vec head output by construction, but
implemented on `pred['joints']` so the block is uniform across
Designs 001/002/003). The design is complete, explicit, and
implementable without guesswork.

## Checklist

- [x] `**Design Description:**` present and precise.
- [x] Starting point declared: `baseline/`.
- [x] Only allowed files are modified: `pose3d_transformer_head.py` and
  `config.py`. `pelvis_utils.py` is explicitly unchanged.
- [x] Design 002 explicitly inherits all `__init__`, buffer, init-scale,
  `_forward_kinematics`, and `forward()` spec from Design 001 verbatim
  (Builder reads design001.md for those). The design cleanly delegates
  shared components and specifies only the delta.
- [x] Exact bone-length loss block given (inserted after the three
  existing loss assignments, before the `with torch.no_grad():` MPJPE
  block):
  - `child_idx = torch.arange(1, 22, device=pred['joints'].device)`
  - `parent_idx = self.bone_parents[1:22].to(device)` (slices skip the
    `-1` sentinel)
  - bones computed on `gt_joints[:, _BODY]` and
    `pred['joints'][:, _BODY]` (recovered coords); magnitudes via
    `.norm(dim=-1)`; plain L1 via `.abs().mean()`.
  - Loss key exactly `'loss/bone_length/train'`, scaled by
    `self.bone_length_loss_weight`.
- [x] Insertion/guard is correct: `if self.kinematic_parametrization and
  self.bone_length_loss_weight > 0.0:` — defensive, baseline-safe, and
  disabled when either flag is off.
- [x] Exact config values: `kinematic_parametrization=True`,
  `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`,
  `bone_length_loss_weight=0.3`, `per_limb_heads=False`, `limb_index=None`.
  Literals only — MMEngine-config compliant.
- [x] Invariants preserved: `persistent_workers=False`, body-only loss
  slice, `custom_imports` include `'pose3d_transformer_head'`, absolute
  imports in head file, seed 2026, batch 4, accumulative_counts=8, LR
  schedule unchanged, hooks untouched.
- [x] Invariant files not modified: `bedlam_metric.py`,
  `bedlam2_dataset.py`, `bedlam2_transforms.py`, backbone, data
  preprocessor, `infra/*`, `train.py`, `tools/train.py`.
- [x] Edge cases handled: magnitude-at-zero has valid gradient
  (`.norm(dim=-1)` returns 0 with zero grad, but
  `|0 - gt_len|` still has a valid gradient for `gt_len > 0`); use of
  plain L1 (not MSE, not Smooth-L1) is intentional and explicitly stated
  for orthogonality vs. the primary Smooth-L1 joint loss.
- [x] Expected outputs correctly described: FOUR loss keys
  (`loss/joints/train`, `loss/depth/train`, `loss/uv/train`,
  `loss/bone_length/train`); `MetricsCSVHook` auto-picks up the new key
  because it follows the `loss/<name>/<split>` naming convention.
- [x] `forward()` and `predict()` explicitly unchanged. Bone-length loss
  is training-only.
- [x] Zero new learnable parameters. `self.bone_length_loss_weight` is a
  scalar attribute, not a Parameter.

## Minor observations (non-blocking)

- Reading `pred_bones` from the recovered coord tensor (not from a
  separately stored `bone_vecs`) is mathematically equivalent but makes
  the block uniform across Designs 001/002/003 — a nice abstraction
  choice that keeps the loss implementation decoupled from the forward
  kinematics routing (which differs in Design 003).
- The defensive `.to(device)` on `parent_idx` after slicing the
  `bone_parents` buffer is redundant under normal GPU training (the
  buffer already moves with `model.to(device)`) but harmless; the design
  rationale (CPU-only unit-test safety) is stated.
- The weight value `0.3` is justified by the relative scale of bone-
  length L1 vs. joint Smooth-L1 late in training (~15% relative pull).
  Reasonable midpoint in the standard `[0.1, 0.5]` range for auxiliary
  priors.

## Verdict

APPROVED — Builder can implement without guessing.

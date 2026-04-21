# Code Review Log — idea013 / design002

## 2026-04-17 — APPROVED

- Reviewer mode: code review
- Idea: idea013 (Kinematic Chain Bone-Vector Output Parameterization)
- Design: 002 (bone-vector head + auxiliary L1 bone-length loss with
  weight 0.3)
- `review-check-implementation`: PASSED.
- Files changed (per `implementation_summary.md`):
  `code/pose3d_transformer_head.py`, `code/config.py`. Both allowed.
- Head file is byte-identical to Design 001's head (confirmed via
  `diff`) — a shared implementation across idea013, as intended by the
  design.
- Config differs from Design 001 only in `output_dir` (setup-design
  patching) and `bone_length_loss_weight=0.3`.
- Invariant files byte-identical to baseline: `pelvis_utils.py`,
  `train.py`. Verified.
- Auxiliary bone-length loss block correctness:
  - Guard `kinematic_parametrization and bone_length_loss_weight > 0.0`
    is True under this config — block executes.
  - Indices: `child_idx = torch.arange(1, 22, device=device)`,
    `parent_idx = self.bone_parents[1:22].to(device)`.
  - Bone vectors computed from recovered coords:
    `pred_bones = pred_body[:, child_idx] - pred_body[:, parent_idx]`,
    `gt_bones = gt_body[:, child_idx] - gt_body[:, parent_idx]`.
  - Magnitudes via `.norm(dim=-1)` → `(B, 21)` each.
  - L1 on magnitudes: `(pred_bone_len - gt_bone_len).abs().mean()`.
  - Key exactly `'loss/bone_length/train'`; weight multiplied
    explicitly by `self.bone_length_loss_weight` (0.3).
- Primary three loss keys unchanged; MPJPE `with torch.no_grad()` block
  unchanged; `predict()` untouched.
- Test output (SLURM 55671440):
  - Training completed 1 epoch (81 iterations) + 1 validation pass.
  - Training log iter 50 shows all four expected losses:
    `loss/joints/train=0.178940, loss/depth/train=1.545295,
    loss/uv/train=0.111114, loss/bone_length/train=0.052109`.
    Bone-length L1 magnitude ~0.17 m before weighting, ~0.05 m after
    the 0.3 scale — consistent with the design spec's early-training
    magnitude estimate.
  - Validation: `composite_val=480.39, mpjpe_body_val=434.27,
    mpjpe_pelvis_val=574.04`. No NaNs, no CUDA errors.
  - `iter_metrics.csv` lists only the three fixed CSV loss columns
    (hook limitation, not a Design 002 bug).
  - Model init: 293/293 backbone tensors loaded.
- Verdict: APPROVED.

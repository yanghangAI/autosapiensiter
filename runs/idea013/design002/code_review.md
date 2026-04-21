# Code Review — idea013 / design002

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 002 uses the same shared head implementation as Design 001,
with `bone_length_loss_weight` flipped from `0.0` to `0.3` so the
auxiliary L1 bone-length loss becomes active. The head source file is
byte-identical to Design 001's (shared across the idea); only the
config differs. The training log confirms the bone-length term is
computed, scaled, and added to the total loss with the correct key
(`loss/bone_length/train`). Implementation matches the design spec on
every required detail.

## Checks performed

- [x] `python scripts/cli.py review-check-implementation
  runs/idea013/design002` passed.
- [x] `implementation_summary.md` lists exactly two changed files
  (`code/pose3d_transformer_head.py`, `code/config.py`) — both allowed
  by the design.
- [x] `pelvis_utils.py` and `train.py` are byte-identical to
  `baseline/`; no other files touched.
- [x] Head file (`pose3d_transformer_head.py`) is byte-identical to
  Design 001's — confirmed via `diff`. Shared implementation across
  idea013 is deliberate and matches the design spec.
- [x] Config correctly differs from Design 001 only in `output_dir`
  (patched by setup-design) and `bone_length_loss_weight=0.3` — all
  other head kwargs identical.
- [x] Auxiliary bone-length loss block is present in `loss()`:
  - guard: `self.kinematic_parametrization and self.bone_length_loss_weight > 0.0`
  - `child_idx = torch.arange(1, 22, device=device)`,
    `parent_idx = self.bone_parents[1:22].to(device)`.
  - `gt_bones`, `pred_bones` computed as
    `body[:, child_idx] - body[:, parent_idx]` from the recovered
    `pred['joints']` and `gt_joints`.
  - `gt_bone_len = gt_bones.norm(dim=-1)`,
    `pred_bone_len = pred_bones.norm(dim=-1)` — correct `(B, 21)`
    shapes.
  - `L_bone_len = (pred_bone_len - gt_bone_len).abs().mean()` — plain
    L1 on magnitudes only, as specified.
  - Key is exactly `'loss/bone_length/train'` (matches the
    `loss/<name>/<split>` convention).
  - Weight is multiplied explicitly: `self.bone_length_loss_weight *
    L_bone_len` (no separate loss module — intentional per spec).
- [x] Primary `loss/joints/train`, `loss/depth/train`, `loss/uv/train`
  terms unchanged; the `with torch.no_grad():` MPJPE block is
  unchanged and reads the recovered `pred['joints'][:, _BODY]`.
- [x] `predict()` untouched (training-only aux loss, per spec).
- [x] No new learnable parameters (bone-length is a tensor op, not an
  nn.Module).
- [x] Invariant files not modified: `bedlam_metric.py`,
  `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`,
  data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`,
  `train.py` wrapper, `tools/train.py` — verified.
- [x] Test output: `test_output/slurm_test_55671440.out` shows a clean
  end-to-end epoch + validation. Training log at iter 50 emits FOUR
  loss scalars:
  `loss/joints/train: 0.178940  loss/depth/train: 1.545295
  loss/uv/train: 0.111114  loss/bone_length/train: 0.052109` —
  confirming the auxiliary loss is active and on the expected order
  (~0.05 m * 0.3 weight ≈ 0.016 contribution, consistent with the
  design's early-training magnitude estimate). Validation computed:
  `composite_val=480.39, mpjpe_body_val=434.27, mpjpe_pelvis_val=574.04`.
  No NaNs, no CUDA errors.
- [x] `iter_metrics.csv` has 81 iteration rows. The CSV shows only
  three loss columns (`loss_joints_train, loss_depth_train,
  loss_uv_train`); this is a property of `infra/metrics_csv_hook.py`'s
  fixed `_ITER_COLS` (which maps only those three loss keys) — it is
  NOT a Design 002 omission. The bone-length term is correctly
  emitted to the runner log buffer (and visible in the slurm log). If
  tracking the bone-length term in the CSV is desired, that is an
  infrastructure change to `metrics_csv_hook.py` (invariant file),
  orthogonal to this design.

## Minor observations (non-blocking)

- `MetricsCSVHook` does not pick up `loss/bone_length/train` because
  `_ITER_COLS` / `_LOSS_MAP` are hardcoded to the three baseline loss
  keys. This is expected and out of scope for this design (the hook
  is an invariant file). The bone-length loss IS active and logged in
  the slurm output.
- Weight scale (`bone_length_loss_weight=0.3`) is stored and used
  exactly as specified.

## Verdict

APPROVED — implementation matches Design 002 spec on every required
detail; test-train completed successfully with the expected four loss
keys visible in the training log and the auxiliary bone-length term
contributing at the expected magnitude.

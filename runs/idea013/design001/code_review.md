# Code Review — idea013 / design001

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 001 implements the minimal bone-vector (kinematic-chain) output
reparameterization for the 22 body joints. The single shared
`Linear(hidden_dim, 3)` is retained; its outputs are reinterpreted as
bone-translation vectors for body tokens and as direct coordinates for
hand tokens. Root-relative body joint positions are recovered by a
topologically-ordered cumulative sum inside `_forward_kinematics`, and
`joints_out.weight` is scale-initialised by `1/sqrt(21)` to match the
baseline's direct-regression variance at init. The implementation
matches the design spec exactly, with no deviations.

## Checks performed

- [x] `python scripts/cli.py review-check-implementation
  runs/idea013/design001` passed.
- [x] `implementation_summary.md` lists exactly two changed files
  (`code/pose3d_transformer_head.py`, `code/config.py`) — both allowed
  by the design.
- [x] `pelvis_utils.py` and `train.py` are byte-identical to
  `baseline/`; `custom_imports` unchanged; no other files touched.
- [x] `config.py` head kwargs correct:
  `kinematic_parametrization=True`,
  `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`,
  `bone_length_loss_weight=0.0`, `per_limb_heads=False`,
  `limb_index=None` — literals only, MMEngine-config compliant.
- [x] `__init__` adds exactly the five new kwargs specified in the
  design, placed after `loss_weight_uv` and before `init_cfg`; defaults
  preserve baseline behaviour bit-for-bit.
- [x] Topological-ordering assertion (`parent[child] < child` for
  `child ∈ {1..21}`) is present and correctly enforced.
- [x] `bone_parents` is registered as a non-persistent long-tensor
  buffer; `self._bone_parents_list` is a Python `list[int]` for
  host-side indexing (avoids per-iteration GPU syncs).
- [x] `_forward_kinematics` clones input, zeroes the root, and writes
  into distinct slots via a `for child in range(1, 22)` loop — matches
  design spec exactly.
- [x] `_init_head_weights` scales `self.joints_out.weight` in-place by
  `1.0 / math.sqrt(21)` inside `torch.no_grad()` when
  `kinematic_parametrization=True`. Bias untouched (already zero).
- [x] `forward()` inserts the kinematic-recovery block after
  `joints = self.joints_out(decoded)` (in the `else` branch, since
  `per_limb_heads=False`): splits body `[0:22]` / hand
  `[22:num_joints]`, applies `_forward_kinematics` to body, concats
  `[body_rr, hand_coords]` on `dim=1`.
- [x] `loss()` body term reads `pred['joints'][:, _BODY]` (recovered
  coords); optional bone-length block is correctly gated by
  `kinematic_parametrization and bone_length_loss_weight > 0.0` — for
  Design 001 the guard is false and the block is skipped.
- [x] `predict()` is untouched.
- [x] Invariant files not modified: `bedlam_metric.py`,
  `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`,
  data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`,
  `train.py` wrapper, `tools/train.py` — verified via per-file diff.
- [x] Test output: `test_output/slurm_test_55671439.out` shows a clean
  end-to-end epoch + validation. Training log emits exactly the THREE
  expected loss scalars (`loss/joints/train`, `loss/depth/train`,
  `loss/uv/train`) with no `bone_length` term (correct for Design 001).
  `iter_metrics.csv` has 81 iteration rows (the configured 1-epoch
  reduced test-train); `metrics.csv` has 1 epoch row
  (`epoch=1, composite_val=475.90`). No NaNs, no CUDA errors, no
  autograd warnings beyond the benign `scheduler.step()` warmup notice
  seen in baseline runs. Model initialises successfully (293/293
  backbone tensors loaded, head randomly initialised).

## Minor observations (non-blocking)

- The scale-init of `joints_out.weight` is global (applies to all
  70 rows), which slightly attenuates the hand direct-regression init
  by `1/sqrt(21)`. The design spec explicitly acknowledges this and
  treats it as acceptable because hand outputs receive no supervised
  signal. Consistent with the spec.
- Loss CSV (`iter_metrics.csv`) lists only the three fixed loss
  columns; this is a property of `MetricsCSVHook._ITER_COLS`, not a
  Design 001 issue. (Relevant when reading Design 002's CSV.)

## Verdict

APPROVED — implementation matches Design 001 spec on every required
detail; test-train completed successfully with expected loss keys,
tensor shapes, and validation outputs.

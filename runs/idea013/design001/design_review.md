# Design Review — idea013 / design001

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 001 specifies the minimal bone-vector (kinematic-chain) output
reparameterization for the 22 body joints. The single shared
`Linear(hidden_dim, 3)` head is retained but its outputs are reinterpreted
as bone-translation vectors for the 22 body tokens and as direct
coordinates for the 48 hand tokens. Root-relative body joint positions
are recovered via a topologically-ordered cumulative-sum
`_forward_kinematics` along the SMPL-X 22-parent chain. The
`joints_out.weight` is in-place scaled by `1/sqrt(21)` to match the
baseline's direct-regression init variance. No extra learnable
parameters are introduced. The design is complete, explicit, and
implementable without guesswork.

## Checklist

- [x] `**Design Description:**` present and precise.
- [x] Starting point declared: `baseline/`.
- [x] Only allowed files are modified: `pose3d_transformer_head.py` and
  `config.py`. `pelvis_utils.py` is explicitly unchanged.
- [x] Exact `__init__` signature change: FIVE new kwargs
  (`kinematic_parametrization: bool = False`, `bone_parents: list = None`,
  `bone_length_loss_weight: float = 0.0`, `per_limb_heads: bool = False`,
  `limb_index: list = None`) placed after `loss_weight_uv` and before
  `init_cfg`, with defaults that reproduce baseline behaviour exactly.
- [x] Exact buffer registration spec: `bone_parents` registered as a
  non-persistent `torch.long` buffer; cached `self._bone_parents_list`
  Python int list for host-side indexing to avoid per-iteration
  host/device syncs.
- [x] Defensive assertions in `__init__`: `len(bone_parents) == 22`,
  `bone_parents[0] == -1`, `0 <= parent[child] < child` for `child ∈ {1..21}`.
- [x] `_init_head_weights` explicitly specified: after trunc-normal
  `std=0.02` init, in-place multiply `self.joints_out.weight` by
  `1/math.sqrt(21)` inside `torch.no_grad()` when
  `kinematic_parametrization=True`. Bias left at zero (scale-invariant).
- [x] `_forward_kinematics` fully specified: clone input, zero root,
  iterate `for child in range(1, 22)` using `self._bone_parents_list`
  for O(1) Python-side indexing; writes to distinct slots; autograd-safe.
- [x] `forward()` insertion point unambiguous: after
  `joints = self.joints_out(decoded)` and before pelvis depth/UV
  computations; split body slice `[0:22]` vs. hand slice
  `[22:num_joints]`; run `_forward_kinematics` on body slice; concatenate
  `[body_rr, hand_coords]` along `dim=1`.
- [x] `loss()` semantics preserved: main body loss
  `smooth_l1(pred['joints'][:, _BODY], gt_joints[:, _BODY])` unchanged
  (reads the recovered coordinate tensor). Optional bone-length block is
  gated by `kinematic_parametrization and bone_length_loss_weight > 0.0`,
  which under Design 001 (`weight=0.0`) is a no-op. The
  `with torch.no_grad():` MPJPE block is explicitly unchanged.
- [x] `predict()` explicitly unchanged.
- [x] Exact config values: `kinematic_parametrization=True`,
  `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`,
  `bone_length_loss_weight=0.0`, `per_limb_heads=False`, `limb_index=None`
  appended to the `head=dict(...)` block. MMEngine-config compliant
  (literals only, no `import` required).
- [x] Parent list matches idea012 (already validated) and the SMPL-X
  22-joint skeleton table included in design is self-consistent.
- [x] Invariants preserved: `persistent_workers=False`, body-only loss
  slice (`_BODY = list(range(0, 22))`), `custom_imports` include
  `'pose3d_transformer_head'`, absolute imports in head file, seed 2026,
  batch 4, accumulative_counts=8, LR schedule (LinearLR warmup +
  CosineAnnealingLR, `convert_to_iter_based=True`), hooks untouched.
- [x] Invariant files not modified: `bedlam_metric.py`,
  `bedlam2_dataset.py`, `bedlam2_transforms.py`, backbone
  (`sapiens_rgbd.py`), data preprocessor, `infra/constants.py`,
  `infra/metrics_csv_hook.py`, `train.py` wrapper, `tools/train.py`.
- [x] Edge cases handled: root-slot overwrite `body_rr[:, 0, :] = 0.0`
  ensures prediction exactly matches `SubtractRootJoint` GT; `clone()`
  prevents in-place mutation of the upstream `joints` view;
  topologically-ordered loop guaranteed correct by `parent[child] < child`
  assertion; no NaN / device-sync / autograd hazards.
- [x] Expected outputs correctly described: same THREE loss keys as
  baseline (`loss/joints/train`, `loss/depth/train`, `loss/uv/train`);
  `(B, 70, 3)` joints tensor shape; zero new learnable parameters
  (`bone_parents` is a buffer, not a Parameter).

## Minor observations (non-blocking)

- The optional bone-length block appears in `loss()` inside a
  `> 0.0` guard so Design 001's `bone_length_loss_weight=0.0`
  configuration cleanly skips it. This is deliberate shared-file
  compatibility with Designs 002/003 and matches the pattern used in
  idea012.
- The design notes that scaling `joints_out.weight` also slightly
  attenuates the hand direct-regression init by `1/sqrt(21)`. Since the
  hand head shares rows with the body head and hands receive no
  supervised signal (loss restricted to body), this is benign and
  documented. Acceptable.

## Verdict

APPROVED — Builder can implement without guessing.

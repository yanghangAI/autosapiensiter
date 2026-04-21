# Code Review Log — idea013 / design001

## 2026-04-17 — APPROVED

- Reviewer mode: code review
- Idea: idea013 (Kinematic Chain Bone-Vector Output Parameterization)
- Design: 001 (minimal bone-vector head, single shared Linear, no
  bone-length auxiliary)
- `review-check-implementation`: PASSED.
- Files changed (per `implementation_summary.md`):
  `code/pose3d_transformer_head.py`, `code/config.py`. Both are the
  only files the design allowed to change.
- Invariant files byte-identical to baseline: `pelvis_utils.py`,
  `train.py`, `custom_imports` list. Verified via per-file diff.
- Head implementation spot-checks:
  - Five new `__init__` kwargs added in the exact specified order,
    inserted after `loss_weight_uv` and before `init_cfg`; defaults
    reproduce baseline behaviour.
  - `bone_parents` validated (len 22, root=-1, topological
    `parent[child] < child`) and registered as non-persistent long
    buffer; `self._bone_parents_list` cached as Python int list.
  - `_forward_kinematics` clones, zeroes root, writes to distinct slots
    via `for child in range(1, 22)` — autograd-safe, no sync.
  - `_init_head_weights` scales `joints_out.weight` in-place by
    `1/math.sqrt(21)` under `torch.no_grad()` when
    `kinematic_parametrization=True`; bias unchanged.
  - `forward()` else branch applies body/hand split + kinematic
    recovery correctly; final shape `(B, 70, 3)` preserved.
  - `loss()` main body term unchanged (reads recovered coords);
    optional bone-length block properly guarded by `> 0.0` so Design
    001 skips it.
  - `predict()` unchanged.
- Config spot-checks:
  - `kinematic_parametrization=True`,
    `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`,
    `bone_length_loss_weight=0.0`, `per_limb_heads=False`,
    `limb_index=None` — all literals, MMEngine-compliant.
  - Optimizer, LR schedule, data pipeline, hooks, seed, batch size,
    accumulation all unchanged from baseline.
- Test output (SLURM 55671439):
  - Training completed 1 epoch (81 iterations) + 1 validation pass.
  - Training log emits exactly THREE loss scalars
    (`loss/joints/train`, `loss/depth/train`, `loss/uv/train`) — no
    `bone_length` key (expected for Design 001).
  - `metrics.csv`: 1 row (`epoch=1, composite_val=475.90,
    mpjpe_body_val=426.62, mpjpe_pelvis_val=575.97`). Values are
    high-but-reasonable for a 1-epoch reduced test-train and match
    Design 001's expected behaviour (`pred['joints']` recovered via
    forward kinematics at init has similar scale to baseline).
  - `iter_metrics.csv`: 81 iteration rows, three loss columns, values
    in the expected range (joints ~0.17–0.20, depth trending down from
    ~2.0 to ~0.7, uv stable).
  - No NaNs, no CUDA errors, no new warnings.
  - Model init: 293/293 backbone tensors loaded.
- Verdict: APPROVED.

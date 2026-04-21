# Design Review — idea013 / design003

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 003 is the kinematic-chain reparameterization (shared with
Designs 001/002) plus a decoupled output projection: five per-limb
`Linear(hidden_dim, 3)` heads (spine, left_leg, right_leg, left_arm,
right_arm) that each produce bone vectors for their assigned body
tokens. A fixed 22-long `limb_index` list routes each body token to its
limb head via pre-registered `_limb_idx_{0..4}` buffers, using
`index_select` for the gather and `index_copy_` for the scatter. Hand
tokens continue to pass through the original shared `joints_out` head.
Each per-limb head's weights are trunc-normal init'd and then in-place
scaled by `1/sqrt(21)`. The design is complete, explicit, and
implementable without guesswork.

## Checklist

- [x] `**Design Description:**` present and precise.
- [x] Starting point declared: `baseline/`.
- [x] Only allowed files are modified: `pose3d_transformer_head.py` and
  `config.py`. `pelvis_utils.py` is explicitly unchanged.
- [x] `__init__` signature matches Designs 001/002 (same five new
  kwargs, same placement, same defaults). Plus:
  - `assert per_limb_heads => kinematic_parametrization`.
  - `assert limb_index is not None and len(limb_index) == 22`.
  - `assert num_limbs == 5`; element-wise `0 <= val < num_limbs`.
  - Construct `self.body_limb_heads = nn.ModuleList([nn.Linear(
    hidden_dim, 3) for _ in range(num_limbs)])`.
  - Register `limb_index` as non-persistent long buffer.
  - Build `self._limb_token_lists` (Python list of list[int]).
  - Pre-register 5 non-persistent long buffers
    `self._limb_idx_0, ..., self._limb_idx_4` for on-device advanced
    indexing with no per-forward allocation.
- [x] `_init_head_weights` explicitly updated: trunc-normal `std=0.02`
  on each per-limb head; then in-place multiply each `body_limb_heads[k]
  .weight` AND the shared `joints_out.weight` by `1/math.sqrt(21)`
  inside a single `torch.no_grad()`. Biases left at zero.
- [x] `forward()` routing explicitly specified:
  - `body_bone_vecs = decoded.new_zeros(B, 22, 3)` canvas.
  - For each `limb_id ∈ {0..4}`: gather `decoded.index_select(1, idx)`,
    apply the corresponding per-limb head, scatter via
    `body_bone_vecs.index_copy_(1, idx, bone_vecs_limb)`.
  - Hand tokens: `hand_coords = self.joints_out(decoded[:, 22:num_joints])`
    — the shared head is retained for hands.
  - Apply `self._forward_kinematics(body_bone_vecs)` on the body canvas,
    concatenate `[body_rr, hand_coords]` along `dim=1` → `(B, 70, 3)`.
  - `else` branch reproduces Designs 001/002 code path exactly.
- [x] `_forward_kinematics` unchanged from Design 001.
- [x] `loss()` unchanged: three baseline loss keys; bone-length
  auxiliary is disabled (`bone_length_loss_weight=0.0`) to isolate the
  architectural effect of per-limb heads.
- [x] `predict()` explicitly unchanged.
- [x] Exact config values: `kinematic_parametrization=True`,
  `bone_parents=[-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]`,
  `bone_length_loss_weight=0.0`, `per_limb_heads=True`,
  `limb_index=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 3, 4, 0, 3, 4, 3, 4, 3, 4]`.
  Literals only — MMEngine-config compliant.
- [x] `limb_index` mapping self-verified in the design: per-limb counts
  (spine 6, left_leg 4, right_leg 4, left_arm 4, right_arm 4) sum to 22
  with each body index in exactly one group. Design also suggests an
  optional `set().union(*_limb_token_lists) == set(range(22))` sanity
  check.
- [x] Invariants preserved: `persistent_workers=False`, body-only loss
  slice, `custom_imports` include `'pose3d_transformer_head'`, absolute
  imports in head file, seed 2026, batch 4, accumulative_counts=8, LR
  schedule unchanged, hooks untouched.
- [x] Invariant files not modified: `bedlam_metric.py`,
  `bedlam2_dataset.py`, `bedlam2_transforms.py`, backbone, data
  preprocessor, `infra/*`, `train.py`, `tools/train.py`.
- [x] Edge cases handled: `per_limb_heads=True` without
  `kinematic_parametrization=True` fails fast via assertion; pelvis
  (index 0) assigned to `spine`, but its slot is overwritten to zero by
  `_forward_kinematics`, so the choice is a formality; `index_copy_` is
  autograd-safe because the destination is freshly allocated via
  `new_zeros`; `decoded` is not mutated by gather/scatter; pelvis
  depth/UV still reads token-0 embedding `decoded[:, 0, :]` — unchanged.
- [x] Expected outputs correctly described: THREE loss keys (same as
  baseline); `(B, 70, 3)` joints tensor shape; extra learnable
  parameters = 4 × (256 × 3 + 3) = 3084 floats (< 0.001% of model) —
  correctly attributed as "four *additional* heads beyond the baseline's
  single shared head."

## Minor observations (non-blocking)

- The design keeps `self.joints_out` in place and uses it exclusively
  for the 48 hand tokens under the `per_limb_heads=True` branch. Under
  this branch the shared head's body rows are effectively unused but
  still receive the `1/sqrt(21)` scale — harmless; matches Design 001's
  scale-init for code-path uniformity.
- Pre-registered limb index buffers (`self._limb_idx_{0..4}`,
  non-persistent) remove the per-forward `torch.tensor(...)` allocation
  that naive implementations would incur. Good performance hygiene.
- The design accepts a slight over-damping from the uniform
  `1/sqrt(21)` scale-init for per-limb heads (the true per-joint depth
  maxes out around 7, not 21) and documents that the LR warmup absorbs
  this within 3 epochs. Conservative but acceptable.
- Bone-length auxiliary is deliberately disabled in Design 003 to
  isolate the architectural effect of per-limb heads; a future design
  could combine them.

## Verdict

APPROVED — Builder can implement without guessing.

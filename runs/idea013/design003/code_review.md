# Code Review — idea013 / design003

**Verdict:** APPROVED

**Reviewed:** 2026-04-17

---

## Summary

Design 003 adds the five decoupled per-limb `Linear(hidden_dim, 3)`
body-bone-vec heads on top of the Design 001 kinematic parametrization.
The head source file is byte-identical to Designs 001/002 (shared
implementation across idea013); the per-limb branch is activated by
the config flag `per_limb_heads=True` with the correct 22-long
`limb_index` list. Routing in `forward()` uses `index_select` / (out-
of-place) `index_copy` to scatter per-limb outputs back into a 22-slot
body canvas, which is then recovered via `_forward_kinematics`. Every
required design detail is present.

## Checks performed

- [x] `python scripts/cli.py review-check-implementation
  runs/idea013/design003` passed.
- [x] `implementation_summary.md` lists exactly two changed files
  (`code/pose3d_transformer_head.py`, `code/config.py`) — both allowed.
- [x] `pelvis_utils.py` and `train.py` are byte-identical to
  `baseline/`.
- [x] Head file is byte-identical to Designs 001 and 002.
- [x] Config correctly differs from Design 001 only in `output_dir`,
  `per_limb_heads=True`, and
  `limb_index=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 3, 4, 0, 3, 4, 3, 4, 3, 4]` —
  all other head kwargs match. All literals (MMEngine-compliant).
- [x] `__init__` per-limb branch:
  - Asserts `kinematic_parametrization` is True before allowing
    `per_limb_heads`. Present and correct.
  - Asserts `len(limb_index) == 22`, all values in `[0, num_limbs)`,
    `num_limbs == 5`. Present.
  - Asserts the union of `self._limb_token_lists` equals
    `set(range(22))` — the 22 body tokens are covered exactly once.
    Present.
  - `self.body_limb_heads = nn.ModuleList([nn.Linear(hidden_dim, 3)
    for _ in range(5)])` — separate heads per limb as required.
  - `self.limb_index` registered as non-persistent long buffer;
    `self._limb_token_lists` cached as Python `list[list[int]]`.
  - Five per-limb index tensors registered as non-persistent buffers
    `_limb_idx_0` .. `_limb_idx_4` — avoids per-forward allocation as
    specified.
- [x] `_init_head_weights` branch for per-limb heads:
  - Trunc-normal `std=0.02` init for each `m` in
    `self.body_limb_heads`, bias zeroed.
  - In-place scaling `m.weight.mul_(1/sqrt(21))` under
    `torch.no_grad()` for each per-limb head AND the shared
    `self.joints_out.weight` (matches spec's shared-consistency rule).
- [x] `forward()` per-limb branch:
  - Allocates `body_bone_vecs = decoded.new_zeros(B, 22, 3)`.
  - Iterates the five `_limb_token_lists`; for each, `idx =
    getattr(self, f'_limb_idx_{limb_id}')`, `sel =
    decoded.index_select(1, idx)`, `bone_vecs_limb =
    self.body_limb_heads[limb_id](sel)`, then scatters back via
    `body_bone_vecs = body_bone_vecs.index_copy(1, idx,
    bone_vecs_limb)`.
    - NOTE: implementation uses the out-of-place `index_copy` (with
      reassignment) rather than the in-place `index_copy_`. Both are
      autograd-safe; the design spec explicitly allows this variant
      (the alternative is called out as a clean autograd choice). No
      correctness concern.
  - Hand tokens: `hand_decoded = decoded[:, 22:self.num_joints, :]`;
    `hand_coords = self.joints_out(hand_decoded)`.
  - Body path: `body_rr = self._forward_kinematics(body_bone_vecs)`.
  - Final: `joints = torch.cat([body_rr, hand_coords], dim=1)` →
    `(B, 70, 3)`.
- [x] `self.joints_out` is NOT deleted; it continues to serve hand
  tokens as required. The shared head's scale-init does not disrupt
  hands (they have no supervised signal).
- [x] `_forward_kinematics` unchanged from Design 001 — same clone,
  zero-root, topological loop.
- [x] `loss()` body term unchanged; bone-length block guarded by
  `> 0.0` so Design 003 (`weight=0.0`) skips it. Same THREE loss keys
  as baseline.
- [x] `predict()` untouched.
- [x] Parameter count increase: `4 × (256 × 3 + 3) = 3084` additional
  floats (four extra limb heads vs. baseline's single shared head).
  Negligible (<0.001% of total).
- [x] Invariant files not modified: `bedlam_metric.py`,
  `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`,
  data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`,
  `train.py` wrapper, `tools/train.py` — verified.
- [x] Test output: `test_output/slurm_test_55671441.out` shows a clean
  end-to-end epoch + validation. Training log iter 50:
  `loss/joints/train=0.175575, loss/depth/train=1.485580,
  loss/uv/val=0.110019, grad_norm=8.62` — exactly THREE loss scalars
  (no bone-length term as expected). Validation:
  `composite_val=422.97, mpjpe_body_val=354.16,
  mpjpe_pelvis_val=562.68, mpjpe_hand_val=480.25, mpjpe_abs_val=771.41`
  — notably better than Designs 001/002 at the 1-epoch checkpoint
  (consistent with the per-limb head specialisation expectation).
  No NaNs, no CUDA errors. `iter_metrics.csv` 81 rows; `metrics.csv`
  1 row. Model init: 293/293 backbone tensors loaded.

## Minor observations (non-blocking)

- Implementation uses `Tensor.index_copy` (out-of-place) with
  reassignment rather than the suggested `Tensor.index_copy_` (in-
  place). Both are autograd-safe; the out-of-place variant avoids any
  version-counter concerns and is a defensible choice.
- Five per-limb index buffers (`_limb_idx_0..4`) correctly registered
  as non-persistent. Device movement handled by PyTorch.
- Scale-init applied consistently to `joints_out.weight` and all five
  per-limb heads, per spec.

## Verdict

APPROVED — implementation matches Design 003 spec on every required
detail; test-train completed successfully with expected loss keys,
correct tensor shapes, and sensible 1-epoch validation metrics.

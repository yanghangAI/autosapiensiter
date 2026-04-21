# Code Review Log — idea013 / design003

## 2026-04-17 — APPROVED

- Reviewer mode: code review
- Idea: idea013 (Kinematic Chain Bone-Vector Output Parameterization)
- Design: 003 (five decoupled per-limb `Linear(hidden_dim, 3)` body-
  bone-vec heads + kinematic recovery; no bone-length aux)
- `review-check-implementation`: PASSED.
- Files changed (per `implementation_summary.md`):
  `code/pose3d_transformer_head.py`, `code/config.py`. Both allowed.
- Head file is byte-identical to Designs 001/002 (shared
  implementation) — per-limb behaviour is activated via config flag.
- Config diff vs. Design 001: `output_dir` (setup-design),
  `per_limb_heads=True`, `limb_index=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1,
  2, 0, 3, 4, 0, 3, 4, 3, 4, 3, 4]`. `bone_length_loss_weight=0.0` as
  specified.
- Invariant files byte-identical to baseline: `pelvis_utils.py`,
  `train.py`. Verified.
- Implementation correctness spot-checks:
  - `__init__` per-limb branch: asserts `kinematic_parametrization`,
    `len(limb_index) == 22`, `num_limbs == 5`, values in range, and
    full coverage of `set(range(22))`. All present.
  - `self.body_limb_heads` is a `nn.ModuleList` of 5
    `Linear(hidden_dim, 3)`.
  - `self.limb_index` registered as non-persistent long buffer;
    `self._limb_token_lists` cached as Python nested list; five
    `_limb_idx_{i}` non-persistent buffers registered to avoid
    per-forward allocation.
  - `_init_head_weights`: trunc-normal `std=0.02` on each per-limb
    head + bias zero; then `mul_(1/sqrt(21))` under `torch.no_grad()`
    on both the five limb heads AND `self.joints_out.weight`.
  - `forward()` per-limb branch: allocates `body_bone_vecs` canvas via
    `new_zeros`, iterates the five limbs via `index_select` +
    per-limb head + `index_copy` scatter (out-of-place, reassigned —
    autograd-safe). Hand tokens routed through `self.joints_out`.
    Body canvas passed through `_forward_kinematics`; final
    concatenation `[body_rr, hand_coords]` along `dim=1`.
  - `self.joints_out` retained for hand tokens; not deleted.
  - `_forward_kinematics` identical to Design 001.
  - `loss()` body term unchanged; bone-length block guarded by
    `> 0.0` so Design 003 skips it.
  - `predict()` untouched.
- Parameter count change: +3084 trainable floats (four extra
  `Linear(256, 3)` heads). Negligible vs. total.
- Test output (SLURM 55671441):
  - Training completed 1 epoch (81 iterations) + 1 validation pass.
  - Training log iter 50 shows THREE loss scalars (no bone-length key
    — correct for Design 003).
  - Validation: `composite_val=422.97, mpjpe_body_val=354.16,
    mpjpe_pelvis_val=562.68, mpjpe_rel_val=484.67,
    mpjpe_hand_val=480.25, mpjpe_abs_val=771.41` — 1-epoch numbers are
    notably better than Designs 001/002 at the same checkpoint,
    consistent with the per-limb specialisation expectation.
  - No NaNs, no CUDA errors, no additional warnings.
  - `iter_metrics.csv` 81 rows; `metrics.csv` 1 row.
  - Model init: 293/293 backbone tensors loaded.
- Verdict: APPROVED.

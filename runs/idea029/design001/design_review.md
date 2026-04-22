# Design Review — idea029/design001

**Verdict: APPROVED**

---

## Checklist

### Feasibility and Completeness

- **Starting point specified:** `baseline/` — correct.
- **Files changed:** `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py` — all within the allowed set.
- **Invariant files untouched:** evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, train.py wrapper — none modified.

### `pelvis_utils.py`

- New function `recover_abs_joints_batched` is fully specified with exact code.
- Mirrors the structure of `compute_mpjpe_abs` exactly; no `.detach()`, `.norm()`, or `* 1000.0` — gradients preserved.
- Uses already-imported `np`, `torch`, and the existing `recover_pelvis_3d` function — no new imports needed.
- Insertion point: after `compute_mpjpe_abs`, at end of file — unambiguous.

### `pose3d_transformer_head.py`

- **Import:** `from pelvis_utils import recover_abs_joints_batched as _recover_abs_joints_batched` — placed after the existing `compute_mpjpe_abs` import at line 36. Unambiguous.
- **`__init__` signature:** Four new kwargs with defaults specified (`abs_joint_loss_weight=0.0`, `abs_joint_indices=22`, `abs_joint_axis_weights=None`, `abs_joint_pelvis_grad_scale=1.0`). Placement after `loss_weight_uv: float = 1.0,` before `init_cfg` — unambiguous given the baseline code.
- **`__init__` body:** Storage of all four attributes; `register_buffer` for `abs_axis_weights` when not None, else set to `None`. Placement after `self.loss_weight_uv = loss_weight_uv` — unambiguous.
- **`loss()` block:** Inserted after `losses['loss/uv/train'] = ...` line, before `with torch.no_grad():` block. The block is self-contained and correctly reuses already-extracted `gt_joints`, `gt_depth`, `gt_uv` tensors — no re-extraction from `batch_data_samples`. Loss key `loss/abs_joints/train` is a differentiable tensor. `_BODY` and other existing lines unchanged.
- **Design 001 path:** `abs_joint_pelvis_grad_scale=1.0` (not < 1.0) → `else` branch executes (no detach). `abs_joint_axis_weights=None` → no per-axis weighting. Both paths correctly handled by the shared block.
- **Constraint on `predict()`:** Not modified. Confirmed.
- **Smooth-L1 implementation:** `beta=0.05`, element-wise, matches `SoftWeightSmoothL1Loss` convention used in baseline. No external dependency needed (inlined formula).

### `config.py`

- Two kwargs added: `abs_joint_loss_weight=0.5`, `abs_joint_indices=22`. All float/int literals — no Python imports. MMEngine config constraint satisfied.
- Full head dict shown with all existing kwargs preserved.
- `abs_joint_axis_weights` and `abs_joint_pelvis_grad_scale` are omitted from config → defaults to `None` and `1.0` in `__init__` respectively — correct baseline behaviour for this design.

### Invariants

- `persistent_workers=False`, seed `2026`, batch 4/accum 8: unchanged.
- Joint loss restricted to `_BODY = list(range(0, 22))`: unchanged.
- `_compute_mpjpe_abs` inside `with torch.no_grad():`: unchanged.
- AMP via `FixedAmpOptimWrapper`, `resume=True`, `max_keep_ckpts=1`: unchanged.

### Edge Cases

- `gt_depth` shape `(B, 1)` and `gt_uv` shape `(B, 2)`: sliced as `[i:i+1]` → `(1, 1)` and `(1, 2)` — both valid inputs to `recover_pelvis_3d`.
- `abs_axis_weights` buffer device placement: handled automatically by `register_buffer` + `.to(device)`.
- AMP safety: smooth-L1 values bounded, no overflow risk.
- Gradient through `recover_pelvis_3d`: fully differentiable (arithmetic only).

### No Issues Found

The Builder can implement this without any guessing. All insertion points, exact code, config values, and constraints are unambiguously specified.

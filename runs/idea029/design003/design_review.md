# Design Review — idea029/design003

**Verdict: APPROVED**

---

## Checklist

### Feasibility and Completeness

- **Starting point specified:** `baseline/` — correct.
- **Files changed:** `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py` — all within the allowed set.
- **Invariant files untouched:** none modified.

### `pelvis_utils.py`

- Identical to Designs 001/002: `recover_abs_joints_batched` added at end of file with exact code. No issues.

### `pose3d_transformer_head.py`

- Identical to Designs 001/002 — same code block. Stated explicitly.
- **Design 003 path:** `abs_joint_pelvis_grad_scale=0.5 < 1.0` → `if` branch executes.
  - Two calls to `_recover_abs_joints_batched`: first with `pred['pelvis_depth']` and `pred['pelvis_uv']` (full gradient); second with `.detach()` versions (no pelvis gradient).
  - `pred_abs = 0.5 * pred_abs_full + 0.5 * pred_abs_det`
  - Gradient to `pred_joints_rel`: 0.5×1 + 0.5×1 = 1.0 (full) ✓
  - Gradient to pelvis heads: 0.5×1 + 0 = 0.5 (half) ✓
  - Gradient analysis in design.md is correct.
- Second call discards `gt_abs` (returned as `_`) — correct since `gt_abs` is identical between the two calls.
- `abs_joint_axis_weights=None` (not in config) → `self.abs_axis_weights = None` → per-axis weighting line does not execute. Correct for this design.

### `config.py`

- Two kwargs added: `abs_joint_loss_weight=0.5`, `abs_joint_indices=22`, `abs_joint_pelvis_grad_scale=0.5`. All float/int literals. No Python imports. MMEngine config constraint satisfied.
- Full head dict shown with all existing kwargs preserved.
- `abs_joint_axis_weights` omitted from config → defaults to `None` — correct for this design.

### Invariants

Same as Designs 001/002 — all preserved.

### Edge Cases

- Both `_recover_abs_joints_batched` calls use the same `gt_joints`, `gt_depth`, `gt_uv` tensors already assembled in `loss()` — no re-extraction. Confirmed.
- The `.detach()` in the second call is applied to `pred['pelvis_depth']` and `pred['pelvis_uv']` before passing to the helper — the helper itself has no `.detach()` inside, so this correctly stops gradient only to the pelvis branch.

### No Issues Found

The Builder can implement this without any guessing. Differences from Designs 001/002 are limited to one additional config kwarg (`abs_joint_pelvis_grad_scale=0.5`) that activates the pre-written `if` branch. All insertion points, exact code, and gradient semantics are unambiguously specified.

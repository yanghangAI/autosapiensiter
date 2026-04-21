# Design Review — idea005/design003

**Verdict: APPROVED**

---

## Review Summary

Design 003 extends Design 002 by adding a fixed `joint_loss_scale: float = 1.0` constructor parameter, set to 2.0 in config, to bias joint regression proportionally to the composite metric (0.67 joint / 0.33 pelvis). The depth and UV tasks still receive learnable uncertainty weights via `uncertainty_pelvis_only=True`. `joint_loss_scale` is a pure Python float multiplier, not a learnable parameter.

---

## Checklist

### Feasibility
- Changes confined to `pose3d_transformer_head.py` and `config.py`. No invariant files touched.
- `joint_loss_scale` is a plain float scalar stored as `self.joint_loss_scale`. No new `nn.Parameter` needed. No imports required.
- `nn.Parameter` registrations for `log_var_depth` and `log_var_uv` are identical to Design 002. No conflict.
- Config entries (`uncertainty_pelvis_only=True`, `joint_loss_scale=2.0`) are plain literals. Compliant with MMEngine no-imports constraint.

### Completeness
- Starting point: `baseline/` — explicit.
- Files to modify: `pose3d_transformer_head.py` (full `__init__` signature with all three designs' parameters, and full loss block) and `config.py` (exact snippet). `pelvis_utils.py` unchanged — explicit.
- The full `__init__` signature includes `use_uncertainty_weighting`, `uncertainty_pelvis_only`, and `joint_loss_scale` — covering the combined state of all three designs. Builder has no ambiguity.
- The loss block is fully written out: `raw_joints = self.joint_loss_scale * self.loss_joints_module(...)` with the conditional uncertainty block below it.
- Explicit key points: `joint_loss_scale` applies before the conditional (affects both branches), `_train_mpjpe` is NOT scaled by `joint_loss_scale`.

### Explicitness
- `joint_loss_scale` default: 1.0 (config sets it to 2.0) — explicit.
- `joint_loss_scale` is NOT an `nn.Parameter` — explicitly stated ("a scalar Python float — not a learnable parameter").
- `log_var_depth` and `log_var_uv` init: `torch.zeros(1)` — explicit.
- Clamp range: `[-4.0, 4.0]` — explicit.
- `_train_mpjpe` unaffected by `joint_loss_scale` — explicitly called out.
- `use_uncertainty_weighting` NOT set in config (defaults to False), and warning not to set both uncertainty flags simultaneously — explicit.
- Baseline compatibility: `uncertainty_pelvis_only=False` and `joint_loss_scale=1.0` reproduce baseline exactly — explicit.
- LR for `log_var_*`: inherits full base LR — explicit.
- `persistent_workers=False`, no imports in `config.py`, absolute imports in head file — all listed.

### Implementation Readiness
- Builder can implement directly from `baseline/pose3d_transformer_head.py`. Every changed line is shown.
- The note that `joint_loss_scale` applies "before the conditional" removes any placement ambiguity in the loss block.
- The full `__init__` signature prevents any parameter ordering issues.

### Invariant Preservation
- No invariant files modified.
- No Python imports added to `config.py`.
- `_train_mpjpe` and `_train_mpjpe_abs` unchanged.
- Joint loss restricted to body joints (indices 0–21): `pred['joints'][:, _BODY]` — explicitly preserved.

### Minor Notes (non-blocking)
- The design correctly notes that `joint_loss_scale=2.0` means the initial loss (before any uncertainty adaptation) is `2 × raw_joints + 1 × raw_depth + 1 × raw_uv`. This matches the 0.67:0.33 composite weighting intent. The approximation (0.67/0.33 ≈ 2.03 → 2.0) is minor and clearly documented.
- The else-branch of the conditional still applies `joint_loss_scale` (via `raw_joints`) to `losses['loss/joints/train']`, which is correct and consistent with the key-point description.

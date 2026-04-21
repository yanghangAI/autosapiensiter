# Design Review — idea005/design002

**Verdict: APPROVED**

---

## Review Summary

Design 002 adds uncertainty weighting only to the depth and UV pelvis tasks via a `uncertainty_pelvis_only: bool` flag; joint loss is anchored at a fixed weight of 1.0. Two scalar `nn.Parameter` objects (`log_var_depth`, `log_var_uv`) are registered when the flag is True.

---

## Checklist

### Feasibility
- Changes confined to `pose3d_transformer_head.py` and `config.py`. No invariant files touched.
- Two `nn.Parameter(torch.zeros(1))` objects — standard PyTorch, registered conditionally on the flag. Correct.
- Clamping done on local variables — correct for gradient flow.
- Config uses only bool/float literals. No Python imports in `config.py`. Compliant.

### Completeness
- Starting point: `baseline/` — explicit.
- Files to modify: `pose3d_transformer_head.py` (with full `__init__` signature and loss block) and `config.py` (with exact snippet). `pelvis_utils.py` unchanged — explicit.
- The note about `use_uncertainty_weighting` potentially already present from design001 is handled: it clarifies that the two flags are independent, and the full `__init__` signature includes both.
- Joint loss assignment (`losses['loss/joints/train'] = raw_joints`) is explicit in both branches of the conditional — no ambiguity.

### Explicitness
- `log_var_depth` and `log_var_uv` init: `torch.zeros(1)` — explicit.
- Clamp range: `[-4.0, 4.0]` — explicit.
- Joint loss fixed weight: 1.0 — explicitly stated and shown in code.
- `use_uncertainty_weighting` NOT set in config (defaults to False) — explicitly stated with a warning not to set both flags simultaneously.
- `log_var` parameters inherit full LR — explicitly stated.
- Baseline compatibility via `uncertainty_pelvis_only=False` — explicit.
- `_train_mpjpe` and `_train_mpjpe_abs`: explicitly unchanged.
- `persistent_workers=False` invariant: explicitly listed.

### Implementation Readiness
- The Builder can implement this directly from `baseline/pose3d_transformer_head.py`. All constructor changes, loss block changes, and config changes are fully spelled out.
- The else-branch in the loss block explicitly writes `losses['loss/joints/train'] = raw_joints` and `losses['loss/depth/train'] = raw_depth` — no guessing required.

### Invariant Preservation
- No invariant files modified.
- No Python imports added to `config.py`.

### Minor Notes (non-blocking)
- The design correctly identifies that `raw_depth = self.loss_weight_depth * self.loss_depth_module(...)` is the input to the uncertainty formula. With `loss_weight_depth=1.0` this is a no-op multiplier, but structurally consistent with the baseline and with Design A.
- The note about sequential application with design001 is well-handled: the full combined `__init__` signature is provided, removing any builder ambiguity.

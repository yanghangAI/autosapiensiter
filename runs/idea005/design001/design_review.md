# Design Review — idea005/design001

**Verdict: APPROVED**

---

## Review Summary

Design 001 adds full uncertainty weighting (Kendall & Gal 2018) to all three loss terms (`loss/joints`, `loss/depth`, `loss/uv`) via three learnable `nn.Parameter` scalars gated by a `use_uncertainty_weighting: bool` flag.

---

## Checklist

### Feasibility
- The change is confined entirely to `pose3d_transformer_head.py` and `config.py`. No invariant files are touched.
- Using `nn.Parameter(torch.zeros(1))` for log-variance scalars is standard PyTorch; all three are registered only when the flag is True.
- Clamping to `[-4, 4]` via a local variable (not in-place) preserves gradient flow correctly.
- No Python `import` statements are added to `config.py`. The flag `use_uncertainty_weighting=True` is a plain bool literal. Compliant.

### Completeness
- Starting point: `baseline/` — explicit.
- Files to modify: `pose3d_transformer_head.py` and `config.py` — both listed, with exact diffs shown.
- `pelvis_utils.py`: explicitly listed as no-change.
- The full `__init__` signature after change is given; the Builder does not need to infer any parameter position or default.
- The exact loss computation block is spelled out, including the baseline fallback branch.

### Explicitness
- `log_var_*` init: `torch.zeros(1)` — explicit.
- Clamp range: `[-4.0, 4.0]` — explicit.
- Clamp applied to local variable — explicitly stated and the correct approach.
- `loss_weight_depth=1.0` and `loss_weight_uv=1.0` remain in config as no-ops — explicitly stated.
- `_train_mpjpe` and `_train_mpjpe_abs`: explicitly unchanged.
- `log_var` parameters are NOT under `paramwise_cfg` backbone key — explicitly stated, so they get full LR.
- Baseline compatibility via `use_uncertainty_weighting=False` — explicitly described.

### Implementation Readiness
- The Builder can implement this from `baseline/pose3d_transformer_head.py` without guessing. The constructor diff, loss block diff, and config snippet are all present and unambiguous.
- The raw loss structure in the baseline (lines 299–304) exactly matches the `raw_joints/raw_depth/raw_uv` extraction described in the design.

### Invariant Preservation
- Does not touch: `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py`, `tools/train.py`.
- `persistent_workers=False` — explicitly listed as an invariant to preserve.
- No Python imports in `config.py` — explicitly confirmed.

### Minor Notes (non-blocking)
- The design uses `raw_depth = self.loss_weight_depth * self.loss_depth_module(...)` as the input to the uncertainty formula. Since `loss_weight_depth=1.0` in this design, this is effectively `raw_depth = loss_depth_module(...)`. This is correct and consistent with the config.
- The risk of `log_var_joints` being spuriously increased is noted; this is the motivation for Designs B and C. No action needed here.

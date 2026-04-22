# Design 002 — Design Review

**Verdict:** APPROVED

**Date:** 2026-04-22 17:01 UTC

## Summary

Design 002 = Design 001 scaffolding, with log-space regression target (`log1p(depth_gt)`) and λ=0.3. Only the config flags differ; the code path already exists in Design 001's loss snippet (guarded by `self.aux_depth_log_space`).

## Checks Passed

- **Starting point:** `baseline/` — explicit.
- **Files to modify:** only `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py` — all allowed.
- **Invariants preserved:** same as Design 001. No invariant file or component is touched.
- **Algorithmic spec:** fully specified by reference to Design 001 §2a–§2d plus the log-space target substitution. Because the Design 001 code snippet already includes the `if self.aux_depth_log_space: target = torch.log1p(depth_gt)` branch, Design 002 is implemented entirely by config flag flip. The Builder needs to read Designs 001 + 002 together, which is acceptable for sibling designs within one idea.
- **Config values explicit:** `use_aux_depth=True, aux_depth_loss_weight=0.3, aux_depth_log_space=True, aux_depth_grad_weight=0.0, aux_depth_valid_min=0.1, aux_depth_valid_max=30.0, aux_depth_denorm_scale=20.0, feat_h=40, feat_w=24` — all literals.
- **Mask semantics correct:** mask computed on raw metric `depth_gt`, then applied to log-space `pred[valid]` and `log1p(depth_gt)[valid]` — this is what the Design 001 snippet already does.
- **log1p numerical safety:** `depth_gt >= 0` post-clamp, so `log1p` is finite everywhere. Documented.
- **AMP/FP16 safety:** `log1p` on values in `[0, 20]` maps to `[0, ~3.04]`, safe in FP16.
- **Zero-init correctness:** unchanged from Design 001.
- **Denorm scale and feature grid:** verified identical to Design 001.

## Verdict

APPROVED. The design is a clean hyperparameter/flag variant of Design 001 with correct rationale (log-space relative-error alignment, 3x higher λ to compensate for smaller log-space residual magnitudes). Builder can implement without guessing by combining Design 001's explicit scaffolding with Design 002's config flags.

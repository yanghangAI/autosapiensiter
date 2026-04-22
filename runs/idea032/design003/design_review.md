# Design 003 — Design Review

**Verdict:** APPROVED

**Date:** 2026-04-22 17:01 UTC

## Summary

Design 003 = Design 002 (log-space aux depth reconstruction at λ=0.3) plus a first-order spatial gradient-consistency term at inner sub-weight 0.5. The total aux loss becomes `λ * (recon_loss + 0.5 * grad_loss)`. The code path already exists in Design 001's loss snippet (guarded by `self.aux_depth_grad_weight > 0`).

## Checks Passed

- **Starting point:** `baseline/` — explicit.
- **Files to modify:** only `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py` — all allowed.
- **Invariants preserved:** same as Design 001/002. No invariant file touched.
- **Algorithmic spec:** fully specified by reference to Design 001's scaffolding (which already contains the `if self.aux_depth_grad_weight > 0:` branch) plus Design 003's explicit gradient-term code block. Log-space space-of-computation made explicit (Invariant #13). Gradient term computed unmasked over full 40x24 grid (Invariant #14). Inner sub-weight 0.5, outer λ 0.3 specified (Invariant #15).
- **Config values explicit:** `use_aux_depth=True, aux_depth_loss_weight=0.3, aux_depth_log_space=True, aux_depth_grad_weight=0.5, aux_depth_valid_min=0.1, aux_depth_valid_max=30.0, aux_depth_denorm_scale=20.0, feat_h=40, feat_w=24` — all literals.
- **Step-0 correctness:** at `pred=0`, `dx_pred=dy_pred=0`, so `grad_loss = |dx_tgt|.mean() + |dy_tgt|.mean()` is a data-dependent constant with zero gradient w.r.t. model parameters. Main losses unaffected at step 0. Correctly reasoned in the design.
- **AMP/FP16 safety:** first-order differences of log-depth tensors are bounded and FP16-safe.
- **Space consistency:** gradient term is computed on log-space tensors (same as recon term); this is explicit and correct — mixing raw and log spaces would be a scale error.
- **Denorm scale and feature grid:** verified identical to Design 001.

## Verdict

APPROVED. All details explicit. The gradient-consistency code block already appears in Design 001's loss snippet, activated by `aux_depth_grad_weight > 0`. Builder has sufficient detail to implement by combining Design 001's scaffolding with Design 003's config flags.

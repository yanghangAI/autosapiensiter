# Design Review — idea018/design002

**Verdict: APPROVED**

---

## Review Summary

Design 002 extends Design 001 with a learnable log-sigma parameter and an auxiliary smooth-L1 loss on the global depth probe output. The design is complete, unambiguous, and implementation-ready.

---

## Checklist

### Feasibility
- Learnable `log_sigma` as `nn.Parameter(torch.zeros(1))` is a clean PyTorch idiom. The forward clamp `torch.exp(self.log_sigma).clamp(min=0.01)` prevents division by zero — correct.
- Reusing `self.loss_depth_module` for the probe auxiliary loss is valid and explicitly noted; no new module needed.
- The auxiliary loss key `'loss/depth_probe/train'` appears in training logs but does not affect `composite_val` — confirmed by design.
- `gt_depth` is already computed earlier in `loss()` in the baseline — the Builder can confirm this directly from baseline code (it is).
- The `hasattr(self, '_depth_probe_z_hat')` guard is a correct safety pattern, consistent with the `_train_mpjpe` caching pattern in the baseline.

### Completeness
- Starting point: `baseline/` — specified.
- Files changed: only `pose3d_transformer_head.py` and `config.py` — within the allowed set.
- `pelvis_utils.py`: explicitly no changes.
- All five modification points covered: `_DecoderLayer.forward()`, `__init__()`, `_init_head_weights()`, `forward()` (with z_hat caching), `loss()` (auxiliary loss).
- Config additions: `depth_gate_type='gaussian_learnable_sigma'`, `depth_probe_loss_weight=0.1` — str/float literals, compliant.
- `log_sigma` is NOT zero-inited in `_init_head_weights()` — it is already initialized to `0.0` via `torch.zeros(1)` in `__init__`. The design explicitly notes this in constraint 12. Correct.

### Explicitness
- New constructor kwargs (`depth_gate_type`, `depth_probe_loss_weight`) and defaults are given.
- `self.depth_probe_loss_weight` is stored as an attribute — design explicitly says so.
- No `depth_gate_sigma` kwarg in Design 002 (sigma is learned) — consistent with the fact that Design 002 uses `gaussian_learnable_sigma` type, not `gaussian`. No confusion with Design 001's `depth_gate_sigma_buf`.
- Effective weight of auxiliary loss (0.1 × 1.0 = 0.1) is explicitly stated in constraint 10.
- `_DecoderLayer.forward()` modification is identical to Design 001 — explicitly stated.
- AMP handling: same as Design 001, valid.

### Invariant Compliance
- Invariant files not touched.
- `persistent_workers=False` preserved.
- MMEngine config no-import constraint satisfied.

### Issues Found

**Minor ambiguity (non-blocking):** The design specifies that `log_sigma` has "no custom `lr_mult` in `paramwise_cfg`" (constraint 7) — the Builder does not need to change `paramwise_cfg` in `config.py`. This is consistent with baseline's `paramwise_cfg` which applies to backbone only. Confirmed: no config change needed.

**Non-issue clarification:** The design does not include `depth_gate_sigma` as a constructor kwarg (unlike Design 001). This is intentional — sigma is learned via `log_sigma`. The Builder must NOT add `depth_gate_sigma` to Design 002's constructor. The design is explicit about this.

---

## Notes for Builder

1. Store `self.depth_probe_loss_weight = depth_probe_loss_weight` in `__init__` — needed for the conditional in `loss()`.
2. `_depth_probe_z_hat` is set in `forward()` and read in `loss()` — `loss()` always calls `forward()` first, so the attribute is always set. The `hasattr` guard is just safety.
3. Do NOT add a `depth_gate_sigma` kwarg or buffer to Design 002 — sigma is entirely learned via `log_sigma`.
4. The loss key `'loss/depth_probe/train'` must be exactly this string (slash-separated format consistent with other loss keys).

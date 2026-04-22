# Design Review — idea026/design003

**Verdict: APPROVED**

---

## Checklist

### Completeness and Explicitness

- **Design Description:** Present and accurate.
- **Starting point:** `baseline/` — explicit.
- **Files to change:** `pose3d_transformer_head.py` and `config.py` only. No invariant files touched. Correct.
- **Algorithmic changes:** Fully specified. Same as Design A (`log_scale_out: Linear(hidden_dim, 1)`, per-token, `log_scale_out_features=1`), with the addition of entropy weight annealing: `w_ent` ramps linearly from 0.1 to 1.0 over 500 gradient steps tracked via `self._loss_call_count`.
- **Config values and defaults:** All new kwargs with explicit types/defaults. Config snippet for `model.head` uses `laplace_entropy_weight_start=0.1`, `laplace_entropy_weight_end=1.0`, `laplace_entropy_anneal_steps=500`. `laplace_entropy_weight` (static) is omitted from config — design explicitly notes the `__init__` default of 1.0 applies but is never used since the annealing branch (`laplace_entropy_anneal_steps=500 > 0`) is always taken. This is unambiguous.
- **Exact `__init__` signature:** Identical to Designs A/B. `log_scale_out_features=1` for Design C.
- **Exact `forward` change:** Code block provided. Identical to Design A. `log_scale_out` applied per body query token → `(B, 22, 1)`.
- **Exact `loss` change:** Full replacement code provided. `laplace_entropy_anneal_steps=500 > 0` → annealing branch taken. `self._loss_call_count` incremented before `progress` computation (first call gives progress=1/500=0.002, w_ent≈0.1018 — acknowledged and deemed acceptable). After step 500, `w_ent` stays at 1.0 (identical to Design A). AMP clamp and scale clamp both specified.
- **`_loss_call_count` initialisation:** Set to 0 in `__init__` when `use_per_joint_uncertainty=True`. Not a registered buffer — resets on preemption resume. Design explicitly acknowledges this and deems it acceptable (gentle re-warm rather than harmful restart). The Builder need not implement buffer registration unless desired — decision is documented.
- **`predict` change:** Explicitly none.
- **`_init_head_weights` change:** Explicitly none — zero-init in `__init__`. Correct.
- **AMP safety:** `log_s.clamp(-10, 5)` before `exp` specified. Correct.
- **Scale clamp:** `s.clamp(min=1e-4)` after `exp` specified. Correct.
- **Broadcast:** `(B, 22, 1)` against `(B, 22, 3)`. Same as Design A, documented.
- **Baseline fallback:** `use_per_joint_uncertainty=False` path preserves original `loss_joints_module`. Confirmed.
- **MMEngine config compliance:** All config values are literals. No imports. Correct.
- **Invariants preserved:** `_BODY`, pelvis losses, `_train_mpjpe`, `_train_mpjpe_abs`, `predict()` output — unchanged and confirmed.
- **Metric invariance:** `log_scale` never reaches metric. Confirmed.
- **Implementation note:** Design recommends a single unified head implementation for all three designs. Consistent and reduces risk.

### Issues

None. The design is self-consistent, explicit, and implementable without guessing. The resume-reset behaviour of `_loss_call_count` is explicitly addressed and the Builder does not need to make a decision independently.

---

**The Builder can implement this design without ambiguity.**

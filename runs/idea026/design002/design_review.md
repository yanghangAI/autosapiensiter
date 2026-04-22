# Design Review — idea026/design002

**Verdict: APPROVED**

---

## Checklist

### Completeness and Explicitness

- **Design Description:** Present and accurate.
- **Starting point:** `baseline/` — explicit.
- **Files to change:** `pose3d_transformer_head.py` and `config.py` only. No invariant files touched. Correct.
- **Algorithmic changes:** Fully specified. `log_scale_out: Linear(hidden_dim, 3)` applied per body query token independently; produces `(B, 22, 3)`. Replaces SoftWeightSmoothL1 body-joint loss with per-axis Laplace NLL (element-wise, no broadcasting).
- **Config values and defaults:** All new `__init__` kwargs listed with explicit defaults preserving baseline behaviour. Config snippet is complete with all literal values. `log_scale_out_features=3` is the distinguishing value.
- **Exact `__init__` signature:** Identical structure to Design 001 with `log_scale_out_features=3`. Fully specified.
- **Exact `forward` change:** Code block provided. Applies `log_scale_out` to `decoded[:, :22, :]` (B, 22, hidden_dim) → `(B, 22, 3)`. Shape documented explicitly.
- **Exact `loss` change:** Full replacement code provided. `log_s` shape `(B, 22, 3)`, `s` shape `(B, 22, 3)`, `abs_err` shape `(B, 22, 3)` — element-wise multiplication, no broadcasting. Explicitly documented as key difference from Design A. AMP clamp and scale clamp both specified.
- **`predict` change:** Explicitly none.
- **`_init_head_weights` change:** Explicitly none — zero-init in `__init__`. Correct.
- **AMP safety:** `log_s.clamp(-10, 5)` before `exp` specified. Correct.
- **Scale clamp:** `s.clamp(min=1e-4)` after `exp` specified. Correct.
- **No broadcast:** explicitly documented — `s` and `abs_err` are both `(B, 22, 3)`, element-wise. Correct and unambiguous.
- **Baseline fallback:** `use_per_joint_uncertainty=False` path preserves original `loss_joints_module`. Confirmed.
- **MMEngine config compliance:** All config values are literals. No imports. Correct.
- **Invariants preserved:** `_BODY`, pelvis losses, `_train_mpjpe`, `_train_mpjpe_abs`, `predict()` output — unchanged and confirmed.
- **Metric invariance:** `log_scale` never reaches metric. Confirmed.
- **Implementation note:** Design explicitly states the Builder can implement a single unified head handling both `shared_scalar` and `per_axis` modes via config kwargs (D001 and D002 share the same `__init__` signature differing only in `log_scale_out_features`). Correct and reduces implementation risk.

### Issues

None. The design is self-consistent, explicit, and implementable without guessing.

---

**The Builder can implement this design without ambiguity.**

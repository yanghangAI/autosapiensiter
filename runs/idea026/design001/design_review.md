# Design Review — idea026/design001

**Verdict: APPROVED**

---

## Checklist

### Completeness and Explicitness

- **Design Description:** Present and accurate.
- **Starting point:** `baseline/` — explicit.
- **Files to change:** `pose3d_transformer_head.py` and `config.py` only. No invariant files touched. Correct.
- **Algorithmic changes:** Fully specified. New `log_scale_out: Linear(hidden_dim, 1)` applied per body query token independently (per-token, not pooled); produces `(B, 22, 1)`. Replaces SoftWeightSmoothL1 body-joint loss with Laplace NLL.
- **Config values and defaults:** All new `__init__` kwargs listed with explicit defaults that preserve baseline behaviour (`use_per_joint_uncertainty=False`). Config snippet for `model.head` is complete with all literal values.
- **Exact `__init__` signature:** Fully specified, including all new parameters and their types/defaults.
- **Exact `forward` change:** Code block provided. Applies `log_scale_out` to `decoded[:, :22, :]` (B, 22, hidden_dim) → `(B, 22, 1)`. Stores result as `pred['log_scale']`. Correct and unambiguous.
- **Exact `loss` change:** Full replacement code provided. Includes `log_s.clamp(-10, 5)`, `s.clamp(min=1e-4)`, broadcast `(B, 22, 1)` against `(B, 22, 3)`, `w_ent = self.laplace_entropy_weight` (1.0, annealing branch not taken since `laplace_entropy_anneal_steps=0`). `.mean()` for scalar loss.
- **`predict` change:** Explicitly none — acknowledged that `log_scale` key is silently ignored.
- **`_init_head_weights` change:** Explicitly none — zero-init done immediately in `__init__`. Correct.
- **AMP safety:** `log_s.clamp(-10, 5)` before `exp` specified as mandatory. Correct.
- **Scale clamp:** `s.clamp(min=1e-4)` after `exp` specified. Correct.
- **Broadcast:** `(B, 22, 1)` broadcast against `(B, 22, 3)` documented and verified correct.
- **Baseline fallback:** `use_per_joint_uncertainty=False` path preserves original `loss_joints_module` call exactly.
- **MMEngine config compliance:** All config values are bool/int/float/str literals. No Python imports in config. Correct.
- **Invariants preserved:** `_BODY`, pelvis losses, `_train_mpjpe`, `_train_mpjpe_abs`, `predict()` output structure — all unchanged and explicitly confirmed.
- **Metric invariance:** `log_scale` never reaches `BedlamMPJPEMetric`. Explicitly confirmed.

### Issues

None. The design is self-consistent, explicit, and implementable without guessing.

---

**The Builder can implement this design without ambiguity.**


---
## 2026-04-16 — Design Review

**Verdict: APPROVED**

All criteria pass. Changes confined to `pose3d_transformer_head.py` and `config.py`. Adds `joint_loss_scale: float = 1.0` constructor param (stored as plain float, not `nn.Parameter`), set to 2.0 in config. Combined with `uncertainty_pelvis_only=True` from design002. Full `__init__` signature including all three designs' parameters provided. `_train_mpjpe` explicitly not scaled by `joint_loss_scale`. Baseline reproduced when `uncertainty_pelvis_only=False` and `joint_loss_scale=1.0`. No invariant files touched.

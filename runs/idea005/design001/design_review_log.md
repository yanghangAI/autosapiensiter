
---
## 2026-04-16 — Design Review

**Verdict: APPROVED**

All three feasibility, completeness, explicitness, and implementation-readiness criteria pass. Changes confined to `pose3d_transformer_head.py` and `config.py`. No invariant files touched. `log_var_joints`, `log_var_depth`, `log_var_uv` are `nn.Parameter(torch.zeros(1))` gated by `use_uncertainty_weighting=True`; clamped to `[-4, 4]` via local variable; full LR from AdamW. Config uses plain bool literal. Baseline reproduced when flag is False.

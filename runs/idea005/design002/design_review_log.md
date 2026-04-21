
---
## 2026-04-16 — Design Review

**Verdict: APPROVED**

All criteria pass. Changes confined to `pose3d_transformer_head.py` and `config.py`. Two `nn.Parameter(torch.zeros(1))` objects (`log_var_depth`, `log_var_uv`) gated by `uncertainty_pelvis_only=True`; joint loss anchored at fixed 1.0; clamped via local variable to `[-4, 4]`; full LR. Config uses plain bool literal. Interaction with design001's `use_uncertainty_weighting` flag is explicitly handled in the combined `__init__` signature. Baseline reproduced when flag is False.

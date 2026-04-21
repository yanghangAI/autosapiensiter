# Design Review Log — idea018/design002

## 2026-04-21

**Verdict: APPROVED**

Learnable-sigma depth gate (`log_sigma` nn.Parameter init=0) + auxiliary smooth-L1 probe loss (weight=0.1) on `z_hat` vs. gt_depth. Reuses `loss_depth_module`. Caches `_depth_probe_z_hat` in forward for use in loss. Fully specified; no ambiguities; all invariants preserved.

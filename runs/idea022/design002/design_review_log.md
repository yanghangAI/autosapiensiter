## 2026-04-21 — APPROVED

**Design:** 2-layer cascaded decoder with dynamic Gaussian reprojection bias (fixed σ=4, γ=2) and auxiliary body-joint loss (weight=0.4) on layer-1 output.

**Verdict:** APPROVED

Inherits design001 structure. Key differences precisely specified: (1) intermediate layer-0 forward runs without torch.no_grad() to enable gradient flow for aux loss; (2) auxiliary loss on layer1_joints[:, _BODY] with weight 0.4, keyed 'loss/joints_aux/train'. Scoping pattern for layer1_joints across bias and aux-loss blocks is explicit. Config uses literals only. No invariant files modified.

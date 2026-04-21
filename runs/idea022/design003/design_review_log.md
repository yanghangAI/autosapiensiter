## 2026-04-21 — APPROVED

**Design:** 2-layer cascaded decoder with dynamic Gaussian reprojection bias, auxiliary loss (weight=0.4), and learnable per-joint σ and γ initialized to (4.0, 2.0).

**Verdict:** APPROVED

Inherits design002 structure. Key difference precisely specified: nn.Parameter bias_sigma (J,) and bias_gamma (J,) created only when reproj_bias_learnable=True, initialized to (4.0, 2.0). Conditional sigma/gamma computation in loss() uses F.softplus for positivity on sigma; gamma unconstrained. Cast to feat_coords.dtype for AMP compatibility. Config sets reproj_bias_learnable=True with all other values literal. No invariant files modified.

## 2026-04-21 — APPROVED

**Design:** 2-layer cascaded decoder with dynamic Gaussian reprojection bias (fixed σ=4, γ=2), no auxiliary loss.

**Verdict:** APPROVED

All files specified. Architecture correct: attn_mask shape (B*nheads, J, H'W'), _reproj_bias side-channel with proper clearing, torch.no_grad() for intermediate layer-0 forward, AMP cast applied. Feature grid orientation consistent (row-major, indexing='ij'). Config uses literals only. No invariant files modified.

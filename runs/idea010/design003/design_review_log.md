# Design Review Log — idea010/design003

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Depth-weighted reprojection loss, lambda=1.0. Extends design002 with a per-joint geometry-aware weight `w_i = (X_i / fx).detach()` applied to a `reduction='none'` smooth_l1 error; the `.detach()` is load-bearing (prevents the optimizer from cheating by shrinking X). Three new head kwargs (all defaulting to baseline behaviour), three new config values. Reduction scheme is `err.mean()` per-sample + `torch.stack(...).mean()` over the batch — equivalent to `reduction='mean'` in the unweighted case. Expected smaller numerical magnitudes are documented (no extra rescale). Fallback to design002 behaviour when `reproj_depth_weighted=False`. Changes confined to the three allowed files. No invariant violations. Implementation-ready.

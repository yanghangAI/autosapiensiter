# Design Review Log — idea010/design002

## Entry 1 — 2026-04-17

**Verdict: APPROVED**

Reprojection loss with explicit pelvis term, lambda=1.0. Extends design001 by adding a pelvis-only reprojection loss under key `'loss/reproj_pelvis/train'` (gated by `reproj_include_pelvis=True`) in addition to the body-joint reprojection under `'loss/reproj/train'`. Changes confined to the three allowed files; helper `project_joints_to_2d` identical to design001. Two new head kwargs with backward-compatible defaults. Body joint indices 0-21 only, smooth_l1 beta=0.05 reduction='mean', shared lambda, X>=0.01 clamp. No invariant violations. Redundancy between pelvis reprojection and `loss/uv/train` is acknowledged and justified by the design. Implementation-ready.

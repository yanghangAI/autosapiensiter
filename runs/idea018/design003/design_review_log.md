# Design Review Log — idea018/design003

## 2026-04-21

**Verdict: APPROVED**

22-query body-only decoder + fixed-sigma Gaussian depth gate (sigma=1.0) combined. `hand_proj: Linear(5632, 144)` with trunc-normal init. Auxiliary hand loss weight=0.1. Depth probe zero-init. Output joints always (B,70,3) via cat. Fully specified; all dynamic shapes computed via num_body_queries/num_joints; all invariants preserved.

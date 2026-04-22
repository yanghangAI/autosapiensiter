# Design Review Log — idea028/design003

## 2026-04-21 — APPROVED

Reviewer: Reviewer agent
Mode: design review

Verdict: APPROVED. Combines design001 (decoupled pelvis decoder) with 22-query body-only joint decoder. `joint_queries` embedding resized to `(22, hidden_dim)` with explicit before/after. Zero-padding to `(B, 70, 3)` specified with exact formula. Both mechanisms are orthogonal; no shared state conflicts. All invariants preserved. Builder can implement without guessing.

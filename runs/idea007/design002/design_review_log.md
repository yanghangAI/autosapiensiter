# Design Review Log — idea007 / design002

## 2026-04-16 — Reviewer (design review)

**Verdict: APPROVED**

Vertical band warm-start cross-attention routing. Extends design001 with a Gaussian anatomical prior: body-lower joints biased toward lower spatial rows (+0.5/−0.5), body-upper joints toward upper rows, hand joints at zero. Prior computation fully specified with exact constants, torch-only implementation, and correct `rounding_mode='floor'` for row index. `cross_routing_type` mapping and backward-compatibility to baseline stated. No issues found.

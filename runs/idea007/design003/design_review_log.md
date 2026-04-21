# Design Review Log — idea007 / design003

## 2026-04-16 — Reviewer (design review)

**Verdict: APPROVED**

Per-head cross-attention routing. Per-head bias `(num_heads, J, S)` expanded to `(B*num_heads, J, S)` at forward. `B` propagated from `Pose3dTransformerHead.forward`. Unified `_DecoderLayer` via `per_head_routing: bool` flag handles all three design variants from baseline. "Revised approach" section in 1c is the authoritative final signature superseding the partial signature earlier in the same section. All constraints including `_per_head`, `_num_heads` storage, `expand` vs `repeat`, and assert location are explicit. No issues found.

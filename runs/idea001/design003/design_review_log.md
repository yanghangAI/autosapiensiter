## 2026-04-16 — Design Review

**Verdict: APPROVED**

4 decoder layers + aux joint loss (weight 0.4) at 3 intermediate layers + explicit shared joints_out, pelvis losses final layer only. Design is complete, unambiguous, and implementation-ready. Loss table with 4 entries provided, OOM mitigation path specified, invariants preserved.

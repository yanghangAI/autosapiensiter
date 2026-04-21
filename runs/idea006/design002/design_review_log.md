# Design Review Log — idea006/design002

---

## Entry 1 — 2026-04-16

**Verdict: APPROVED**

Skeleton-graph-initialized `(70,70)` attention bias with `+0.5` for 69 kinematic edges (bidirectional), `-0.5` pelvis diagonal. Module-level helper `_build_skeleton_attn_bias` fully specified with hardcoded edge lists. All index ranges verified (max=69). `attn_bias_type='skeleton_init'` string literal in config.py. All invariants preserved. Builder can implement without guessing.

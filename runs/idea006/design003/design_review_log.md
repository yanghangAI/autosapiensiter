# Design Review Log — idea006/design003

---

## Entry 1 — 2026-04-16

**Verdict: APPROVED**

Per-head `(8,70,70)` zero-initialized attention bias, expanded to `(B*8,70,70)` via `.unsqueeze(0).expand(B,-1,-1,-1).reshape(B*8,J,J)`. Expand/reshape layout verified correct for `batch_first=True` per-head semantics. `attn_bias_mode` string API with `'per_head'`/`'shared'`/`'none'` fully specified. `attn_bias_type='per_head'` string literal in config.py. All invariants preserved. Builder can implement without guessing.

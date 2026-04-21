# Design Review Log — idea018/design001

## 2026-04-21

**Verdict: APPROVED**

Fixed-sigma Gaussian depth gate on cross-attention logits. Two zero-init linear probes (global + per-token). Sigma=1.0 fixed buffer. Additive logit bias to `nn.MultiheadAttention` via float `attn_mask`. Fully specified; no ambiguities; all invariants preserved.

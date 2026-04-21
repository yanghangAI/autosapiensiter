## 2026-04-21 — Design Review

**Verdict: APPROVED**

EMA per-joint difficulty weighting (alpha=1.0, linear, group-normalised upper/lower + 5-epoch warmup). Design is fully specified and implementable without guessing. Group index ranges (upper=0..12, lower=13..21) are anatomically mislabeled but the design explicitly mandates exact index ranges — unambiguous. Module-level constants, 7 init params, conditional buffers, group-normalised weight computation, warmup ramp, and loss replacement block all fully specified. Config: 6 literal kwargs. No invariant files modified. Ready for Builder.

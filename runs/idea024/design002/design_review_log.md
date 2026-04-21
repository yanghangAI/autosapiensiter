## 2026-04-21 — Design Review

**Verdict: APPROVED** (with flagged risk)

EMA per-joint difficulty weighting (alpha=1.0, softmax T=1.0). Design is fully specified and implementable without guessing. RISK FLAGGED: softmax applied to raw mm EMA values at T=1.0 will produce near-degenerate one-hot weights (hardest joint gets ≈ all weight), contrary to the design's claim of "well-calibrated" distribution. Experiment may underperform. Build proceeds as specified — risk is noted for Orchestrator.

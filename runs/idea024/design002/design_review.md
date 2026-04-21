**Verdict: APPROVED** (with a flagged risk — see Softmax Concentration Warning)

**Design:** idea024/design002 — EMA per-joint difficulty weighting (alpha=1.0, softmax T=1.0)

---

## Review Summary

The design is fully specified and implementable without guessing. All required sections are present and unambiguous. One significant mathematical risk is flagged below — it does not block the build but the Builder (and Orchestrator) should be aware.

### Completeness Check

| Section | Status |
|---|---|
| Design Description | Present |
| Starting point | `baseline/` — explicit |
| Files to change | `pose3d_transformer_head.py`, `config.py` only; `pelvis_utils.py` unchanged |
| Algorithmic changes | Fully described with exact formulas |
| Config values and defaults | All 5 kwargs listed as bool/float/float/str/float literals |
| Training/loss changes | Joint loss replacement block specified with exact code |
| Invariant preservation | Explicitly enumerated |
| Edge cases | Explicitly enumerated |

### Algorithm Correctness

- EMA update: identical to design001 (`beta=0.99`) — correct.
- Softmax normalisation: `w = 22 * softmax(ema / T)` with `T=1.0` — correctly sums to 22 (softmax output sums to 1, multiplied by 22).
- Manual smooth-L1 with `beta=0.05` — same as design001, correct.
- `w.view(1, 22, 1)` broadcast — correct.
- EMA init to `torch.ones(22)` → `softmax([1.0]*22) = [1/22]*22` → `w = [1.0]*22` — uniform at step 0, correct.

### Softmax Concentration Warning (non-blocking risk)

The design claims that `T=1.0` at "typical EMA values in the 50–400 mm range" produces "well-calibrated weights without concentrating excessively on a single joint." This claim is **mathematically incorrect**.

With 22 joints and EMA values spanning [50, 400] mm, `softmax(ema / 1.0)` is:
- `exp(400) / (exp(50) + exp(100) + ... + exp(400))` ≈ 1.0 (the exp(400) term dominates exponentially)
- The hardest joint will receive nearly all of the weight (≈ 22), and all other joints will receive near-zero weight.

This is the opposite of "well-calibrated" — it approaches a one-hot weighting scheme. In practice, training will assign essentially all joint-loss gradient to the single hardest joint, which is a degenerate regime likely to hurt overall mpjpe_body_val.

**The design is still implementable exactly as specified** — the Builder does not need to guess anything. The mathematical outcome is determined by the design's own choices. This is flagged as a design-level risk for the Orchestrator's awareness. If the experiment produces worse-than-baseline results, this is the likely cause.

If a temperature rescaling is desired to produce well-calibrated weights, a temperature on the order of the mean EMA value (e.g., `T = mean(ema) ≈ 200`) would be needed, or alternatively normalising the EMA before applying softmax: `softmax((ema - mean(ema)) / std(ema))`. The current design does neither — it applies raw mm values to softmax at T=1.0.

### Implementation Readiness

- `__init__` parameter placement (after `loss_weight_uv`, before `init_cfg`) — explicitly specified.
- Buffer registration placement — explicitly specified; same buffers as design001 (`joint_err_ema`, `_train_iter`).
- `_get_adaptive_weights`: `weight_norm='softmax'` branch uses `F.softmax`; `import torch.nn.functional as F` inside method or at module top — both accepted.
- `loss()` replacement: exact original line identified, exact replacement block provided.

### Invariant Check

- `_train_mpjpe` / `_train_mpjpe_abs`, depth/uv losses, `predict()` — unchanged.
- `per_joint_difficulty_weighting=False` → bit-identical to baseline.
- No invariant files modified.

### Config Check

- 5 literal kwargs: `per_joint_difficulty_weighting=True`, `ema_alpha=1.0`, `ema_momentum=0.99`, `weight_norm='softmax'`, `weight_temperature=1.0` — MMEngine-compliant, no import statements.

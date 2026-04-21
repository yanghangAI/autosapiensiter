**Verdict: APPROVED**

**Design:** idea024/design001 — EMA per-joint difficulty weighting (alpha=0.5, linear normalisation)

---

## Review Summary

The design is fully specified, self-consistent, and implementable without guessing. All required sections are present and unambiguous.

### Completeness Check

| Section | Status |
|---|---|
| Design Description | Present |
| Starting point | `baseline/` — explicit |
| Files to change | `pose3d_transformer_head.py`, `config.py` only; `pelvis_utils.py` unchanged |
| Algorithmic changes | Fully described with exact formulas |
| Config values and defaults | All 3 kwargs listed as bool/float/float literals |
| Training/loss changes | Joint loss replacement block specified with exact code |
| Invariant preservation | Explicitly enumerated |
| Edge cases | Explicitly enumerated |

### Algorithm Correctness

- EMA update: `ema[j] ← 0.99 * ema[j] + 0.01 * batch_err[j]` — correct, stable.
- Weights: `w[j] = (ema[j] / mean(ema))^0.5`, renormalised so sum=22 — correct gradient-scale preserving formula.
- Manual smooth-L1 with `beta=0.05`: matches baseline `SoftWeightSmoothL1Loss(beta=0.05)` — correct bypass of per-joint API limitation.
- `w.view(1, 22, 1)` broadcast over `(B, 22, 3)` — correct.
- EMA initialised to `torch.ones(22)` → uniform weights at step 0 = baseline — correct.

### Implementation Readiness

- `__init__` parameter placement (after `loss_weight_uv`, before `init_cfg`) — explicitly specified.
- Buffer registration placement (after nn layers, before `_init_head_weights()`) — explicitly specified.
- `_get_adaptive_weights` placement (between `_get_pos_enc` and `forward`) — explicitly specified.
- `loss()` replacement: exact original line identified, exact replacement block provided.
- `_train_iter` buffer incremented inside `torch.no_grad()` — correct; `dtype=torch.long` — specified.
- Duplicate `_BODY` definition handled — explicitly noted.

### Invariant Check

- `_train_mpjpe` / `_train_mpjpe_abs` unchanged — required.
- `depth` / `uv` loss lines unchanged — required.
- `predict()` unchanged — required.
- `per_joint_difficulty_weighting=False` → bit-identical to baseline — required and verified by design.
- No changes to invariant files (metric, dataset, transforms, backbone, train.py, infra) — confirmed.

### Config Check

- 3 literal kwargs: `per_joint_difficulty_weighting=True`, `ema_alpha=0.5`, `ema_momentum=0.99` — MMEngine-compliant, no import statements.

### Minor Notes (non-blocking)

- `_train_iter` buffer is registered and incremented even though warmup is not used in design001. This is harmless overhead and does not affect correctness. If the Builder wants to omit it for design001 only, they may — but including it is also correct.
- The design does not use `weight_norm` or `weight_temperature` parameters (they are not part of design001's interface). The Builder must not add them for design001, as design002 and design003 introduce them separately from their own starting points (`baseline/`).

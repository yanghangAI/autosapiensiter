**Design Review Verdict: APPROVED**

**Design ID:** idea016/design002
**Reviewer:** Reviewer Agent
**Date:** 2026-04-21

---

## Summary

Design 002 implements dual-pool FiLM conditioning (avg + max concatenated → MLP → γ, β) as Design B from idea.md. The specification is complete, explicit, and implementation-ready.

---

## Checklist

### Feasibility
- PASS. The dual-pool variant is a clean extension of design001: the only structural difference is `film_in_dim = 2 * hidden_dim = 512` and the `ctx` computation (`cat([mean, max])` instead of `mean`). The insertion point in `forward()` is identical.
- PASS. `spatial.max(dim=1).values` is explicitly called out with the `.values` attribute requirement — the Builder cannot accidentally omit it.
- PASS. Identity initialisation is identical to design001 — zero-init on `film_net[-1]`.

### Completeness
- PASS. Starting point is `baseline/` — explicit.
- PASS. Files changed: `pose3d_transformer_head.py` and `config.py`. Both fully described.
- PASS. Invariant files explicitly listed as unchanged.
- PASS. Constructor signature is fully specified with backward-compatible defaults.
- PASS. `film_net` architecture for `avg_max`: `Linear(512, 128) → GELU → Linear(128, 512)` — input dim of 512 (`2 * hidden_dim`) is correctly specified and distinguished from `film_hidden_dim`.
- PASS. Forward insertion point identical to design001: after `spatial + pos_enc`, before `queries = ...`.
- PASS. FiLM formula with dual pool: `ctx = cat([mean, max])` → MLP → chunk → `gamma + 1.0` → modulate. Exact code provided.
- PASS. Config additions: `film_pool_type='avg_max'`, `film_hidden_dim=128` as str/int literals — fully specified.
- PASS. Full updated `head` dict shown.
- PASS. Parameter count: 131K parameters for `avg_max` case — consistent with `512×128 + 128×512 = 131K`.
- PASS. Constraint explicitly stated: "MLP input dim for avg_max is `2 * hidden_dim = 512` (not `2 * film_hidden_dim`)."

### Explicitness
- PASS. No ambiguity. Dual-pool logic is given as exact code, not pseudocode.
- PASS. Behaviour at step 0 explicitly verified.
- PASS. `.values` attribute usage for `max(dim=1)` explicitly noted — removes a common implementation pitfall.
- PASS. Gradient flow through `max(dim=1)` (subgradient at argmax) addressed.

### Invariants Not Violated
- PASS. No modification to evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, or `train.py` wrapper.
- PASS. Loss restricted to body joints 0-21 unchanged.
- PASS. `persistent_workers=False`, `resume=True`, `max_keep_ckpts=1`, seed 2026, AMP unchanged.
- PASS. Config uses str/int literals only — no Python imports.

---

## Issues Found

None.

---

## Notes

The distinction between `film_in_dim = 2 * hidden_dim` (MLP input) and `film_hidden_dim = 128` (MLP bottleneck) is made explicit in both the constructor and constraints sections — this prevents a common error of conflating the two. The design is self-consistent and can be implemented directly.

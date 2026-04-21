**Verdict: APPROVED**

**Design:** design002 — Stack 3 decoder layers + intermediate supervision (aux_loss_weight=0.4)
**Idea:** idea001 — Multi-Layer Decoder with Intermediate Supervision
**Reviewer date:** 2026-04-16

---

## Review Summary

The design is complete, unambiguous, and implementation-ready. All required structural, algorithmic, config, and constraint details are explicitly specified. The Builder can implement this without guessing.

---

## Checklist

### Feasibility
- Three `_DecoderLayer` instances with `nn.ModuleList`, collecting intermediates and computing auxiliary losses: standard PyTorch. No new dependencies.
- Memory estimate (~200–250 MB overhead) is plausible for hidden_dim=256, batch_size=4 on 1080 Ti.

### Completeness and Explicitness

| Required field | Present? | Notes |
|---|---|---|
| Design Description | Yes | Clear one-line summary |
| Starting point | Yes | `baseline/` |
| Files to change | Yes | `pose3d_transformer_head.py`, `config.py` only |
| Exact algorithmic changes | Yes | Full constructor signature diff, forward loop with intermediate collection, loss loop verbatim |
| Exact config values | Yes | All head kwargs listed with values including `num_decoder_layers=3`, `aux_loss_weight=0.4` |
| Training/loss/data/inference changes | Yes | Aux losses described fully with table; predict/inference unchanged |
| Constraints and invariants | Yes | 9 constraints listed |
| Expected outputs / edge cases | Yes | Memory, loss table, predict backward-compatibility |

### Correctness Against Baseline

- `self.decoder_layer` → `self.decoder_layers` (ModuleList of 3): correct.
- `forward` replaces the single call with a loop that appends all outputs to `intermediate_outputs`. Return dict adds `'intermediate_joints'` key (list of all intermediate projected outputs). Final output keys `joints`, `pelvis_depth`, `pelvis_uv` unchanged — correct.
- The design specifies that `self.joints_out` is **shared** (called multiple times in `forward`). This is consistent with the baseline code, which defines `self.joints_out = nn.Linear(hidden_dim, 3)` once. No new per-layer heads are created. Explicitly stated in constraint 3.
- `loss()` additions: auxiliary losses keyed `loss/joints_aux{layer_idx}/train` weighted by `self.aux_loss_weight`, body-joints-only masking (`_BODY = list(range(0, 22))`), no pelvis aux losses. All stated explicitly with table.
- `_train_mpjpe` uses `pred['joints']` (final layer) — stated explicitly as unchanged.
- `predict()` unchanged — accesses only `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']`. The `intermediate_joints` key is present in the dict but ignored. Stated as constraint 8.
- Config: `num_decoder_layers=3`, `aux_loss_weight=0.4` added. All other values identical to baseline. Verified against baseline config.
- `persistent_workers=False`, no imports in config.py, absolute imports in head file — all stated.
- No changes to invariant files — stated as constraint 9.

### Invariant Violations
None.

### Ambiguities / Issues

One minor observation (not a rejection issue): The design says `self.joints_out` is "shared" and notes this is "consistent with DETR-style per-layer prediction." In the baseline, `self.joints_out` is already a single `nn.Linear` that will naturally be called multiple times in the new forward loop — so no code change is needed to the projection definition. The design makes this explicit (constraint 3 and the Note in the forward section), so the Builder will not be confused. No ambiguity.

---

## Decision

**APPROVED.** The design is complete, correct, and implementation-ready.

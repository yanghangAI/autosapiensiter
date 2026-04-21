**Verdict: APPROVED**

**Design:** design003 — Stack 4 decoder layers + intermediate supervision + shared output projection
**Idea:** idea001 — Multi-Layer Decoder with Intermediate Supervision
**Reviewer date:** 2026-04-16

---

## Review Summary

The design is complete, unambiguous, and implementation-ready. All required structural, algorithmic, config, and constraint details are explicitly specified. The distinction from Design B (explicit shared-head intent with 4 layers) is clearly articulated. The Builder can implement this without guessing.

---

## Checklist

### Feasibility
- Four `_DecoderLayer` instances with `nn.ModuleList`, single shared `joints_out`, 3 intermediate aux losses: standard PyTorch, no new dependencies.
- Memory estimate (~300–350 MB overhead) is plausible for hidden_dim=256, batch_size=4 on 1080 Ti. OOM mitigation path (hidden_dim=192) is provided and checked for `embed_dim % 4 == 0` validity.

### Completeness and Explicitness

| Required field | Present? | Notes |
|---|---|---|
| Design Description | Yes | Clear one-line summary |
| Starting point | Yes | `baseline/` |
| Files to change | Yes | `pose3d_transformer_head.py`, `config.py` only |
| Exact algorithmic changes | Yes | Full constructor signature, output projection definitions, forward loop, loss loop — all verbatim |
| Exact config values | Yes | All head kwargs listed: `num_decoder_layers=4`, `aux_loss_weight=0.4`, all loss configs |
| Training/loss/data/inference changes | Yes | 3 aux losses described with table, predict unchanged |
| Constraints and invariants | Yes | 10 constraints listed including OOM mitigation |
| Expected outputs / edge cases | Yes | Memory overhead, loss table, shared-head gradient note, OOM path |

### Correctness Against Baseline

- `self.decoder_layer` → `self.decoder_layers` (ModuleList of 4): correct.
- Output projections: the design explicitly states `joints_out`, `depth_out`, `uv_out` are "defined once" — matching the baseline's single-definition pattern. The shared-head design intent is architecturally the same code as Design B; the difference is intentionality and 4 vs. 3 layers. This is clearly explained.
- `forward` collects `layer_outputs` for all 4 layers, returns `intermediate_joints` as list of 3 entries (layers 0, 1, 2), final output from layer 3. Correct for `num_decoder_layers=4`.
- `loss()`: 3 auxiliary loss keys (`loss/joints_aux0/train`, `loss/joints_aux1/train`, `loss/joints_aux2/train`), each weighted 0.4. Body-joints masking applies to all. Pelvis losses on final layer only. Loss table matches the code description.
- `predict()` unchanged — accesses only `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']`. Stated as constraint 8.
- Config: `num_decoder_layers=4`, `aux_loss_weight=0.4`. All other values identical to baseline. Verified.
- `persistent_workers=False`, no imports in config.py, absolute imports in head file, `_DecoderLayer` unchanged — all stated.
- No changes to invariant files — stated as constraint 10.

### Invariant Violations
None.

### Ambiguities / Issues

The design notes that the shared `joints_out` is "identical in code to baseline" and that sharing is "enforced by the forward loop calling `self.joints_out` multiple times." This is accurate and not ambiguous — the Builder will not need to make any structural change to `joints_out` beyond not adding per-layer linear heads. The design makes this explicit. No ambiguity.

The OOM mitigation (constraint 9) specifies reducing `hidden_dim` to 192 in `config.py` head dict only if OOM occurs — this is a conditional fallback, not a default change, and is clearly labeled as such.

---

## Decision

**APPROVED.** The design is complete, correct, and implementation-ready.

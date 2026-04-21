# Design Review — idea017/design003

**Verdict: APPROVED**

---

## Checklist

### Feasibility
- 3-layer decoder over 22 queries: VRAM analysis is credible (3×22 FFN applications = 66 vs. baseline 70; self-attention 3×484=1,452 vs. baseline 4,900). Feasible on 2080 Ti 8 GB.
- The intermediate weight formula `[self.aux_body_loss_weight * (1.0 + 0.5 * k) for k in range(n_inter)]` yields `[0.4, 0.6]` for `n_inter=2` and `[0.4]` for `n_inter=1` — correct and explicitly verified in the design.
- The formula is parameterized and safe for both 2-layer and 3-layer cases (since the Builder implements one unified `loss()` that handles both Designs 002 and 003 via this formula).
- Two loss keys `loss/joints_aux_0/train` (w=0.4) and `loss/joints_aux_1/train` (w=0.6) are correct for 3 decoder layers.
- No new components beyond what Design 001/002 already introduced.

### Completeness
- Starting point: `baseline/` — specified.
- Files to change: `pose3d_transformer_head.py` and `config.py` — both fully specified. `pelvis_utils.py` explicitly unchanged.
- Full constructor signature reproduced with `num_decoder_layers=3` default, `aux_body_loss_weight=0.4`.
- `__init__` body: identical to Designs 001/002 with 3 layers in `nn.ModuleList`.
- `_init_head_weights()`: identical to Designs 001/002.
- `forward()`: identical to Designs 001/002 (full code reproduced). `intermediate_outputs` will have 3 elements: `[layer1_out, layer2_out, layer3_out]`.
- `loss()`: the intermediate block uses the explicit formula, handling `n_inter=2` elements (`intermediate_outputs[:-1]` = `[layer1_out, layer2_out]`) with weights `[0.4, 0.6]`.
- `config.py`: `num_decoder_layers=3`, `aux_body_loss_weight=0.4` as literals.
- Additional constraints 13–17 documented.

### Explicitness
- The critical constraint 13 specifies that `intermediate_weights` must have `n_inter = len(self._intermediate_outputs) - 1` elements (not a hard-coded 2-element list) so the same code works for Designs 002 and 003. This is explicit.
- Constraint 14 documents the weight escalation factor 1.5 and verifies: `k=0 → 0.4*1.0=0.4`, `k=1 → 0.4*1.5=0.6`. Correct.
- The design acknowledges a potential length mismatch if a simpler hard-coded list were used for 2-layer designs, and provides the safe parameterized formula as the required approach. This is exactly what the Builder needs.
- The design includes a derivation section (constraint 15) with two incorrect alternative formulas before arriving at the correct one — this could cause confusion if the Builder reads it carelessly. However, the **final specified formula** is clearly marked and correct:
  ```python
  intermediate_weights = [self.aux_body_loss_weight * (1.0 + 0.5 * k) for k in range(n_inter)]
  ```
  The Builder must use this exact formula. The preceding derivation is commentary, not specification. This is slightly risky but acceptable since the final formula is explicitly labeled.
- Constraint 16 specifies loss key names `loss/joints_aux_0/train` and `loss/joints_aux_1/train` — no collision with existing baseline keys.
- Constraint 17 confirms `joints_out` is shared across all three output levels — intentional, documented.
- OOM fallback instruction is explicit: if OOM occurs, reduce to `num_decoder_layers=2` in config and report to Orchestrator (not silently downgrade).

### Invariant Compliance
- Same as Designs 001/002: no changes to invariant files, `pelvis_utils.py` unchanged, loss restricted to body joints 0-21 for main body loss, `persistent_workers=False` preserved, no Python imports in config.

### Issues / Notes
- The derivation section in constraint 15 includes two incorrect intermediate formulas before the correct one. The Builder should read only the **final specified formula** labeled "Final specified formula" and use that exactly. The review acknowledges this minor presentation risk but the final formula is unambiguous and correctly labeled. This does not constitute a reason to REJECT.
- No other issues. The design is complete and implementable.

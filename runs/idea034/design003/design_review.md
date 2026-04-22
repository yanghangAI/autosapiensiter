# Design Review — idea034 / design003 (Variant C: Metric 3D PE injected into cross-attn keys only)

**Verdict: APPROVED**

## Checks
- **Design Description:** present — PE_3D added to cross-attention *keys* only, values remain pure appearance. Decouples routing (geometry) from aggregation (appearance).
- **Starting point:** `baseline/` — explicit.
- **Files to modify:** only `pose3d_transformer_head.py`, `pelvis_utils.py`, `config.py` — compliant.
- **Algorithmic changes:**
  - `_DecoderLayer.forward` signature extended with `spatial_keys: Tensor | None = None`, defaulting to `spatial_values` (backwards-compatible, baseline call still works). Verified the baseline layer currently passes `spatial_tokens` as both K and V (head file line 118), so the patch point is correct.
  - Head `forward()` code block given verbatim: `spatial_values = spatial + pos_enc`, `spatial_keys = spatial_values + pe3d` only when enabled; this is the load-bearing distinction from Variant A and the design explicitly forbids collapsing to `spatial + pe3d`.
  - `_Metric3DPE` and `unproject_grid_to_metric_3d` reused from design001 spec exactly.
- **Config values:** `use_metric_pe_3d=True`, `metric_pe_variant='keys_only'`, `metric_pe_mlp_hidden=256`, `metric_pe_depth_clamp_min=0.1`, `metric_pe_depth_clamp_max=50.0` — MMEngine-compatible literals.
- **Init / numerical safety:** zero-init of `_Metric3DPE.fc2` → `pe3d=0` → `spatial_keys = spatial_values` → bit-for-bit baseline at step 0.
- **Invariant preservation:** only the head's decoder layer is patched, in a backwards-compatible way; self-attention, FFN, output projections, and loss signatures unchanged. Evaluation/dataset/transforms/backbone/preprocessor/infra/`train.py` untouched. `recover_pelvis_3d` unchanged.
- **Constraints 14–17** correctly call out the Variant-A collapse risk and self-attention invariance.
- **Edge cases:** `metric_xyz=None` and `use_metric_pe_3d=False` paths degrade gracefully to baseline.

Builder has no guesswork; the Variant-A-vs-C distinction is explicitly enumerated.

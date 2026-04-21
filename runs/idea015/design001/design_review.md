**Verdict: APPROVED**

**Design:** idea015/design001 — K=32 super-tokens via single slot-attention layer replacing full 960-token cross-attention.

---

## Review Summary

All review criteria pass. The design is feasible, complete, explicit, and implementation-ready.

### Checklist

**Feasibility:**
- The `nn.MultiheadAttention(256, num_heads=8, batch_first=True)` slot-attention layer is a standard PyTorch module. No custom CUDA or exotic ops required.
- Memory claim is correct: slot attention adds ~10 MB activation; decoder cross-attention saves ~67 MB. Net reduction is strongly negative vs. baseline.
- Backward pass through the 960-key softmax is well-behaved under AMP (already configured).

**Completeness and Explicitness:**
- Starting point: `baseline/` — specified.
- Files to change: `pose3d_transformer_head.py` and `config.py` — specified. `pelvis_utils.py` untouched — confirmed.
- Constructor signature with all four new kwargs (`num_super_tokens`, `slot_pos_init`, `num_decoder_layers`, `aux_loss_weight`) — specified with types and defaults.
- New modules (`slot_queries`, `slot_attn`, `slot_norm`) and their init (`trunc_normal_(std=0.02)`) — specified.
- `self.decoder_layer` (singular) replaced by `self.decoder_layers` (ModuleList with 1 element) — specified.
- `forward()` slot-attention path fully spelled out: expand slot queries to batch, pre-norm, cross-attend over spatial, use super_tokens as `spatial_for_decoder` — specified.
- `loss()` is unchanged (calls `forward()`); `aux_loss_weight=0.0` means no auxiliary branch — specified.
- `predict()` unchanged — specified.
- Config `head=dict(...)` block with literal values `num_super_tokens=32, slot_pos_init=False, num_decoder_layers=1, aux_loss_weight=0.0` — specified.
- Output shapes unchanged: `(B, 70, 3)`, `(B, 1)`, `(B, 2)` — confirmed.
- All 12 constraints listed and consistent with baseline invariants.

**Design Values Verified:**
- `num_super_tokens=32` (int literal) ✓
- `slot_pos_init=False` (bool literal) ✓
- `num_decoder_layers=1` (int literal) ✓
- `aux_loss_weight=0.0` (float literal) ✓
- All other config values (LR, weight decay, warmup, batch, seed, workers) remain at baseline values ✓

**Invariants Preserved:**
- No changes to `pelvis_utils.py`, `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `train.py`, `infra/constants.py`, `infra/metrics_csv_hook.py` ✓
- Joint loss restricted to `_BODY = list(range(0, 22))` ✓
- `persistent_workers=False` preserved ✓
- No Python `import` statements in `config.py` ✓
- Absolute imports in `pose3d_transformer_head.py` preserved ✓
- `_DecoderLayer` class not modified ✓

**Potential Ambiguities — All Resolved:**
- The `intermediate_outputs` list is populated in `forward()` even when `aux_loss_weight=0.0`, but never consumed by `loss()`. This is a no-op accumulation with negligible overhead. The design explicitly covers this ("no changes to the loss body are required when `aux_loss_weight == 0.0`"). No ambiguity.
- Pre-norm on slot queries (via `slot_norm`) before cross-attending over spatial tokens is consistent with `_DecoderLayer`'s pre-norm design. Specified in Constraint 10.

**No issues found.**

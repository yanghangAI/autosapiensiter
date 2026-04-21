**Verdict: APPROVED**

**Design:** idea015/design002 — K=64 super-tokens with positional slot initialization grounding each slot to a spatial block.

---

## Review Summary

All review criteria pass. The design is feasible, complete, explicit, and implementation-ready.

### Checklist

**Feasibility:**
- Same slot-attention mechanism as Design A with K=64 instead of K=32. Standard PyTorch modules only.
- Positional block init logic is a one-time CPU computation at `__init__` time. `_build_2d_sincos_pos_enc(24, 40, 256)` returns a CPU tensor (uses `torch.arange` and arithmetic only — no device argument required). Valid to call inside `_init_head_weights()`.
- Block partition math verified: 8 rows × 8 columns = 64 blocks; each block 3 rows × 5 columns = 15 tokens; 8×3=24 (complete row coverage), 8×5=40 (complete column coverage). No overlap, no gap. Constraint 8 confirms this.

**Completeness and Explicitness:**
- Starting point: `baseline/` — specified.
- Files to change: `pose3d_transformer_head.py` and `config.py` — specified. `pelvis_utils.py` untouched — confirmed.
- Constructor signature identical to Design A — specified.
- New modules (`slot_queries`, `slot_attn`, `slot_norm`) — identical to Design A — specified.
- `_init_head_weights()` changes: `trunc_normal_(std=0.02)` first, then if `slot_pos_init=True`, overwrite with block-averaged positional encodings — exact code given.
- `assert self.num_super_tokens == 64` guard at `__init__` time — specified (Constraint 7).
- Tensor flow in `_init_head_weights`: `pos` computed on CPU, reshaped to `(24, 40, 256)`, blocked and averaged, stacked to `(64, 256)`, assigned via `.data.copy_()`. All explicit.
- `forward()` identical to Design A — specified.
- `loss()` and `predict()` unchanged — specified.
- Config: `num_super_tokens=64, slot_pos_init=True, num_decoder_layers=1, aux_loss_weight=0.0` — all literal values — specified.
- Output shapes unchanged: `(B, 70, 3)`, `(B, 1)`, `(B, 2)` — confirmed.
- All 12 constraints listed.

**Design Values Verified:**
- `num_super_tokens=64` (int literal) ✓
- `slot_pos_init=True` (bool literal) ✓
- `num_decoder_layers=1` (int literal) ✓
- `aux_loss_weight=0.0` (float literal) ✓
- All other config values remain at baseline ✓

**Invariants Preserved:**
- No changes to `pelvis_utils.py`, `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `train.py`, `infra/constants.py`, `infra/metrics_csv_hook.py` ✓
- Joint loss restricted to `_BODY = list(range(0, 22))` ✓
- `persistent_workers=False` preserved ✓
- No Python `import` statements in `config.py` ✓
- `_build_2d_sincos_pos_enc` is already a module-level function — no import needed in `_init_head_weights` ✓
- `_DecoderLayer` class not modified ✓

**Potential Ambiguities — All Resolved:**
- `_build_2d_sincos_pos_enc` is called with hardcoded H'=24, W'=40. Constraint 9 explicitly acknowledges this assumption and notes it matches the 640×384 input at 1/16 stride. Acceptable.
- The assert only fires if `slot_pos_init=True` and `num_super_tokens != 64` — prevents misconfiguration. Correctly placed at `__init__` time.
- The pos_enc tensor created in `_init_head_weights` is a local variable (CPU, not registered as buffer) — discarded after weight assignment. Constraint 6 confirms this explicitly.

**No issues found.**

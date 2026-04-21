# Code Review — idea019/design003

**Verdict: APPROVED**

---

## Review Summary

### 1. review-check-implementation
Passed with exit code 0.

### 2. Files Changed
`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design.
- `code/config.py` — required by design.

Both files permitted. No unexpected files modified. Invariant files unchanged.

### 3. Implementation vs. Design Fidelity

The `pose3d_transformer_head.py` for Design 003 is identical to Design 001/002's file (verified by diff: no differences). All Design 003 behaviour is controlled by config kwargs — the correct implementation strategy.

**Design-003-specific behaviour gated by config:**
- `num_decoder_layers=2`: `decoder_layers = nn.ModuleList` of 2 independent `_DeformableDecoderLayer` instances. Each has its own `ref_points`, `offset_net`, `attn_weight_net`, `value_proj`, `out_proj`, `self_attn`, `ffn`, norms — independent parameters, not shared. Matches design constraint 14.
- `aux_body_loss_weight=0.4`, `num_decoder_layers=2`: `has_intermediate_sup = True`. `intermediate_joints_out = nn.ModuleList([nn.Linear(256, 3)])` — 1 intermediate head (`num_decoder_layers - 1 = 1`). Matches design.
- `num_body_queries=22`, `hand_aux_loss_weight=0.1`: same as Design 002 — `has_hand_proj = True`, hand loss active.

**`forward()` for Design 003 (2-layer path):**
- Iterates `decoder_layers` with index `i`.
- `i < len(decoder_layers) - 1` (i.e., `i < 1`, only `i=0`): appends `queries` (layer 0 output) to `intermediate_decoded`.
- `self._intermediate_decoded = intermediate_decoded` — stores `[(B, 22, 256)]` for `loss()` access.
- `decoded = queries` after loop is final layer (layer 1) output. Correct.

**`loss()` for Design 003:**
- `has_intermediate_sup` guard enters with `_intermediate_decoded` having 1 element (layer 0 output).
- `intermediate_joints_out[0](inter_decoded)`: `(B, 22, 3)` intermediate body predictions.
- `loss/joints_inter0/train = 0.4 * loss_joints_module(inter_body_joints[:, _BODY], gt_joints[:, _BODY])` — weight 0.4, restricted to body indices 0–21. Matches design spec exactly.
- `loss/hand_aux/train = 0.1 * ...` — active, same as Design 002.
- `loss_joints_module` reused for both — no new loss module.

**`_init_head_weights`:**
- Near-zero init applied to each of 2 `_DeformableDecoderLayer` instances in `decoder_layers` — both layers initialised correctly.
- `intermediate_joints_out` init: `trunc_normal_(weight, std=0.02)`, zero bias — present.
- `hand_proj` init — present.

**`config.py`:**
- `num_decoder_layers=2`, `aux_body_loss_weight=0.4`, `num_body_queries=22`, `hand_aux_loss_weight=0.1` — all correct literals.
- All other config values identical to baseline.

### 4. Invariant Files
- `pelvis_utils.py`, `train.py` unchanged. No invariant components touched.

### 5. Test Output
- `slurm_test_55859336.out`: Training ran to completion. "Done training!" reached. No errors.
- Loss log at iter 50/72: `loss/joints/train: 0.231`, `loss/depth/train: 1.950`, `loss/uv/train: 0.151`, `loss/joints_inter0/train: 0.079`, `loss/hand_aux/train: 0.060` — all five expected loss terms present and finite. Intermediate supervision and hand auxiliary losses both firing.
- GPU memory: 8643 MB — slightly above Design 002 (8629 MB) due to second decoder layer. Well within 2080 Ti budget. No OOM.
- AMP active. No dtype errors. No NaN/Inf.
- Model loaded 293/293 backbone tensors. Head randomly initialised.

### 6. Notable Observations
- All 5 expected loss terms (`loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/joints_inter0/train`, `loss/hand_aux/train`) appear in the training log, confirming the intermediate supervision and auxiliary hand loss are both correctly active.
- Memory footprint difference between Design 002 and Design 003 is only 14 MB (8629 → 8643), consistent with the design's memory estimate of < 1 MB additional VRAM for the second layer.
- The implementation strategy of a single shared `pose3d_transformer_head.py` for all three designs is clean and correct.

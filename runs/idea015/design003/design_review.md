**Verdict: APPROVED**

**Design:** idea015/design003 — K=32 super-tokens + 2 decoder layers with auxiliary loss weight 0.4 on intermediate layer output.

---

## Review Summary

All review criteria pass. The design is feasible, complete, explicit, and implementation-ready.

### Checklist

**Feasibility:**
- Same slot-attention mechanism as Design A (K=32). Two `_DecoderLayer` instances in a `nn.ModuleList`. Standard PyTorch modules only.
- `_forward_with_intermediates()` refactor is clean: moves the computation body out of `forward()` into a shared helper. `forward()` and `predict()` call it and discard the intermediate list. `loss()` calls it and uses the intermediates. No duplication of computation.
- Memory: 2 decoder layers × 32 K/V tokens is still cheaper than baseline 1 layer × 960 tokens. Net memory reduction confirmed by the design's analysis (proportional to 64 vs. 960).

**Completeness and Explicitness:**
- Starting point: `baseline/` — specified.
- Files to change: `pose3d_transformer_head.py` and `config.py` — specified. `pelvis_utils.py` untouched — confirmed.
- Constructor signature with all four new kwargs — specified (identical to Designs A and B).
- New modules: `slot_queries`, `slot_attn`, `slot_norm` (conditional on `num_super_tokens > 0`); `decoder_layers` ModuleList with 2 elements — specified.
- `self.decoder_layer` (singular) removed — specified.
- `_init_head_weights()`: `trunc_normal_(std=0.02)` for slot queries; no spatial block init (`slot_pos_init=False`) — specified.
- `_forward_with_intermediates()` full implementation given: returns `(pred_dict, intermediate_outputs)` where `pred_dict` matches the baseline return dict and `intermediate_outputs` is a list of raw `decoded` tensors (one per decoder layer) — specified.
- `forward()` updated to call `pred, _ = self._forward_with_intermediates(feats); return pred` — specified.
- `loss()` updated to call `pred, intermediate_outputs = self._forward_with_intermediates(feats)` and compute auxiliary loss for `intermediate_outputs[:-1]` — full code given.
- Auxiliary loss: applies `self.joints_out(inter_decoded)` on layer-1 raw decoded output, then `aux_loss_weight * loss_joints_module(inter_joints[:, _BODY], gt_joints[:, _BODY])` — specified.
- Auxiliary loss key: `f'loss/joints_aux_{i}/train'` where `i=0` for `num_decoder_layers=2` — specified.
- `predict()` unchanged — specified.
- Config: `num_super_tokens=32, slot_pos_init=False, num_decoder_layers=2, aux_loss_weight=0.4` — all literal values — specified.
- Output shapes unchanged: `(B, 70, 3)`, `(B, 1)`, `(B, 2)` — confirmed.
- All 14 constraints listed.

**Design Values Verified:**
- `num_super_tokens=32` (int literal) ✓
- `slot_pos_init=False` (bool literal) ✓
- `num_decoder_layers=2` (int literal) ✓
- `aux_loss_weight=0.4` (float literal) ✓
- All other config values remain at baseline ✓

**Invariants Preserved:**
- No changes to `pelvis_utils.py`, `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `train.py`, `infra/constants.py`, `infra/metrics_csv_hook.py` ✓
- Joint loss restricted to `_BODY = list(range(0, 22))` applied to BOTH primary and auxiliary joint losses ✓
- Pelvis depth and UV computed from final layer only (`decoded[:, 0, :]` from `decoded_2`) ✓
- `persistent_workers=False` preserved ✓
- No Python `import` statements in `config.py` ✓
- `_DecoderLayer` class not modified ✓
- Super-tokens computed once and reused by both decoder layers — Constraint 6 explicitly confirmed ✓
- `self.joints_out` shared projection head used for both primary and auxiliary losses — Constraint 7 confirmed ✓

**Potential Ambiguities — All Resolved:**
- `intermediate_outputs` contains raw `decoded` tensors (shape `(B, num_joints, hidden_dim)`), not projected joints. The aux loss applies `self.joints_out(inter_decoded)` inside `loss()`. This is explicit in the design and consistent with the full `loss()` code given.
- `intermediate_outputs[:-1]` for `num_decoder_layers=2` yields exactly one element (`intermediate_outputs[0]` = layer-1 output). The design states this explicitly in Constraint 14.
- `_forward_with_intermediates` is an internal helper not registered with MMEngine and not callable by any hook. `forward()` and `predict()` maintain their public API unchanged. Constraint 9 confirms.
- The `_train_mpjpe` and `_train_mpjpe_abs` attributes (for `TrainMPJPEAveragingHook`) are still computed in `loss()` from `pred` (final layer output) — unchanged from baseline. The full `loss()` code given confirms this.

**No issues found.**

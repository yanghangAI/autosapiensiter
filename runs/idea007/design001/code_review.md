# Code Review — idea007/design001

**Verdict: APPROVED**

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea007/design001` — PASSED.

---

## Files Changed vs. Design Requirements

Design specifies changes to `pose3d_transformer_head.py` and `config.py` only. Implementation summary lists exactly these two files. `pelvis_utils.py` and `train.py` are present in the code directory but unchanged relative to baseline (expected, as they are carried along). No invariant files were modified.

---

## Fidelity Check: `pose3d_transformer_head.py`

### `_DecoderLayer.__init__`
- New signature accepts `num_joints: int = 70` and `num_spatial: int = 960` — exactly as specified.
- `self.cross_attn_bias = nn.Parameter(torch.zeros(num_joints, num_spatial))` — correct shape `(70, 960)`, zero init, registered as `nn.Parameter`. Design constraint 1 satisfied.
- `cross_attn_bias` is NOT touched in `_init_head_weights` — constraint 8 satisfied.

### `_DecoderLayer.forward`
- Shape assertion present: `assert spatial_tokens.shape[1] == self.cross_attn_bias.shape[-1]` with the exact error message specified — constraint 5 satisfied.
- `attn_mask=self.cross_attn_bias` passed to `self.cross_attn(q, spatial_tokens, spatial_tokens, ...)` — correct.
- `key_padding_mask` and `is_causal` not set — constraint 2 satisfied.
- `batch_first=True` confirmed on `self.cross_attn` — constraint 9 satisfied; `(num_joints, num_spatial)` shape is correct.

### `Pose3dTransformerHead.__init__`
- `num_spatial: int = 960` added as constructor kwarg — present.
- `self.num_spatial = num_spatial` stored — present.
- `_DecoderLayer` constructed with `num_joints=num_joints, num_spatial=num_spatial` — correct.
- `cross_routing_type` is NOT present in this design (design001 is the simpler variant that hardwires zero init without the routing-type dispatch). The design does not require `cross_routing_type` — it was only introduced in design002. This is consistent with the design.

### `Pose3dTransformerHead.forward`
- `decoded = self.decoder_layer(queries, spatial)` — unchanged; design001 does not pass `B` (per-head routing not applicable). Correct.

---

## Fidelity Check: `config.py`

- `num_spatial=960` added as integer literal to head kwargs — present.
- No Python import statements added — compliant.
- All other config values (optimizer, LR schedule, batch size, hooks, seed, data pipeline) are unchanged from baseline.

---

## Invariant Check

No modifications to `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `infra/constants.py`, `infra/metrics_csv_hook.py`, or `train.py` wrapper. Confirmed.

---

## Test Output

- Training ran for 1 epoch with no errors or exceptions.
- Backbone loaded correctly: 293/293 tensors, 4-channel patch_embed, pos_embed interpolated.
- Head tensors randomly initialised (correct — no checkpoint for head).
- Training completed, checkpoint saved, validation ran, metric CSV written.
- Epoch 1 metrics: `composite_val=491.04`, `mpjpe_body_val=443.23`, `mpjpe_pelvis_val=588.12`. These are epoch-1 cold-start values (not yet converged); plausible and consistent with baseline-level initialisation.
- `metrics.csv` contains the required CSV columns: `epoch, composite_val, mpjpe_body_val, mpjpe_pelvis_val, mpjpe_rel_val, mpjpe_hand_val, mpjpe_abs_val`. No runtime issues.

---

## Summary

All required design details are present and correctly implemented. No missing constraints, no wrong-file changes, no invariant violations. Test run completed cleanly.

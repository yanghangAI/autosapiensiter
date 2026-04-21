# Code Review — idea007/design002

**Verdict: APPROVED**

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea007/design002` — PASSED.

---

## Files Changed vs. Design Requirements

Design specifies changes to `pose3d_transformer_head.py` and `config.py` only. Implementation summary lists exactly these two files. No invariant files were modified.

---

## Fidelity Check: `pose3d_transformer_head.py`

### `_DecoderLayer.__init__` — new signature
New signature: `(embed_dim, num_heads=8, dropout=0.1, num_joints=70, num_spatial=960, cross_attn_bias_init='zero')` — matches design exactly.

### Band prior computation
- `LOWER_BODY_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]` — matches design.
- `UPPER_BODY_JOINTS = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]` — matches design.
- `_H_prime = 40`, `_W_prime = 24` — hardcoded as specified.
- `row_idx = torch.arange(num_spatial, dtype=torch.float32).div(_W_prime, rounding_mode='floor')` — uses float-safe integer division as required by design constraint 2.
- `sigma = 5.0`, Gaussian centres at `30.0` (lower) and `10.0` (upper) — correct.
- `bias_lower = g_lower - 0.5`, `bias_upper = g_upper - 0.5` — range `[-0.5, +0.5]`, constraint 1 (no hard binary masks) satisfied.
- Loop assigns `bias_lower` to all lower-body joint rows and `bias_upper` to all upper-body rows; hand joints (22–69) left at zero — correct.
- `self.cross_attn_bias = nn.Parameter(init_bias)` — warm-start values passed directly into `nn.Parameter`, copied on construction; constraint 4 satisfied.
- `_init_head_weights` does NOT touch `cross_attn_bias` — constraint 5 satisfied.

### `_DecoderLayer.forward`
- Shape assertion present — constraint 6 satisfied.
- `attn_mask=self.cross_attn_bias` passed — correct additive semantics (constraint 7).
- No `key_padding_mask` or `is_causal` — correct.

### `Pose3dTransformerHead.__init__`
- `num_spatial: int = 960` and `cross_routing_type: str = 'none'` added.
- `_bias_init_map = {'none': 'zero', 'zero_init': 'zero', 'band_prior': 'band_prior'}` — correct mapping; `'none'` recovers zero init (design constraint 11 / backward compatibility).
- `_DecoderLayer` constructed with `cross_attn_bias_init=_bias_init` — correct.
- `self.num_spatial = num_spatial` stored — present.

### `Pose3dTransformerHead.forward`
- `decoded = self.decoder_layer(queries, spatial)` — design002 is not per-head; `B` not required. Correct.

---

## Fidelity Check: `config.py`

- `num_spatial=960` and `cross_routing_type='band_prior'` added as plain literals to head kwargs — both present, exactly as specified.
- No Python import statements added — compliant.
- All other config values unchanged from baseline.

---

## Invariant Check

No modifications to evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, or train.py wrapper. Confirmed.

---

## Test Output

- Training ran for 1 epoch without errors.
- Backbone loaded correctly: 293/293 tensors.
- Epoch 1 metrics: `composite_val=484.83`, `mpjpe_body_val=441.18`, `mpjpe_pelvis_val=573.46`.
- Notably, design002's epoch-1 composite (484.83) is already slightly lower than design001's (491.04) and design003's (491.04), consistent with the warm-start prior providing a better starting point from the first evaluation pass.
- `metrics.csv` has correct columns and no missing rows. Training completed cleanly with `[test] Finished`.

---

## Summary

All required design details — band prior init values, Gaussian centres/sigma, joint group lists, float-safe row_idx division, `cross_routing_type` mapping, config literals — are present and exactly correct. No missing constraints, no wrong-file changes, no invariant violations. Test run completed cleanly.

# Code Review — idea007/design003

**Verdict: APPROVED**

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea007/design003` — PASSED.

---

## Files Changed vs. Design Requirements

Design specifies changes to `pose3d_transformer_head.py` and `config.py` only. Implementation summary lists exactly these two files. No invariant files were modified.

---

## Fidelity Check: `pose3d_transformer_head.py`

### `_DecoderLayer.__init__` — new signature
New signature: `(embed_dim, num_heads=8, dropout=0.1, num_joints=70, num_spatial=960, cross_attn_bias_init='zero', per_head_routing=False)` — matches design exactly.

### Per-head routing branch (`per_head_routing=True`)
- `self._per_head = True` stored — constraint 5 satisfied.
- `self._num_heads = num_heads` stored — constraint 5 satisfied.
- `self.cross_attn_bias = nn.Parameter(torch.zeros(num_heads, num_joints, num_spatial))` — shape `(8, 70, 960)`, zero init — constraint 1 satisfied. Exact parameter count 8×70×960 = 537,600 as specified.

### Non-per-head branch (`per_head_routing=False`)
- Falls through to the shared-bias path (including optional `band_prior` init), backward-compatible with designs 001/002 — constraint 7 satisfied.
- `self._per_head = False` stored; `self._num_heads = num_heads` also stored (correct, needed if switching).

### `_DecoderLayer.forward`
- Signature: `(queries, spatial_tokens, B: int = 1)` — `B=1` default present as required by constraint 12.
- Shape assertion on `spatial_tokens.shape[1] == self.cross_attn_bias.shape[-1]` — checks last dim of the per-head bias tensor `(H, J, S)`, so `.shape[-1]` = S = 960 — constraint 6 satisfied.
- Per-head expansion:
  ```python
  bias_expanded = (
      self.cross_attn_bias       # (num_heads, J, S)
      .unsqueeze(0)              # (1, num_heads, J, S)
      .expand(B, -1, -1, -1)    # (B, num_heads, J, S)
      .reshape(B * self._num_heads,
               self.cross_attn_bias.shape[1],
               self.cross_attn_bias.shape[2])  # (B*H, J, S)
  )
  ```
  Uses `.expand` not `.repeat` — constraint 3 satisfied. Shape `(B*8, 70, 960)` — constraint 4 satisfied.
- Comment present noting `B must be passed explicitly by Pose3dTransformerHead.forward` — constraint 12 addressed.
- `attn_mask=bias_expanded` passed to cross-attention in per-head branch; shared `self.cross_attn_bias` passed in non-per-head branch — correct branching.

### `Pose3dTransformerHead.__init__`
- `num_spatial: int = 960` and `cross_routing_type: str = 'none'` added.
- `_per_head = (cross_routing_type == 'per_head')` — correct mapping.
- `_bias_init_map` retained for `band_prior` and `zero_init` types — backward compatible.
- `_DecoderLayer` constructed with `per_head_routing=_per_head` and `cross_attn_bias_init=_bias_init` — correct.
- `_init_head_weights` does NOT touch `cross_attn_bias` — constraint 8 satisfied.

### `Pose3dTransformerHead.forward`
- `B, C, H, W = feat.shape` present (existing line).
- `decoded = self.decoder_layer(queries, spatial, B=B)` — `B` extracted from features and passed explicitly, not hardcoded — constraint 2 satisfied.

---

## Fidelity Check: `config.py`

- `num_spatial=960` and `cross_routing_type='per_head'` added as plain literals to head kwargs — both present.
- No Python import statements added — compliant.
- All other config values unchanged from baseline.

---

## Invariant Check

No modifications to evaluation metric, dataset, transforms, backbone, data preprocessor, infra files, or train.py wrapper. Confirmed.

---

## Test Output

- Training ran for 1 epoch without errors.
- Backbone loaded correctly: 293/293 tensors. Memory usage 10619 MB (vs 10612 for design001/002), consistent with the larger per-head bias (~2 MB vs ~262 KB) — plausible.
- Epoch 1 metrics: `composite_val=491.04`, `mpjpe_body_val=443.20`, `mpjpe_pelvis_val=588.15`. These are essentially identical to design001's epoch-1 values, which is expected: both are zero-initialised and behave identically at epoch 1 before the per-head biases have had time to diverge.
- `metrics.csv` contains correct columns. Training completed cleanly with `[test] Finished`.

---

## Summary

All required design details — per-head bias shape, zero init, `_per_head`/`_num_heads` attributes, `.expand` not `.repeat`, `(B*H, J, S)` expansion, `B` passed from feature shape, `B=1` default, backward-compatible non-per-head path, config literals — are present and correctly implemented. No missing constraints, no wrong-file changes, no invariant violations. Test run completed cleanly.

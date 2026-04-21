**Files changed:**
- `pose3d_transformer_head.py`
- `config.py`

**Changes:**

`pose3d_transformer_head.py`: Extended `_DecoderLayer.__init__` with `num_joints`, `num_spatial`, `cross_attn_bias_init`, and `per_head_routing` arguments. When `per_head_routing=True`, registers `self.cross_attn_bias` as shape `(num_heads, num_joints, num_spatial)` zero-initialised; stores `self._per_head = True` and `self._num_heads = num_heads` for use in forward. Updated `_DecoderLayer.forward` to accept `B: int = 1`; when `_per_head=True`, expands the bias via `.unsqueeze(0).expand(B, ...).reshape(B * num_heads, J, S)` and passes it as `attn_mask` to cross-attention; when `_per_head=False`, passes the shared `(J, S)` bias directly (backward-compatible with designs 001/002). A shape assertion is present in both branches. In `Pose3dTransformerHead.__init__`, added `num_spatial: int = 960` and `cross_routing_type: str = 'none'` arguments; maps `'per_head'` to `per_head_routing=True` when constructing `_DecoderLayer`. In `Pose3dTransformerHead.forward`, changed the decoder call to `self.decoder_layer(queries, spatial, B=B)` so the batch size is always passed explicitly.

`config.py`: Added `num_spatial=960` and `cross_routing_type='per_head'` as plain literals to the head kwargs dict, enabling the per-head routing variant.

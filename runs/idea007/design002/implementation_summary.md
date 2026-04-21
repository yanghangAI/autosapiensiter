**Files changed:**
- `pose3d_transformer_head.py`
- `config.py`

**Changes:**

`pose3d_transformer_head.py`: Extended `_DecoderLayer.__init__` to accept `num_joints`, `num_spatial`, and `cross_attn_bias_init` arguments. When `cross_attn_bias_init='band_prior'`, the bias is initialised with a Gaussian vertical-band prior: body-lower joints (hardcoded indices `[1,2,4,5,7,8,10,11]`) are biased toward lower spatial rows (Gaussian centre at row 30, σ=5, scaled to ±0.5), body-upper joints (indices `[0,3,6,9,12,13,14,15,16,17,18,19,20,21]`) toward upper rows (centre row 10), and hand joints (22–69) remain zero. Row indices are computed with float-safe `.div(..., rounding_mode='floor')`. The forward pass asserts spatial shape and passes `attn_mask=self.cross_attn_bias` to cross-attention. In `Pose3dTransformerHead.__init__`, added `num_spatial: int = 960` and `cross_routing_type: str = 'none'` arguments; `cross_routing_type` is mapped to the `cross_attn_bias_init` string before constructing `_DecoderLayer`.

`config.py`: Added `num_spatial=960` and `cross_routing_type='band_prior'` as plain literals to the head kwargs dict, activating the Gaussian warm-start prior.

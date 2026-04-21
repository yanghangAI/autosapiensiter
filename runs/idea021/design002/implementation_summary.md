**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Same changes as design001 — added optional `cross_attn_bias` argument to `_DecoderLayer.forward()`, new kwargs and instance attributes to `Pose3dTransformerHead.__init__()`, parameter allocation block creating `cross_attn_bias_row (70, 40)` and `cross_attn_bias_col (70, 24)` when `cross_attn_bias_type='factored'`, warm-start logic in `_init_head_weights()` (inactive for design002), and bias computation in `forward()` via outer sum broadcast.
- `code/config.py`: Added `use_cross_attn_bias=True`, `cross_attn_bias_type='factored'`, `feat_h=40`, `feat_w=24` to the head dict to enable the factored row+column cross-attention bias with 15× fewer parameters than the full bias matrix in design001.

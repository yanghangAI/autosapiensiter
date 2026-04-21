**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added optional `cross_attn_bias` argument to `_DecoderLayer.forward()` which, when provided, is passed as `attn_mask` (cast to query dtype for AMP compatibility) to `nn.MultiheadAttention`; added new kwargs `use_cross_attn_bias`, `cross_attn_bias_type`, `feat_h`, `feat_w`, `joint_row_prior` to `Pose3dTransformerHead.__init__()` with backward-compatible defaults; allocated `self.cross_attn_bias` parameter `(70, 960)` when `use_cross_attn_bias=True` and `cross_attn_bias_type='full'`; updated `forward()` to compute and pass the bias to `decoder_layer`; added warm-start Gaussian logic in `_init_head_weights()` (inactive for design001 since `cross_attn_bias_type='full'`).
- `code/config.py`: Added `use_cross_attn_bias=True`, `cross_attn_bias_type='full'`, `feat_h=40`, `feat_w=24` to the head dict so the full `(70, 960)` zero-initialized cross-attention bias is enabled for this design.

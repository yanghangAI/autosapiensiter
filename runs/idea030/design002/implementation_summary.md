**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `_EncoderLayer` class (single transformer encoder layer with self-attention over spatial tokens + FFN, zero-initialized output projections) before `_DecoderLayer`; extended `Pose3dTransformerHead.__init__` with `use_spatial_encoder`, `num_encoder_layers`, `encoder_num_heads`, `encoder_dropout`, `encoder_zero_init` kwargs and conditional `nn.ModuleList` registration; inserted encoder loop in `forward()` after positional encoding and before decoder cross-attention. Identical to design001's head changes.
- `code/config.py`: Added `use_spatial_encoder=True`, `num_encoder_layers=1`, `encoder_num_heads=4`, `encoder_dropout=0.1`, `encoder_zero_init=True` to the `head` dict; uses 4 heads instead of design001's 8 for memory efficiency (halves encoder attention from ~58 MB to ~29 MB in float16).

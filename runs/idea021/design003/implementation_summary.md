**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Same structural changes as design002 — added `cross_attn_bias` argument to `_DecoderLayer.forward()`, new kwargs and instance attributes to `Pose3dTransformerHead.__init__()`, factored parameter allocation (`cross_attn_bias_row (70, 40)` and `cross_attn_bias_col (70, 24)`), and bias computation in `forward()` via outer sum; additionally the warm-start logic in `_init_head_weights()` is now active — for the 22 body joint row biases, a Gaussian with σ=4.0 and α=1.0 centered at each joint's expected row position is written to `.data`, while hand joints remain zero-initialized.
- `code/config.py`: Added `use_cross_attn_bias=True`, `cross_attn_bias_type='factored_warmstart'`, `feat_h=40`, `feat_w=24`, and `joint_row_prior=[...]` (22 float literals for body joint row positions in the 40-row feature grid) to the head dict so the Gaussian warm-start activates at model construction.

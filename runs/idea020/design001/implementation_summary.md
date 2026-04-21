**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added `_temp_scaled_attn` module-level helper function (after `_build_2d_sincos_pos_enc`) that performs manual MHA forward with per-query temperature division of dot-product logits before softmax, with AMP-safe dtype cast (`to(attn.dtype)`) and min-clamp of 0.1. Modified `_DecoderLayer.__init__` to remove temperature storage (temperatures are not stored in `_DecoderLayer` to avoid duplicate `nn.Parameter` registration between head and decoder layer); modified `_DecoderLayer.forward` to accept optional `cross_temp` and `self_temp` arguments and route through `_temp_scaled_attn` when non-None. Modified `Pose3dTransformerHead.__init__` to accept four new kwargs (`use_cross_temp`, `use_self_temp`, `temp_log_space`, `temp_reg_weight`) with backward-compatible defaults; creates `self.cross_temp = nn.Parameter(torch.ones(num_joints))` when `use_cross_temp=True` and `temp_log_space=False`; asserts `cross_attn._qkv_same_embed_dim=True`. Modified `Pose3dTransformerHead.forward` to compute the effective temperature tensors and pass them to the decoder layer each call. Added optional L2 regularisation loss in `loss()` for designs with `temp_reg_weight > 0`.

`code/config.py`: Added four new head kwargs as bool/float literals: `use_cross_temp=True`, `use_self_temp=False`, `temp_log_space=False`, `temp_reg_weight=0.0`. No Python import statements introduced.

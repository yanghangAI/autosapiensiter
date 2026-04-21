**Files changed:**
- `pose3d_transformer_head.py`
- `config.py`

**Changes:**

`pose3d_transformer_head.py`: Added `spatial_drop_prob: float = 0.0` parameter to `_DecoderLayer.forward`; during training when `spatial_drop_prob > 0`, a fresh boolean `key_padding_mask` of shape `(B, N_spatial)` is generated via `torch.rand(...) < spatial_drop_prob` and passed to `self.cross_attn` to mask 15% of spatial tokens from cross-attention. Added `spatial_drop_prob: float = 0.0` parameter to `Pose3dTransformerHead.__init__` (stored as `self.spatial_drop_prob`) and updated the `decoder_layer` call in `forward` to pass `spatial_drop_prob=self.spatial_drop_prob`.

`config.py`: Added `spatial_drop_prob=0.15` to the head kwargs dict, enabling 15% uniform spatial token dropout during training cross-attention.

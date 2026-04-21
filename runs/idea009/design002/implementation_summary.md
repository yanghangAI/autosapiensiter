**Files changed:**
- `pose3d_transformer_head.py`
- `config.py`

**Changes:**

`pose3d_transformer_head.py`: Identical changes to design001 — added `spatial_drop_prob: float = 0.0` parameter to `_DecoderLayer.forward` with fresh `key_padding_mask` generation during training; added `spatial_drop_prob: float = 0.0` parameter to `Pose3dTransformerHead.__init__` (stored as `self.spatial_drop_prob`) and updated the `decoder_layer` call in `forward` to pass `spatial_drop_prob=self.spatial_drop_prob`.

`config.py`: Added `spatial_drop_prob=0.30` to the head kwargs dict, enabling 30% uniform spatial token dropout during training cross-attention (more aggressive regularisation than design001's 15%).

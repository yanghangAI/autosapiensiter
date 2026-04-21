**Files changed:**
- `pose3d_transformer_head.py`
- `config.py`

**Changes:**

`pose3d_transformer_head.py`: Extended `_DecoderLayer.__init__` to accept `num_joints: int = 70` and `num_spatial: int = 960` arguments and register `self.cross_attn_bias = nn.Parameter(torch.zeros(num_joints, num_spatial))` — a zero-initialised learnable additive bias for cross-attention logits. In `_DecoderLayer.forward`, added a shape assertion and passed `attn_mask=self.cross_attn_bias` to the cross-attention call. In `Pose3dTransformerHead.__init__`, added `num_spatial: int = 960` as a constructor argument (stored as `self.num_spatial`) and forwarded `num_joints` and `num_spatial` to `_DecoderLayer`.

`config.py`: Added `num_spatial=960` as an integer literal to the head kwargs dict, exposing the spatial grid size to the constructor without using any Python imports.

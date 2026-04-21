**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added `decouple_pelvis: bool = False` and `pelvis_decoder_type: str = 'shared'` constructor parameters. When `decouple_pelvis=True` and `pelvis_decoder_type='depth_fused'`, creates `self.pelvis_query` (nn.Embedding(1, hidden_dim)), `self.pelvis_decoder` (independent _DecoderLayer), and `self.depth_proj` (Linear(hidden_dim, hidden_dim)). In `forward()`, a global depth-context token is produced by mean-pooling `spatial` over the spatial dimension and projecting through `depth_proj`; this token is prepended to `spatial` to form `spatial_with_depth` which is used only in the pelvis cross-attention call. The pelvis query runs through `pelvis_decoder` cross-attention (self-attention skipped), giving it access to a global scale anchor alongside local spatial features. Joint queries use the original `spatial` unchanged. Updated module docstring to reflect the new pathway.

`code/config.py`: Added `decouple_pelvis=True` and `pelvis_decoder_type='depth_fused'` to the `head` dict to activate the depth-fused pelvis decoder pathway. All other config values are identical to baseline.

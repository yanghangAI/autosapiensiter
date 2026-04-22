**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added `_SpatialContextNet` module class (before `_DecoderLayer`) implementing a two-layer stacked depthwise-separable 2D convolution with GroupNorm(32) normalization, GELU activation, and zero-initialized final pointwise conv only (`zero_init_last=True`) for baseline-equivalent start; added `use_spatial_ctx`, `spatial_ctx_kernel`, `spatial_ctx_layers`, `spatial_ctx_norm`, `spatial_ctx_groups`, `spatial_ctx_act` kwargs to `Pose3dTransformerHead.__init__` and conditionally instantiate `self.spatial_ctx_net`; added `if self.use_spatial_ctx: spatial = self.spatial_ctx_net(spatial, H, W)` in `forward()` after positional encoding and before the decoder.

`code/config.py`: Added `use_spatial_ctx=True`, `spatial_ctx_kernel=3`, `spatial_ctx_layers=2`, `spatial_ctx_norm='groupnorm'`, `spatial_ctx_groups=32`, `spatial_ctx_act='gelu'` to the head dict, enabling two-layer stacked GroupNorm-normalized spatial context enrichment (5x5 effective receptive field) for design003 (Design C).

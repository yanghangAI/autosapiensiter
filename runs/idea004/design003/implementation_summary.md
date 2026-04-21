**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added `numpy` and `torch.nn.functional` imports. Added `depth_pos_enc_type` parameter to `__init__` and a 2-layer `pos_mlp = nn.Sequential(Linear(3,64), GELU, Linear(64,hidden_dim))` with trunc_normal weight and zero bias initialization. Added `_extract_depth_map()` helper (identical to Design A/B). Added `_build_3d_pos_grid()` helper that constructs an `(B, H*W, 3)` tensor of normalised (x, y, depth) coordinates (x/y in [-1,1], depth clamped [0,10]/10 in [0,1]; fallback depth=0.5 when depth_map is None). Updated `forward()` to accept `depth_map`, build the 3D position grid via `_build_3d_pos_grid`, run it through `pos_mlp`, and add the result to spatial tokens — entirely replacing the fixed 2D sinusoidal positional encoding (`_get_pos_enc` is not called in forward). Updated `loss()` and `predict()` to extract depth and pass to `forward()`.

`code/config.py`: Added `depth_pos_enc_type='mlp'` to the `head` dict. All other config values are unchanged from baseline.

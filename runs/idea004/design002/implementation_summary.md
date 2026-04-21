**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added `numpy` and `torch.nn.functional` imports. Added module-level helper `_build_1d_sincos_enc()` that builds a 1D sinusoidal encoding for scalar depth values (analogous to the 2D DETR-style encoding). Added `depth_pos_enc_type` parameter to `__init__` and a `depth_pos_proj = nn.Linear(hidden_dim + hidden_dim//2, hidden_dim)` layer (trunc_normal weight, zero bias) that projects the concatenated [2D sincos || depth sinusoidal] positional signal to `hidden_dim`. Added `_extract_depth_map()` helper method identical to Design A. Updated `forward()` to accept an optional `depth_map` kwarg; when provided, normalises depth to [0,1], builds a depth sinusoidal encoding via `_build_1d_sincos_enc`, concatenates with the 2D positional encoding, and projects through `depth_pos_proj`; the fallback path pads depth with zeros so `depth_pos_proj` always participates in the graph. Updated `loss()` and `predict()` to extract depth and pass to `forward()`.

`code/config.py`: Added `depth_pos_enc_type='sinusoidal'` to the `head` dict. All other config values are unchanged from baseline.

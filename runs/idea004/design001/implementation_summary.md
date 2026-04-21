**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added `numpy` and `torch.nn.functional` imports. Added `depth_pos_enc_type` parameter to `__init__` and a `depth_proj = nn.Linear(1, hidden_dim)` layer with zero-init weights and bias so the depth signal starts at zero (baseline-equivalent at epoch 0). Added `_extract_depth_map()` helper method that loads depth NPZ from `batch_data_samples` metainfo, crops to `img_shape`, and bilinearly resizes to feature map resolution. Updated `forward()` to accept an optional `depth_map` kwarg and, when provided, normalises depth to [0,1] (clamp [0,10]/10), projects with `depth_proj`, and adds the result to spatial tokens after the 2D sinusoidal positional encoding. Updated `loss()` and `predict()` to call `_extract_depth_map()` and pass the result to `forward()`.

`code/config.py`: Added `depth_pos_enc_type='linear'` to the `head` dict so the head is constructed with Design A's linear depth projection enabled. All other config values are unchanged from baseline.

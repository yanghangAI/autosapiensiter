**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `_PelvisCrossAttnDecoder` class at module level (after `_DecoderLayer`). Added four new kwargs (`use_decoupled_pelvis`, `pelvis_hidden_dim`, `pelvis_num_heads`, `num_body_queries`) to `__init__`. Changed `joint_queries` embedding from `nn.Embedding(num_joints, hidden_dim)` to `nn.Embedding(num_body_queries, hidden_dim)` so the joint decoder operates on only 22 body queries (no hand contamination in self-attention). Updated `forward()` to: (1) run the decoder on 22 queries, (2) zero-pad hand joints (indices 22–69) to produce full `(B, 70, 3)` joints tensor, and (3) use the dedicated pelvis decoder path for `depth_out`/`uv_out` when `use_decoupled_pelvis=True`. Updated `_init_head_weights()` to initialize `pelvis_coord_queries` when enabled.
- `code/config.py`: Added `use_decoupled_pelvis=True`, `pelvis_hidden_dim=256`, `pelvis_num_heads=8`, `num_body_queries=22` to the head config dict, activating both the 22-query body-only joint decoder and the decoupled pelvis coordinate decoder simultaneously.

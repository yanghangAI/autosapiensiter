**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Added `_PelvisCrossAttnDecoder` class at module level (after `_DecoderLayer`) — a lightweight cross-attention-only module (no self-attn, no FFN) for 2 pelvis coordinate queries. Added four new kwargs (`use_decoupled_pelvis`, `pelvis_hidden_dim`, `pelvis_num_heads`, `num_body_queries`) to `__init__`, conditionally instantiates `pelvis_coord_queries` (Embedding of size 2) and `pelvis_decoder`. Updated `_init_head_weights()` to initialize `pelvis_coord_queries` when enabled, and updated `forward()` to use the dedicated pelvis decoder path when `use_decoupled_pelvis=True` so `depth_out` and `uv_out` read from dedicated pelvis query outputs rather than joint token 0.
- `code/config.py`: Added `use_decoupled_pelvis=True`, `pelvis_hidden_dim=256`, `pelvis_num_heads=8`, `num_body_queries=70` to the head config dict to activate the decoupled pelvis decoder with 8 attention heads.

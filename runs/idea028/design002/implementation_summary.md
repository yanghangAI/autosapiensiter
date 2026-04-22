**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Identical changes to design001 — added `_PelvisCrossAttnDecoder` class at module level, four new kwargs (`use_decoupled_pelvis`, `pelvis_hidden_dim`, `pelvis_num_heads`, `num_body_queries`) to `__init__`, conditional instantiation of `pelvis_coord_queries` and `pelvis_decoder`, updated `_init_head_weights()` and `forward()` to use the dedicated pelvis decoder path when enabled.
- `code/config.py`: Added `use_decoupled_pelvis=True`, `pelvis_hidden_dim=256`, `pelvis_num_heads=4`, `num_body_queries=70` to the head config dict. The sole difference from design001 is `pelvis_num_heads=4` (lighter pelvis head, 4 instead of 8 attention heads, with wider per-head dimension 64 vs 32).

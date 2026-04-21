**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `pose3d_transformer_head.py` — Added module-level `_build_skeleton_attn_bias()` function that constructs a `(70, 70)` float32 tensor with `+0.5` for each kinematic edge (body SMPL-X tree + hand finger chains + face) bidirectionally, and `-0.5` on the pelvis diagonal `[0,0]`; all other entries remain `0.0`.
- `pose3d_transformer_head.py` — `_DecoderLayer.__init__` accepts `attn_bias_init: torch.Tensor | None` and registers `self.attn_bias = nn.Parameter(attn_bias_init.float().clone())` when provided, else falls back to `torch.zeros`; `forward` passes `attn_mask=self.attn_bias` to `self.self_attn`.
- `pose3d_transformer_head.py` — `Pose3dTransformerHead.__init__` accepts `attn_bias_type: str = 'none'` and builds the skeleton init tensor when `attn_bias_type == 'skeleton_init'`, passing it to `_DecoderLayer` as `attn_bias_init`.
- `config.py` — Added `attn_bias_type='skeleton_init'` as a string literal to the head dict so the skeleton-graph warm-start is activated.

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `pose3d_transformer_head.py` — `_DecoderLayer.__init__` accepts `num_joints: int = 70` and `attn_bias_mode: str = 'none'`; stores `self.num_heads` and `self.attn_bias_mode`; registers `self.attn_bias = nn.Parameter(torch.zeros(num_heads, num_joints, num_joints))` for mode `'per_head'`, a `(num_joints, num_joints)` param for `'shared'`, or `None` for `'none'`.
- `pose3d_transformer_head.py` — `_DecoderLayer.forward` reads `B = queries.shape[0]`, expands the per-head bias to `(B * num_heads, J, J)` via `.unsqueeze(0).expand(B, -1, -1, -1).contiguous().reshape(...)` and passes it as `attn_mask`; falls through to the shared or no-bias paths via `elif`/`else`.
- `pose3d_transformer_head.py` — `Pose3dTransformerHead.__init__` accepts `attn_bias_type: str = 'none'` and passes it as `attn_bias_mode` to `_DecoderLayer`, enabling the per-head bias for this design.
- `config.py` — Added `attn_bias_type='per_head'` as a string literal to the head dict to activate the 8-head independent bias matrices.

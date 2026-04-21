**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Same implementation as Design 001 plus self-attention temperature support. When `use_self_temp=True`, `__init__` creates `self.self_temp = nn.Parameter(torch.ones(num_joints))`; `forward()` passes it as `self_temp` to `_DecoderLayer.forward()`. The decoder layer applies `_temp_scaled_attn` to the self-attention block with `query=q, key=q, value=q` (all `(B, num_joints, D)`). Temperatures are stored only in the head (not in `_DecoderLayer`) to prevent duplicate `nn.Parameter` registration. Both `cross_attn._qkv_same_embed_dim` and `self_attn._qkv_same_embed_dim` assertions are present.

`code/config.py`: Set `use_cross_temp=True`, `use_self_temp=True`, `temp_log_space=False`, `temp_reg_weight=0.0` as bool/float literals in the head dict. No Python import statements introduced.

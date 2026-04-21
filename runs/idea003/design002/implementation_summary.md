**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pose3d_transformer_head.py`: Added `query_cond_type: str = 'mlp'` parameter to `__init__`. After `self.decoder_layer` creation, added `self.query_cond_net` as an `nn.Sequential` with two linear layers (`hidden_dim → hidden_dim//2 → num_joints*hidden_dim`) separated by `nn.GELU()`. Both linear layers are initialised with `trunc_normal_(std=0.02)` weights and zero biases. In `forward()`, replaced the static query broadcast with content-adaptive queries: mean-pools spatial tokens (after pos_enc), passes through the bottleneck MLP, reshapes offsets to `(B, num_joints, hidden_dim)`, and adds them to the static queries before the decoder layer.

- `code/config.py`: Added `query_cond_type='mlp'` to the `head` dict in the model config so the new parameter is passed to `Pose3dTransformerHead`.

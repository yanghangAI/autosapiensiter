**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pose3d_transformer_head.py`: Added `query_cond_type: str = 'linear'` parameter to `__init__`. After `self.decoder_layer` creation, added `self.query_cond_net = nn.Linear(hidden_dim, num_joints * hidden_dim)` initialised with `trunc_normal_(std=0.02)` weights and zero bias. In `forward()`, replaced the static query broadcast with content-adaptive queries: mean-pools spatial tokens (after pos_enc) into a global feature, projects through `query_cond_net`, reshapes to `(B, num_joints, hidden_dim)`, and adds the offset to the static queries before passing to the decoder layer.

- `code/config.py`: Added `query_cond_type='linear'` to the `head` dict in the model config so the new parameter is passed to `Pose3dTransformerHead`.

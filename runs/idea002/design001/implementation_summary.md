**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added `decouple_pelvis: bool = False` constructor parameter stored as `self.decouple_pelvis`. When `True`, an `nn.Embedding(1, hidden_dim)` named `pelvis_query` is created and initialised with `trunc_normal_(std=0.02)`. In `forward()`, the pelvis token is obtained by running the dedicated `pelvis_query` through the existing `decoder_layer`'s cross-attention sub-components directly (skipping self-attention), so pelvis depth/UV heads read from this decoupled token instead of `decoded[:, 0, :]`. Updated module docstring to document the new optional pathway.

`code/config.py`: Added `decouple_pelvis=True` to the `head` dict to activate the decoupled pelvis query pathway. All other config values are identical to baseline.

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added `decouple_pelvis: bool = False` and `pelvis_decoder_type: str = 'shared'` constructor parameters. When `decouple_pelvis=True` and `pelvis_decoder_type='independent'`, creates `self.pelvis_query` (nn.Embedding(1, hidden_dim)) and `self.pelvis_decoder` (a fully independent `_DecoderLayer` instance with its own weights). In `forward()`, the pelvis token is obtained by running `pelvis_query` through `pelvis_decoder`'s cross-attention sub-components only (self-attention skipped for single-token efficiency), completely decoupling pelvis localisation from body joint decoding. Updated module docstring accordingly.

`code/config.py`: Added `decouple_pelvis=True` and `pelvis_decoder_type='independent'` to the `head` dict to activate the independent pelvis decoder pathway. All other config values are identical to baseline.

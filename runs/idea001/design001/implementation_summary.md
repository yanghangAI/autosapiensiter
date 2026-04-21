**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added `num_decoder_layers: int = 2` constructor parameter and replaced the single `self.decoder_layer = _DecoderLayer(...)` with `self.decoder_layers = nn.ModuleList([_DecoderLayer(...) for _ in range(num_decoder_layers)])`. The forward loop iterates over all layers in sequence, with the final layer's output driving all downstream projections (joints_out, depth_out, uv_out). No auxiliary losses added — this is the pure capacity ablation.

`code/config.py`: Added `num_decoder_layers=2` to the head config dict so the model is constructed with 2 stacked decoder layers. All other hyperparameters (LR, schedule, batch size, seed, loss weights) are identical to baseline.

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added `num_decoder_layers: int = 4` and `aux_loss_weight: float = 0.4` constructor parameters. Replaced single `self.decoder_layer` with `self.decoder_layers = nn.ModuleList([...])` of 4 independent `_DecoderLayer` instances. Output projections (`joints_out`, `depth_out`, `uv_out`) are each defined once — `joints_out` is intentionally a single shared `Linear(hidden_dim, 3)` called at every decoder layer output (including intermediates), enforcing a common pose-space geometry across refinement stages. The forward loop collects all layer outputs; the return dict includes `intermediate_joints` projected via the shared `joints_out`. In `loss()`, auxiliary joint losses (body joints 0-21 only) are added for each of the 3 intermediate layers at weight `aux_loss_weight=0.4`; pelvis losses remain final-layer-only.

`code/config.py`: Added `num_decoder_layers=4` and `aux_loss_weight=0.4` to the head config dict. All other hyperparameters (LR, schedule, batch size, seed, loss weights) are identical to baseline.

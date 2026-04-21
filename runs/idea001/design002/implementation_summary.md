**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added `num_decoder_layers: int = 3` and `aux_loss_weight: float = 0.4` constructor parameters. Replaced single `self.decoder_layer` with `self.decoder_layers = nn.ModuleList([...])` of 3 independent `_DecoderLayer` instances. The forward loop collects all intermediate outputs; the return dict includes `intermediate_joints` (list of projected intermediate layer outputs using the shared `joints_out` head) so that `loss()` can supervise them. In `loss()`, added a loop over `pred['intermediate_joints']` to compute auxiliary joint losses (restricted to body joints 0-21) weighted at `aux_loss_weight=0.4` each; pelvis depth and UV losses remain on the final layer only.

`code/config.py`: Added `num_decoder_layers=3` and `aux_loss_weight=0.4` to the head config dict. All other hyperparameters (LR, schedule, batch size, seed, loss weights) are identical to baseline.

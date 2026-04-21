**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Replaced single 70-query `decoder_layer` with an `nn.ModuleList` of 2 `_DecoderLayer` instances operating over 22 body-only queries (`nn.Embedding(22, 256)`); added `hand_proj = nn.Linear(22*256, 48*3)` to recover hand joints from flattened body query features; `forward()` iterates through decoder layers collecting intermediate outputs, then concatenates body joints (via `joints_out`) and hand joints (via `hand_proj`) to produce `(B, 70, 3)`; `loss()` adds auxiliary hand loss (`hand_aux_loss_weight=0.1`) and supports optional intermediate body loss (`aux_body_loss_weight=0.0` → disabled for this design).
- `code/config.py`: Updated `head=dict(...)` to pass `num_body_queries=22`, `num_decoder_layers=2`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.0` to the new head constructor; all other config values (optimizer, LR schedule, data pipeline, hooks) are unchanged from baseline.

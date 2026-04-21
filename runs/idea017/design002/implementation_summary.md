**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Identical implementation to design001 — 22-query body decoder with 2-layer `nn.ModuleList`, linear hand projection, and shared `joints_out` for intermediate supervision; the `aux_body_loss_weight > 0.0` branch in `loss()` is now active (weight=0.4), emitting `loss/joints_aux_0/train` from the layer-1 intermediate output to force early convergence.
- `code/config.py`: Updated `head=dict(...)` with `num_body_queries=22`, `num_decoder_layers=2`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.4`; the only difference from design001 config is `aux_body_loss_weight=0.4` (was 0.0).

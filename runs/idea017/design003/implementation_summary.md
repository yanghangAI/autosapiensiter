**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**
- `code/pose3d_transformer_head.py`: Same architecture as design001/002 but with `num_decoder_layers=3` passed via config; the intermediate loss block in `loss()` uses the formula `intermediate_weights[k] = aux_body_loss_weight * (1.0 + 0.5 * k)` which yields [0.4, 0.6] for n_inter=2 (layers 1 and 2), emitting `loss/joints_aux_0/train` (w=0.4) and `loss/joints_aux_1/train` (w=0.6) alongside the final `loss/joints/train` (w=1.0); the escalating-weight curriculum forces progressive pose quality at each decoder layer.
- `code/config.py`: Updated `head=dict(...)` with `num_body_queries=22`, `num_decoder_layers=3`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.4`; the only difference from design002 config is `num_decoder_layers=3` (was 2).

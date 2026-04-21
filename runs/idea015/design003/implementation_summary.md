**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Same refactored head file as design001 (all four new params, ModuleList decoder, `_forward_with_intermediates`, slot-attention modules). With `num_decoder_layers=2`, the model runs two `_DecoderLayer` passes both cross-attending over the same K=32 super-tokens (computed once). With `aux_loss_weight=0.4`, `loss()` computes an auxiliary joint loss on the first decoder layer's output using the shared `joints_out` projection with key `'loss/joints_aux_0/train'`, weighted by 0.4, restricted to body joints (indices 0-21). Primary losses are computed from the final (second) decoder layer output, identical to baseline. This design uses `num_super_tokens=32`, `slot_pos_init=False`, `num_decoder_layers=2`, `aux_loss_weight=0.4`.

`code/config.py`: Added four new head kwargs as literals (`num_super_tokens=32`, `slot_pos_init=False`, `num_decoder_layers=2`, `aux_loss_weight=0.4`) to the `head=dict(...)` block. All other config values unchanged.

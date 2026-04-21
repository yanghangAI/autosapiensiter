**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added four new constructor parameters (`num_super_tokens=0`, `slot_pos_init=False`, `num_decoder_layers=1`, `aux_loss_weight=0.0`) and corresponding attributes. Replaced `self.decoder_layer` (single `_DecoderLayer`) with `self.decoder_layers` (an `nn.ModuleList` of `num_decoder_layers` layers). When `num_super_tokens > 0`, adds `self.slot_queries` (nn.Embedding), `self.slot_attn` (nn.MultiheadAttention, batch_first=True), and `self.slot_norm` (nn.LayerNorm) for slot-attention pooling; slot queries are initialized with trunc_normal (std=0.02). Extracted core forward computation into `_forward_with_intermediates()` that returns both the final pred dict and a list of intermediate decoder outputs; `forward()` calls it and discards intermediates, `loss()` calls it and uses intermediates for auxiliary losses (when `aux_loss_weight > 0`). In the forward path, 960 spatial tokens are optionally compressed to K super-tokens before being passed to all decoder layers. This design uses `num_super_tokens=32`, `num_decoder_layers=1`, `aux_loss_weight=0.0`.

`code/config.py`: Added four new head kwargs as literals (`num_super_tokens=32`, `slot_pos_init=False`, `num_decoder_layers=1`, `aux_loss_weight=0.0`) to the `head=dict(...)` block. All other config values unchanged.

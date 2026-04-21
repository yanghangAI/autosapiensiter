**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Same refactored head file as design001 (all four new params, ModuleList decoder, `_forward_with_intermediates`, slot-attention modules). When `slot_pos_init=True`, `_init_head_weights()` overwrites slot query weights with block-averaged 2D sinusoidal positional encodings: the 24×40 feature grid is partitioned into 8×8=64 non-overlapping blocks (3 rows × 5 cols each), and each slot is initialized with the mean positional encoding of its block, spatially grounding the slots at init. An assertion enforces `num_super_tokens == 64` when `slot_pos_init=True`. This design uses `num_super_tokens=64`, `slot_pos_init=True`, `num_decoder_layers=1`, `aux_loss_weight=0.0`.

`code/config.py`: Added four new head kwargs as literals (`num_super_tokens=64`, `slot_pos_init=True`, `num_decoder_layers=1`, `aux_loss_weight=0.0`) to the `head=dict(...)` block. All other config values unchanged.

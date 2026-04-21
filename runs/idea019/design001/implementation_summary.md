**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added the `_DeformableDecoderLayer` class implementing per-query deformable sparse cross-attention (K_s=8 bilinear-sampled points per query via a shared offset MLP and learnable per-query reference points, replacing the standard dense 960-token cross-attention). Updated `Pose3dTransformerHead.__init__` to accept `deform_num_points`, `deform_hidden_dim`, `num_body_queries`, `num_decoder_layers`, `hand_aux_loss_weight`, and `aux_body_loss_weight` kwargs; uses an `nn.ModuleList` for decoder layers (with backward-compat `decoder_layer` alias); added `has_hand_proj` and `has_intermediate_sup` guards (inactive for design001 since `num_body_queries=70` and `aux_body_loss_weight=0.0`). Updated `forward()` to keep spatial features as a 2D grid `(B, hidden_dim, H, W)` for `grid_sample` in the deformable path, with AMP-safe dtype cast. Updated `loss()` to include optional intermediate body supervision and auxiliary hand loss (both inactive for design001). Updated `_init_head_weights()` to apply near-zero init to offset network and attention weight net outputs for stable cold-start.

`code/config.py`: Replaced the `head=dict(...)` block to add the new kwargs with literal values: `in_channels=1024`, `deform_num_points=8`, `deform_hidden_dim=64`, `num_body_queries=70`, `num_decoder_layers=1`, `hand_aux_loss_weight=0.0`, `aux_body_loss_weight=0.0`. All values are int/float literals; no Python import statements introduced.

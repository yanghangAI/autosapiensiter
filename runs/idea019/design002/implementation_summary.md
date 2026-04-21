**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Same comprehensive implementation as design001 — adds `_DeformableDecoderLayer` with K_s=8 deformable sparse cross-attention. For design002, the key difference from design001 is controlled entirely by config: `num_body_queries=22` causes `joint_queries` to be `nn.Embedding(22, 256)` and activates the `has_hand_proj` path (`hand_proj = Linear(22*256, 48*3)`). Hand joints 22–69 are recovered from flattened body query features via this linear projection, then concatenated to body joints for the final `(B, 70, 3)` output. The `hand_aux_loss_weight=0.1` activates the auxiliary hand loss in `loss()`. Near-zero init is applied to the deformable offset/attention weight networks for stable cold-start.

`code/config.py`: Replaced the `head=dict(...)` block with literal values specific to design002: `in_channels=1024`, `deform_num_points=8`, `deform_hidden_dim=64`, `num_body_queries=22`, `num_decoder_layers=1`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.0`. All values are int/float literals; no Python import statements introduced.

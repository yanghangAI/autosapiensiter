**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Modified `_DecoderLayer.forward()` to accept an optional `attn_logit_bias` argument (same pattern as designs 001 and 002). Added `num_body_queries`, `hand_aux_loss_weight`, `depth_gate_type`, and `depth_gate_sigma` kwargs to `Pose3dTransformerHead.__init__()`; replaced `nn.Embedding(num_joints, hidden_dim)` with `nn.Embedding(num_body_queries, hidden_dim)` so only 22 body queries are decoded; added `hand_proj: Linear(22*256, 48*3)` (trunc-normal initialized) that projects flattened 22-query decoder output to 48 hand joint coordinates, and concatenates with body joints to restore the full `(B, 70, 3)` output shape; added Gaussian depth gate probes (zero-initialized) that produce per-token log-gate logits passed to the decoder; added an auxiliary `loss/hand_aux/train` loss (weight 0.1) on hand joints [22–69] in `loss()`.

`code/config.py`: Added `num_body_queries=22`, `hand_aux_loss_weight=0.1`, `depth_gate_type='gaussian'`, and `depth_gate_sigma=1.0` as literal kwargs to the `model.head` dict, enabling the combined 22-query body-only decoder with fixed-sigma Gaussian depth gate.

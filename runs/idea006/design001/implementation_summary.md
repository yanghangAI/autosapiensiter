**Files changed:**
- `code/pose3d_transformer_head.py`

**Changes:**
- `_DecoderLayer.__init__`: Added `num_joints: int = 70` parameter and registered `self.attn_bias = nn.Parameter(torch.zeros(num_joints, num_joints))` after `self.dropout2`, providing a shared learnable additive bias over self-attention logits initialized to zero (baseline-identical start).
- `_DecoderLayer.forward`: Changed `self.self_attn(q, q, q)[0]` to `self.self_attn(q, q, q, attn_mask=self.attn_bias)[0]` so the bias is added to attention logits before softmax on every forward pass.
- `Pose3dTransformerHead.__init__`: Updated `_DecoderLayer` construction to pass `num_joints=num_joints` so the bias shape matches the actual joint count.

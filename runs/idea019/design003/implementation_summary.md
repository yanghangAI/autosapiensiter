**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Same comprehensive implementation as design001/002 — adds `_DeformableDecoderLayer` with K_s=8 deformable sparse cross-attention, plus all design002 features (22-query body-only decoder, hand linear projection). For design003, `num_decoder_layers=2` creates an `nn.ModuleList` of 2 independent `_DeformableDecoderLayer` instances (each with their own `ref_points`, `offset_net`, etc.). `aux_body_loss_weight=0.4` activates `has_intermediate_sup`, creating `intermediate_joints_out = nn.ModuleList([nn.Linear(256, 3)])` for intermediate body supervision after decoder layer 0. In `forward()`, the intermediate output of layer 0 is stored in `self._intermediate_decoded`; in `loss()`, `loss/joints_inter0/train` (weight 0.4) and `loss/hand_aux/train` (weight 0.1) are both added to the losses dict so MMEngine sums them into the total backward loss.

`code/config.py`: Replaced the `head=dict(...)` block with literal values specific to design003: `in_channels=1024`, `deform_num_points=8`, `deform_hidden_dim=64`, `num_body_queries=22`, `num_decoder_layers=2`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.4`. All values are int/float literals; no Python import statements introduced.

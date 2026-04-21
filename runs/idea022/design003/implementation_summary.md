**Files changed:**
- `pelvis_utils.py`
- `pose3d_transformer_head.py`
- `config.py`

**Changes:**

`pelvis_utils.py`: Added `project_joints_to_feat_grid` helper function (identical to design001/002) that projects absolute 3D joints to feature-grid coordinates, clamped to grid bounds.

`pose3d_transformer_head.py`: Same structural changes as design002 (added `_build_gaussian_bias`, extended `_DecoderLayer.forward` with optional `cross_attn_bias`, `nn.ModuleList` of decoder layers, new constructor parameters). Key difference from design002: added conditional creation of `self.bias_sigma = nn.Parameter(torch.ones(num_joints) * reproj_bias_sigma)` and `self.bias_gamma = nn.Parameter(torch.ones(num_joints) * reproj_bias_gamma)` when `reproj_bias_learnable=True`. In `loss()`, replaced the fixed-tensor sigma/gamma construction with a conditional branch: when `reproj_bias_learnable=True`, `sigma = F.softplus(self.bias_sigma).to(device=..., dtype=...)` (ensuring sigma > 0) and `gamma = self.bias_gamma.to(device=..., dtype=...)` are used; otherwise falls back to fixed full-tensors as in design002. Auxiliary loss with weight 0.4 is preserved (same as design002). This allows the model to learn per-joint focus widths: narrow for distal joints, broad for proximal joints.

`config.py`: Added the new head kwargs with `reproj_bias_learnable=True`, `aux_loss_weight=0.4`, all as float/bool/int literals.

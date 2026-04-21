**Files changed:**
- `pelvis_utils.py`
- `pose3d_transformer_head.py`
- `config.py`

**Changes:**

`pelvis_utils.py`: Added `project_joints_to_feat_grid` helper function (identical to design001) that projects absolute 3D joints to feature-grid coordinates, clamped to grid bounds.

`pose3d_transformer_head.py`: Same structural changes as design001 (added `_build_gaussian_bias`, extended `_DecoderLayer.forward` with optional `cross_attn_bias`, replaced single decoder layer with `nn.ModuleList`, added new constructor parameters). Key difference from design001: the intermediate layer-0 forward in `loss()` does NOT use `torch.no_grad()`, allowing gradients to flow through layer-0 weights from both the auxiliary loss and the full forward pass. Added an auxiliary body-joint loss block after the main losses: `losses['loss/joints_aux/train'] = aux_loss_weight * loss_joints_module(layer1_joints[:, 0:22], gt_joints[:, 0:22])`, which directly supervises layer-0 intermediate predictions to bootstrap reprojection bias quality from early training epochs.

`config.py`: Added the new head kwargs with `aux_loss_weight=0.4` and `reproj_bias_learnable=False`, all as float/bool/int literals.

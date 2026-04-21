**Files changed:**
- `pelvis_utils.py`
- `pose3d_transformer_head.py`
- `config.py`

**Changes:**

`pelvis_utils.py`: Added `project_joints_to_feat_grid` helper function that projects absolute 3D joints (camera-frame) to feature-grid coordinates (h_frac, w_frac) using the BEDLAM2 convention (X=forward, Y=left, Z=up), clamping outputs to valid grid bounds.

`pose3d_transformer_head.py`: Added `_build_gaussian_bias` module-level function to build per-sample dynamic Gaussian cross-attention bias tensors. Extended `_DecoderLayer.forward` to accept an optional `cross_attn_bias` argument that, when provided, is reshaped to `(B*nheads, J, H'W')` and passed as `attn_mask` to `nn.MultiheadAttention`. Replaced the single `decoder_layer` with a `nn.ModuleList` of `num_decoder_layers` decoder layers; added new constructor parameters (`num_decoder_layers`, `use_reproj_bias`, `reproj_bias_sigma`, `reproj_bias_gamma`, `reproj_bias_learnable`, `aux_loss_weight`, `feat_h`, `feat_w`) all with backward-compatible defaults. In `loss()`, before calling `self.forward()`, a pre-compute block runs layer-0 under `torch.no_grad()` to obtain intermediate joint/pelvis predictions, recovers absolute joint positions via `recover_pelvis_3d`, projects them to feature-grid coordinates via `project_joints_to_feat_grid`, and builds the `_reproj_bias` tensor that is consumed by the second decoder layer inside `forward()`. The bias is cleared at the end of `forward()` to prevent stale values from leaking into validation. Design A uses fixed `sigma=4.0`, `gamma=2.0` and `aux_loss_weight=0.0`.

`config.py`: Added the new head kwargs (`num_decoder_layers=2`, `use_reproj_bias=True`, `reproj_bias_sigma=4.0`, `reproj_bias_gamma=2.0`, `reproj_bias_learnable=False`, `aux_loss_weight=0.0`, `feat_h=40`, `feat_w=24`) as float/bool/int literals — fully compliant with the MMEngine no-import constraint.

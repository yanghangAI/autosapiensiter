**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Modified `_DecoderLayer.forward()` to accept an optional `attn_logit_bias` argument that is expanded to `(B*num_heads, Nq, N_spatial)` and passed as a float additive `attn_mask` to `nn.MultiheadAttention`. Added `depth_gate_type` and `depth_probe_loss_weight` kwargs to `Pose3dTransformerHead.__init__()`; when `depth_gate_type='gaussian_learnable_sigma'`, creates two zero-initialized linear probes and a learnable `log_sigma` parameter (init=0.0 → sigma=1.0); in `forward()`, computes the per-token Gaussian log-gate using `sigma = exp(log_sigma).clamp(min=0.01)` and caches `z_hat` as `self._depth_probe_z_hat`; in `loss()`, adds an auxiliary `loss/depth_probe/train` term (`depth_probe_loss_weight * smooth_l1(z_hat, gt_depth)`) that directly supervises the global depth probe to predict pelvis depth.

`code/config.py`: Added `depth_gate_type='gaussian_learnable_sigma'` and `depth_probe_loss_weight=0.1` as literal kwargs to the `model.head` dict, enabling the learnable-sigma depth gate with auxiliary probe supervision.

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Modified `_DecoderLayer.forward()` to accept an optional `attn_logit_bias` argument (shape `(B, num_spatial)`) that is expanded to `(B*num_heads, Nq, N_spatial)` and passed as a float additive `attn_mask` to `nn.MultiheadAttention`, enabling depth-gated cross-attention. Added `depth_gate_type` and `depth_gate_sigma` kwargs to `Pose3dTransformerHead.__init__()`, which when `depth_gate_type='gaussian'` creates two zero-initialized linear probes (`depth_probe_global: Linear(hidden_dim,1)` and `depth_probe_token: Linear(hidden_dim,1)`) and registers a fixed `depth_gate_sigma_buf` buffer; in `forward()`, these probes compute a per-token Gaussian log-gate logit (`-0.5 * ((z_tok - z_hat)/sigma)^2`) that is passed to the decoder layer, suppressing cross-attention to spatial tokens at implausible depths while defaulting to the baseline when both probes output zero at initialization.

`code/config.py`: Added `depth_gate_type='gaussian'` and `depth_gate_sigma=1.0` as literal kwargs to the `model.head` dict, activating the Gaussian depth gate with fixed sigma=1.0.

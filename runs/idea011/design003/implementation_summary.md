# Design 003 — Implementation Summary

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pose3d_transformer_head.py`: Identical to Design 001/002 changes, PLUS the construction of an independent pass-2 decoder layer. Added three new `__init__` kwargs (`num_refine_passes`, `shared_decoder`, `intermediate_supervision_weight`) stored as attributes. Built the zero-init `self.coord_enc` Sequential, then conditionally built `self.decoder_layer_2 = _DecoderLayer(hidden_dim, num_heads, dropout)` only when `shared_decoder=False` AND `num_refine_passes >= 2` (so Design 001/002 retain the exact same parameter set and no unused `decoder_layer_2` is created). `_init_head_weights` trunc-normal-inits the first `coord_enc` Linear and zero-inits both weight and bias of the second `coord_enc` Linear; `decoder_layer_2` uses PyTorch defaults (matching the treatment of `self.decoder_layer`). `forward()` short-circuits to baseline behaviour when `num_refine_passes <= 1`, otherwise runs pass 1, builds `queries_2 = decoded_1 + coord_enc(joints_1)`, and routes pass 2 through `self.decoder_layer_2` when `shared_decoder=False` (this branch is taken for Design 003) or through the shared `self.decoder_layer` otherwise. Residual joint output `joints_final = joints_1 + joints_residual` and pelvis from pass-2 token 0. `loss()` emits the `loss/joints_init/train` auxiliary term with weight `intermediate_supervision_weight=0.5` (body joints 0-21 only). The `torch.no_grad()` MPJPE block and `predict()` are unchanged. Absolute imports are preserved.

- `code/config.py`: Added three new kwargs at the end of the `head=dict(...)` block: `num_refine_passes=2`, `shared_decoder=False`, `intermediate_supervision_weight=0.5` — `shared_decoder=False` is the config-level difference from Design 002 that triggers the construction and use of `self.decoder_layer_2`. All other config values are identical to baseline. All new kwargs are int/bool/float literals, respecting the MMEngine-no-imports rule.

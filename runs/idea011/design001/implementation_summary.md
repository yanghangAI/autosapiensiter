# Design 001 — Implementation Summary

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pose3d_transformer_head.py`: Added three new `__init__` kwargs (`num_refine_passes: int = 1`, `shared_decoder: bool = True`, `intermediate_supervision_weight: float = 0.0`) stored as attributes. Built a new `self.coord_enc = nn.Sequential(Linear(3, hidden_dim), GELU, Linear(hidden_dim, hidden_dim))` module (placed between the decoder layer and the output projections). Extended `_init_head_weights` to trunc-normal-init the first `coord_enc` Linear and zero-init BOTH the weight AND bias of the last `coord_enc` Linear, so `coord_enc(joints_1) == 0` at init and training starts from an approximately baseline-equivalent state. Rewrote `forward()` to (1) short-circuit to the baseline single-pass path when `num_refine_passes <= 1`, and (2) otherwise run pass 1, read `joints_1`, build pass-2 queries as `decoded_cur + coord_enc(joints_cur)`, invoke `self.decoder_layer` (shared weights) for pass 2, read a residual via the shared `joints_out`, and return `joints_final = joints_1 + joints_residual`. Pelvis depth/UV are read from the pass-2 token 0. Extended `loss()` so that if `intermediate_supervision_weight > 0.0` and `joints_initial` is in the forward-output dict, an auxiliary `loss/joints_init/train` term (body joints 0-21 only) is added. For Design 001 the weight is 0.0, so the branch is not taken and the loss dict is identical in shape to the baseline. The `torch.no_grad()` MPJPE recording block and `predict()` are unchanged — both read `pred['joints']` which now holds the refined final joints. Absolute imports are preserved.

- `code/config.py`: Added three new kwargs at the end of the `head=dict(...)` block: `num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.0`. All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, seed, `custom_imports`, pretrained weights) are identical to baseline. All new kwargs are int/bool/float literals, respecting the MMEngine-no-imports rule.

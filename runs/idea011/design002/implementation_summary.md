# Design 002 — Implementation Summary

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

- `code/pose3d_transformer_head.py`: Identical to Design 001's head file. Added three new `__init__` kwargs (`num_refine_passes`, `shared_decoder`, `intermediate_supervision_weight`) stored as attributes. Built the zero-init `self.coord_enc` Sequential (Linear→GELU→Linear) between the decoder layer and the output projections; `_init_head_weights` trunc-normal-inits the first Linear and zero-inits both weight and bias of the second Linear. Rewrote `forward()` to short-circuit to baseline behaviour when `num_refine_passes <= 1`, otherwise run pass 1, build `queries_2 = decoded_1 + coord_enc(joints_1)`, run pass 2 using the shared `self.decoder_layer`, and return `joints_final = joints_1 + joints_residual` with pelvis depth/UV read from the pass-2 token 0. Extended `loss()` with the conditional `loss/joints_init/train` term (body joints 0-21, weighted by `self.intermediate_supervision_weight`); in Design 002 this weight is 0.5 (set via config), so the auxiliary term IS emitted. The `torch.no_grad()` MPJPE block and `predict()` are unchanged. Absolute imports are preserved.

- `code/config.py`: Added three new kwargs at the end of the `head=dict(...)` block: `num_refine_passes=2`, `shared_decoder=True`, `intermediate_supervision_weight=0.5` (the only config-level difference from Design 001 is the 0.5 intermediate-supervision weight). All other config values (optimizer, LR schedule, data pipeline, hooks, batch size, seed, `custom_imports`, pretrained weights) are identical to baseline. All new kwargs are int/bool/float literals, respecting the MMEngine-no-imports rule.

**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Added per-joint Laplace NLL uncertainty support to `Pose3dTransformerHead`. New `__init__` parameters (`use_per_joint_uncertainty`, `per_joint_uncertainty_mode`, `log_scale_out_features`, `laplace_entropy_weight`, `laplace_entropy_weight_start`, `laplace_entropy_weight_end`, `laplace_entropy_anneal_steps`) with baseline-preserving defaults. When `use_per_joint_uncertainty=True`, a `log_scale_out: Linear(hidden_dim, log_scale_out_features)` head is created and zero-initialised; `forward()` applies it to each of the 22 body query tokens independently to produce `pred['log_scale']` of shape `(B, 22, 1)` or `(B, 22, 3)`; `loss()` replaces the fixed SoftWeightSmoothL1 body joint loss with Laplace NLL (`w_ent * log(2s) + |pred - gt| / s`), with `log_s` clamped to `[-10, 5]` for AMP safety and `s` clamped to `min=1e-4` to prevent `log(0)`.

`code/config.py`: Added Design A config kwargs to `model.head`: `use_per_joint_uncertainty=True`, `per_joint_uncertainty_mode='shared_scalar'`, `log_scale_out_features=1`, `laplace_entropy_weight=1.0`, `laplace_entropy_anneal_steps=0`. All values are bool/int/float literals, MMEngine config compliant.

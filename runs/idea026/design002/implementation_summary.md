**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Same unified implementation as Design A. When `use_per_joint_uncertainty=True` and `log_scale_out_features=3`, `log_scale_out: Linear(hidden_dim, 3)` is applied to each body query token independently producing `pred['log_scale']` of shape `(B, 22, 3)`. In `loss()`, the Laplace NLL is computed element-wise with `s` shape `(B, 22, 3)` matching `abs_err` shape `(B, 22, 3)` — no broadcasting needed, each axis has an independent scale.

`code/config.py`: Added Design B config kwargs to `model.head`: `use_per_joint_uncertainty=True`, `per_joint_uncertainty_mode='per_axis'`, `log_scale_out_features=3`, `laplace_entropy_weight=1.0`, `laplace_entropy_anneal_steps=0`. All values are bool/int/float literals, MMEngine config compliant.

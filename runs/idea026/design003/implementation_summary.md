**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Same unified implementation as Design A/B. When `laplace_entropy_anneal_steps=500 > 0`, the entropy weight `w_ent` ramps linearly from `laplace_entropy_weight_start=0.1` to `laplace_entropy_weight_end=1.0` over the first 500 `loss()` calls (tracked via `self._loss_call_count`). This allows `s` to adapt freely in early training without the entropy term dominating, then enforces full entropy penalisation after ~5 epochs. After step 500, `w_ent` stays at 1.0 and behaviour is identical to Design A.

`code/config.py`: Added Design C config kwargs to `model.head`: `use_per_joint_uncertainty=True`, `per_joint_uncertainty_mode='shared_scalar'`, `log_scale_out_features=1`, `laplace_entropy_weight_start=0.1`, `laplace_entropy_weight_end=1.0`, `laplace_entropy_anneal_steps=500`. All values are bool/int/float literals, MMEngine config compliant.

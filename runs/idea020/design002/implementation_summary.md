**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Changes:**

`code/pose3d_transformer_head.py`: Same implementation as Design 001. The shared code supports both direct (`temp_log_space=False`) and log-space (`temp_log_space=True`) parameterisations. When `use_cross_temp=True` and `temp_log_space=True`, `__init__` creates `self.log_cross_temp = nn.Parameter(torch.zeros(num_joints))` instead of `self.cross_temp`; `forward()` computes `tau = softplus(self.log_cross_temp)` at runtime and passes it as `cross_temp` to the decoder layer. Temperatures are stored only in the head (not in `_DecoderLayer`) to prevent duplicate `nn.Parameter` registration. The `loss()` method adds `loss/temp_reg/train = temp_reg_weight * log_cross_temp.pow(2).mean()` when `temp_reg_weight > 0` and `log_cross_temp` exists.

`code/config.py`: Set `use_cross_temp=True`, `use_self_temp=False`, `temp_log_space=True`, `temp_reg_weight=0.01` as float/bool literals in the head dict. No Python import statements introduced.

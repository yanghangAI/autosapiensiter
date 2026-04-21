**Files changed:**
- `code/pose3d_transformer_head.py`
- `code/config.py`

**Note:** The `setup-design` CLI placed files in `code/code/` (an extra nesting level due to passing `design001/code` as destination); files were manually moved up to `code/` to match the expected layout used by `slurm_test.sh`. The `output_dir` in `config.py` was also corrected from `…/design001/code` to `…/design001` to match baseline convention.

**Changes:**

`code/pose3d_transformer_head.py`: Added `num_body_queries: int = 22` kwarg to `__init__` (stored as `self.num_body_queries`); changed `self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)` so decoder self-attention and cross-attention operate over 22 body queries only (reducing self-attention from 70×70 to 22×22). In `forward()`, after decoding 22 body queries, `joints_out` produces `(B, 22, 3)`; the output is then zero-padded with `torch.zeros(B, 48, 3)` and concatenated to yield the required `(B, 70, 3)` shape — the pad region has no gradient by default. `self.num_joints` remains 70 for correct `predict()` behaviour.

`code/config.py`: Added `num_body_queries=22` as an integer literal to the `model.head` dict so MMEngine passes the new kwarg to the head constructor. All other config values are identical to baseline.

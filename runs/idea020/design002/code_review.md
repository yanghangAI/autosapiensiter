# Code Review — idea020/design002

**Verdict: APPROVED**

## Checklist

### review-check-implementation
PASS.

### Files Changed
`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design. PRESENT.
- `code/config.py` — required by design. PRESENT and correct.

No extra files changed. `pelvis_utils.py` and `train.py` are unchanged (verified by diff against baseline).

### Code vs Design Fidelity

**`pose3d_transformer_head.py`:**
The head file is identical to design001's implementation. Both designs share a single implementation that handles both `temp_log_space=False` and `temp_log_space=True` via runtime branching in `__init__` and `forward()`. This is the correct approach documented in the implementation summary.

Key design002-specific behaviours present in the shared code:
- When `use_cross_temp=True` and `temp_log_space=True`: creates `self.log_cross_temp = nn.Parameter(torch.zeros(num_joints))` — correct.
- In `forward()`: computes `cross_temp = torch.nn.functional.softplus(self.log_cross_temp)` at runtime — correct.
- In `loss()`: adds `losses['loss/temp_reg/train'] = temp_reg_weight * log_cross_temp.pow(2).mean()` when `temp_reg_weight > 0` and `log_cross_temp` exists — correct.

**Minor deviation — clamp value:**
Design002 spec specifies `tau = temperature.clamp(min=1e-6)` for the log-space path (since `softplus` guarantees positivity far above 1e-6). The actual implementation uses `clamp(min=0.1)` uniformly. This is a conservative (stricter) bound that is functionally benign: `softplus(x) >= 0.1` for `x >= softplus_inv(0.1) ≈ -2.25`, and in practice log-temps far from -2.25 will not reach this bound. The design's intent ("safety guard — do not remove") is satisfied. This does not warrant rejection.

**`config.py`:**
- `use_cross_temp=True`, `use_self_temp=False`, `temp_log_space=True`, `temp_reg_weight=0.01` — all correct literals, no import statements.

### Invariants
- `pelvis_utils.py`: unchanged (diff exits 0).
- `train.py`: unchanged (diff exits 0).

### Test Output
- First test run (55859711) succeeded: model loaded, 1 epoch completed, `loss/temp_reg/train: 0.000000` visible in SLURM log (correct — L2 reg on `torch.zeros` at init is 0), training finished cleanly.
- Second test run (55859756) failed with `FileNotFoundError` for checkpoint — this is an infra race condition where the checkpoint from the first run was deleted before the second run could find it. Not a code defect.
- Third test run (55859814) succeeded: fresh start, 1 epoch completed, training finished cleanly.
- `iter_metrics.csv`: 72 rows for epoch 1. Note: `loss/temp_reg/train` does not appear as a column in `iter_metrics.csv` — this is expected because `MetricsCSVHook._ITER_COLS` is hardcoded and is an invariant file. The loss is still computed and logged by MMEngine (visible in SLURM log). This is not a defect.

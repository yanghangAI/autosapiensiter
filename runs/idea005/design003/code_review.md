## Code Review — idea005/design003

**Date:** 2026-04-16
**Verdict: APPROVED**

---

### Pre-flight check

`python scripts/cli.py review-check-implementation runs/idea005/design003` — PASSED.

---

### Files-changed audit

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design.
- `code/config.py` — required by design.

No files were changed that are not specified in `design.md`. `pelvis_utils.py` and `train.py` confirmed identical to baseline (diff clean).

---

### Change-by-change verification

**`pose3d_transformer_head.py`**

1. `use_uncertainty_weighting: bool = False`, `uncertainty_pelvis_only: bool = False`, and `joint_loss_scale: float = 1.0` all added as constructor parameters — present (lines 161-163). All stored as instance attributes (lines 175-178). Correct.
2. `nn.Parameter` registrations gated on `self.uncertainty_pelvis_only`: `log_var_depth` and `log_var_uv`, both `torch.zeros(1)` — present (lines 184-186). Same logic as design002. Correct.
3. `loss()` applies `self.joint_loss_scale` as a fixed multiplier to `raw_joints` **before** the conditional branch (line 309):
   ```python
   raw_joints = self.joint_loss_scale * self.loss_joints_module(...)
   ```
   This is exactly as specified in the design: `joint_loss_scale` takes effect whether or not `uncertainty_pelvis_only` is True. Correct.
4. `uncertainty_pelvis_only` branch (lines 316-321): depth and UV use uncertainty formula; `raw_joints` (already scaled) is assigned directly. Correct.
5. `_train_mpjpe` and `_train_mpjpe_abs` computations (lines 331-337) do NOT use `joint_loss_scale` — they compute plain MPJPE in mm from raw predictions. Correct; design explicitly states this invariant.
6. `__init__` signature matches the design's specified full signature including all three flags from designs 001, 002, and 003.
7. `joint_loss_scale` is stored as `self.joint_loss_scale` (a plain float), NOT as `nn.Parameter`. Correct; design specifies it is a pure Python float multiplier, not a learnable parameter.

**`config.py`**

- `uncertainty_pelvis_only=True` set in head dict (line 146). Correct.
- `joint_loss_scale=2.0` set in head dict (line 147). Correct; design specifies 2.0 = 0.67/0.33.
- `use_uncertainty_weighting` NOT set (defaults to False) — confirmed. Correct.
- `loss_weight_depth=1.0`, `loss_weight_uv=1.0` retained. Correct.
- No Python `import` statements. Correct.
- `persistent_workers=False` in both dataloaders. Correct.
- `output_dir` points to `runs/idea005/design003`. Correct.
- All other hyperparameters match design.

---

### Invariant check

- `pelvis_utils.py` — diff clean vs. baseline.
- `train.py` — diff clean vs. baseline.
- No modifications to evaluation metric, dataset, transforms, backbone, data preprocessor, or infra files.
- Joint loss restricted to body joints indices 0-21 — preserved.

---

### Test output check

- Epoch 1 completed successfully. Training log shows `loss/joints/train: 0.376784` (approximately 2× the 0.192 seen in design001/002, consistent with `joint_loss_scale=2.0`), `loss/depth/train: 1.565473`, `loss/uv/train: 0.110974`. The 2× joint loss scaling is reflected correctly in the logged loss values.
- Validation produced all required metric columns: `composite_val=496.20`, `mpjpe_body_val=434.15`, `mpjpe_pelvis_val=622.18`. Body MPJPE is lower than design001/002 at epoch 1 (434 vs 443), consistent with stronger joint loss signal.
- `metrics.csv` written correctly.
- No runtime errors in SLURM output.
- Checkpoint saved at epoch 1.

---

### Conclusion

Implementation is fully faithful to `design.md`. All three required changes are present and correct: `joint_loss_scale=2.0` multiplier applied before the conditional, `uncertainty_pelvis_only` depth/UV uncertainty, `_train_mpjpe` not affected by scale. `joint_loss_scale` is correctly a plain float, not an `nn.Parameter`. The 2× scaling is confirmed visible in training logs. No invariant files modified. Test run completed without errors. **APPROVED.**

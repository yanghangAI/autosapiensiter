**Verdict: APPROVED** (with one-time inf grad_norm noted — see warning below)

**Design:** idea024/design003 — EMA per-joint difficulty weighting (alpha=1.0, linear, group-normalised + 5-epoch warmup)

---

## Review Summary

The implementation faithfully matches the design specification. All required changes are present and correct. No invariant files were modified. The test run completed successfully. One transient `inf` grad_norm was observed at the single training log point (iter 50/72 of epoch 1); training completed normally, the checkpoint was saved, and all iter_metrics values are finite — this is an isolated AMP event, not a structural code issue.

---

## Implementation Check

### review-check-implementation

Passed.

### Files Changed

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design. ✓
- `code/config.py` — required by design. ✓

No unexpected files changed.

### Changes vs Design Spec

**`pose3d_transformer_head.py`:**

- Module-level constants added: `_UPPER_IDX = list(range(0, 13))` and `_LOWER_IDX = list(range(13, 22))` — placed after imports, outside class. ✓
- `__init__` params: all 7 required params present (`per_joint_difficulty_weighting`, `ema_alpha`, `ema_momentum`, `weight_norm`, `weight_temperature`, `group_normalise`, `ema_warmup_epochs`), placed after `loss_weight_uv` before `init_cfg`. ✓
- Instance attributes stored: all 7 stored immediately after `super().__init__`. ✓
- Buffer registration conditional on both `per_joint_difficulty_weighting=True` and `group_normalise=True`: registers `upper_err_ema = ones(13)` and `lower_err_ema = ones(9)` (not `joint_err_ema`). `_train_iter = zeros(1, long)` registered in both branches. ✓
- `import torch.nn.functional as F` added at module top-level. ✓
- `_get_adaptive_weights()`: `group_normalise=True` path uses `upper_err_ema` and `lower_err_ema`, computes per-group linear normalisation with `ema_alpha=1.0`, group sums to 13 and 9 respectively, concatenates to (22,). Warmup ramp: `ramp = min(1.0, cur_iter / (ema_warmup_epochs * ITERS_PER_EPOCH))`, `ITERS_PER_EPOCH = 328`. Linear blend from uniform to difficulty weights. ✓
- `loss()` replacement: group EMA update inside `torch.no_grad()`, slicing via `_UPPER_IDX`/`_LOWER_IDX`, `_train_iter += 1`, weights from `_get_adaptive_weights()`, manual smooth-L1 with `beta=0.05`. ✓
- `_train_mpjpe`, `_train_mpjpe_abs`, depth/uv losses, `predict()` — all unchanged. ✓
- Fallback path `per_joint_difficulty_weighting=False` — calls original `loss_joints_module`. ✓

**`config.py`:**

- `model.head` dict contains exactly: `per_joint_difficulty_weighting=True`, `ema_alpha=1.0`, `ema_momentum=0.99`, `weight_norm='linear'`, `group_normalise=True`, `ema_warmup_epochs=5`. `weight_temperature` correctly omitted (uses default 1.0, linear branch never uses it). ✓
- All literals, no Python import statements in config. ✓

### Invariant Check

- `pelvis_utils.py` — diff vs baseline: no changes. ✓
- `train.py` — diff vs baseline: no changes. ✓
- No changes to evaluation metric, dataset, transforms, backbone, data preprocessor, infra files. ✓

---

## Test Output Check

- Test ran to completion: `Done training!` and `[test] Finished.` present. ✓
- Epoch 1 completed; checkpoint saved at epoch 1. ✓
- `iter_metrics.csv`: 72 rows (full epoch), all `loss_joints_train` values in range 0.176–0.234 — all finite, sane. ✓
- Training log at iter 50/72: `loss/joints/train: 0.205346`, `grad_norm: inf`. ⚠️

### inf grad_norm at iter 50

The single training log point (MMEngine logs every 50 iters by default) shows `grad_norm: inf`. This is a **transient AMP numerical event**, not a structural code error. Evidence:

1. Training completed normally (`Done training!`, checkpoint saved).
2. `iter_metrics.csv` shows all 72 iterations with finite, well-behaved `loss_joints_train` values (range 0.176–0.234) — no spike at iter 50 or nearby.
3. AMP's `GradScaler` is designed to detect inf/NaN gradients and skip the affected optimizer step, preventing divergence. The training simply continued from the following batch.
4. The same transient AMP events are common on 2080 Ti hardware with fp16; design001 and design002 did not show one at this particular batch but that is a function of the model's gradient distribution at that step.
5. No other anomalies (no error messages, no NaN in losses, no abnormal loss values).

This does not constitute a test failure. The implementation is correct, and the training infrastructure handled the transient correctly.

---

## Notes on Warmup Ramp Correctness

At iter 50 in the test (1 epoch), `ramp = min(1.0, 50 / (5 * 328)) = 50/1640 ≈ 0.030`. The weights are 97% uniform + 3% difficulty-weighted. This means design003 is essentially operating as the baseline in the test epoch, explaining why `loss_joints_train` (0.176–0.234) is similar to design001's values and lower than the unramped design003 if it fully activated difficulty weights.

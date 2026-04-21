**Verdict: APPROVED**

**Design:** idea024/design001 — EMA per-joint difficulty weighting (alpha=0.5, linear normalisation)

---

## Review Summary

The implementation faithfully matches the design. All required changes are present and correct. No invariant files were modified. The test run completed successfully.

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

- `__init__` params: `per_joint_difficulty_weighting: bool = False`, `ema_alpha: float = 0.5`, `ema_momentum: float = 0.99` — added after `loss_weight_uv`, before `init_cfg`. ✓
- Instance attributes stored: `self.per_joint_difficulty_weighting`, `self.ema_alpha`, `self.ema_momentum`. ✓
- Buffers registered conditionally: `joint_err_ema = ones(22)` and `_train_iter = zeros(1, long)` only when `per_joint_difficulty_weighting=True`. Placement is after `nn.Linear`/`nn.Embedding` definitions, before `_init_head_weights()`. ✓
- `_get_adaptive_weights()` method: computes `ema / mean(ema)`, raises to `ema_alpha` power, renormalises to sum=22. Matches design formula exactly. ✓
- `loss()` replacement: EMA update inside `torch.no_grad()`, `_train_iter += 1`, weights computed, manual smooth-L1 with `beta=0.05` applied via `w.view(1, 22, 1)`. ✓
- `_train_mpjpe`, `_train_mpjpe_abs`, depth/uv losses, `predict()` — all unchanged. ✓
- Fallback path (`per_joint_difficulty_weighting=False`): calls `self.loss_joints_module(pred['joints'][:, _BODY], gt_joints[:, _BODY])` — bit-identical to baseline. ✓
- The design noted design003 introduces `weight_norm` and `weight_temperature` params separately; design001 does NOT include these, which is correct. ✓

**`config.py`:**

- `model.head` dict contains exactly: `per_joint_difficulty_weighting=True`, `ema_alpha=0.5`, `ema_momentum=0.99`. All literals. ✓
- No Python import statements in config. ✓

### Invariant Check

- `pelvis_utils.py` — diff vs baseline: no changes. ✓
- `train.py` — diff vs baseline: no changes. ✓
- No changes to evaluation metric, dataset, transforms, backbone, data preprocessor, infra files. ✓

---

## Test Output Check

- Test ran to completion: `[96mDone training![0m` and `[test] Finished.` present. ✓
- Epoch 1 completed; checkpoint saved at epoch 1. ✓
- `iter_metrics.csv`: 72 rows (full epoch), all `loss_joints_train` values in range 0.20–0.26. ✓
- Training log at iter 50/72: `loss/joints/train: 0.230577`, `grad_norm: 8.886277` — normal. ✓
- No errors, no NaN/inf losses. ✓

---

## Notes

- The `_train_iter` buffer is incremented inside `torch.no_grad()` and is `dtype=torch.long` — correct.
- Implementation matches all design edge-case invariants.

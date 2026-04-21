## Code Review — idea005/design001

**Date:** 2026-04-16
**Verdict: APPROVED**

---

### Pre-flight check

`python scripts/cli.py review-check-implementation runs/idea005/design001` — PASSED.

---

### Files-changed audit

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design.
- `code/config.py` — required by design.

No files were changed that are not specified in `design.md`. `pelvis_utils.py` and `train.py` are present but confirmed identical to baseline (diff clean).

---

### Change-by-change verification

**`pose3d_transformer_head.py`**

1. `use_uncertainty_weighting: bool = False` added as constructor parameter — present (line 161). Stored as `self.use_uncertainty_weighting` (line 174). Correct.
2. `nn.Parameter` registrations gated on `self.use_uncertainty_weighting`: `log_var_joints`, `log_var_depth`, `log_var_uv`, all `torch.zeros(1)` — present (lines 180-183). Correct. NOT added when flag is False — correct.
3. `loss()` computes raw losses first (lines 306-311), then applies uncertainty branch (lines 313-319):
   - Local variables `lv_j`, `lv_d`, `lv_u` with `.clamp(-4.0, 4.0)` — correct, gradients flow through clamp.
   - Formula `torch.exp(-lv_j) * raw_joints + lv_j` matches design exactly.
   - Else branch passes raw losses unchanged — correct, preserves baseline.
4. `_train_mpjpe` and `_train_mpjpe_abs` computations are unchanged (lines 329-335) — correct.
5. `__init__` signature matches the design's specified full signature (all parameters present in correct order, `init_cfg` last).

**`config.py`**

- `use_uncertainty_weighting=True` is set in head dict (line 146) — correct.
- `loss_weight_depth=1.0` and `loss_weight_uv=1.0` retained (lines 144-145) — correct (no-ops as specified).
- No Python `import` statements in config — correct (uses `__import__()`).
- `persistent_workers=False` in both dataloaders — correct (lines 175, 193).
- `output_dir` points to `runs/idea005/design001` — correct.
- All other hyperparameters (AdamW lr=1e-4, wd=0.03, backbone lr_mult=0.1, clip_grad max_norm=1.0, accumulative_counts=8, seed=2026, warmup=3 epochs LinearLR factor 0.333, CosineAnnealingLR 3-20) — all match design exactly.

---

### Invariant check

- `pelvis_utils.py` — diff clean vs. baseline.
- `train.py` — diff clean vs. baseline.
- No modifications to evaluation metric, dataset, transforms, backbone, data preprocessor, or infra files.
- Joint loss restricted to body joints indices 0-21 (`_BODY = list(range(0, 22))`) — preserved.

---

### Test output check

- Epoch 1 completed successfully. Training log shows losses: `loss/joints/train: 0.191977`, `loss/depth/train: 1.551167`, `loss/uv/train: 0.110799` — all three tasks producing gradients normally under uncertainty weighting.
- Validation ran and produced all required metric columns: `composite_val=491.12`, `mpjpe_body_val=443.24`, `mpjpe_pelvis_val=588.34`, `mpjpe_rel_val=525.10`, `mpjpe_hand_val=496.20`, `mpjpe_abs_val=814.69`.
- `metrics.csv` written correctly with all CSV columns.
- No runtime errors or exceptions in SLURM output.
- Checkpoint saved at epoch 1.

Note: Epoch-1 metrics are far from the 20-epoch composite target (491 vs. target <163) — this is expected; the test run only executes 1 epoch to validate correctness.

---

### Conclusion

Implementation is fully faithful to `design.md`. All required changes are present and correct. No invariant files were modified. Test run completed without errors and produced all expected outputs. **APPROVED.**

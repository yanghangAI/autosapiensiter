## Code Review — idea005/design002

**Date:** 2026-04-16
**Verdict: APPROVED**

---

### Pre-flight check

`python scripts/cli.py review-check-implementation runs/idea005/design002` — PASSED.

---

### Files-changed audit

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design.
- `code/config.py` — required by design.

No files were changed that are not specified in `design.md`. `pelvis_utils.py` and `train.py` confirmed identical to baseline (diff clean).

---

### Change-by-change verification

**`pose3d_transformer_head.py`**

1. `use_uncertainty_weighting: bool = False` and `uncertainty_pelvis_only: bool = False` added as constructor parameters — present (lines 161-162). Both stored as instance attributes (lines 175-176). Correct.
2. `nn.Parameter` registrations gated on `self.uncertainty_pelvis_only`: only `log_var_depth` and `log_var_uv`, both `torch.zeros(1)` — present (lines 182-184). `log_var_joints` is correctly NOT registered in this design. Correct.
3. `loss()` computes raw losses (lines 307-312). `uncertainty_pelvis_only` branch (lines 314-319):
   - Joint loss is `raw_joints` with no uncertainty scaling — fixed weight 1.0, anchored. Correct.
   - `lv_d` and `lv_u` clamped to `[-4.0, 4.0]` via local variables. Correct (gradients flow).
   - Depth and UV use `torch.exp(-lv_d) * raw_depth + lv_d` and `torch.exp(-lv_u) * raw_uv + lv_u` — exact Kendall & Gal formula. Correct.
   - Else branch passes raw losses unchanged — preserves baseline. Correct.
4. `_train_mpjpe` and `_train_mpjpe_abs` unchanged (lines 329-335). Correct.
5. `__init__` signature matches the design's specified full signature including both `use_uncertainty_weighting` and `uncertainty_pelvis_only` parameters.

**`config.py`**

- `uncertainty_pelvis_only=True` is set in head dict (line 146). Correct.
- `use_uncertainty_weighting` is NOT set in config (defaults to False) — confirmed by inspection. Correct; design explicitly states both flags must not be active simultaneously.
- `loss_weight_depth=1.0` and `loss_weight_uv=1.0` retained — correct.
- No Python `import` statements — correct.
- `persistent_workers=False` in both dataloaders — correct.
- `output_dir` points to `runs/idea005/design002` — correct.
- All other hyperparameters match design (AdamW lr=1e-4, wd=0.03, backbone lr_mult=0.1, clip_grad max_norm=1.0, accumulative_counts=8, seed=2026, warmup 3 epochs factor 0.333, CosineAnnealingLR 3-20).

---

### Invariant check

- `pelvis_utils.py` — diff clean vs. baseline.
- `train.py` — diff clean vs. baseline.
- No modifications to evaluation metric, dataset, transforms, backbone, data preprocessor, or infra files.
- Joint loss restricted to body joints indices 0-21 — preserved.

---

### Test output check

- Epoch 1 completed successfully. Training log shows `loss/joints/train: 0.192060`, `loss/depth/train: 1.551162`, `loss/uv/train: 0.110799` — all three tasks producing gradients; joint loss value closely matches design001 (0.192 vs 0.192, same anchor), depth/UV differ slightly due to uncertainty reweighting. Behaviour is consistent with the design.
- Validation produced all required metric columns: `composite_val=491.04`, `mpjpe_body_val=443.25`, `mpjpe_pelvis_val=588.08`.
- `metrics.csv` written correctly.
- No runtime errors in SLURM output.
- Checkpoint saved at epoch 1.

---

### Conclusion

Implementation is fully faithful to `design.md`. All required changes are present and correct. `log_var_joints` is correctly absent. Joint loss is correctly anchored at fixed weight. No invariant files modified. Test run completed without errors. **APPROVED.**

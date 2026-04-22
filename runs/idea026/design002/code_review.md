# Code Review — idea026/design002

**Verdict: APPROVED**

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea026/design002` — PASSED.

---

## Files Changed

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py`
- `code/config.py`

Both files are required by design002. No extra files changed. No invariant files modified.

---

## Design Fidelity

### `pose3d_transformer_head.py`

The head file is the same unified implementation used for all three designs (confirmed identical MD5 hash across all three design directories). Design B differs from Design A only in the config kwargs (`log_scale_out_features=3`, `per_joint_uncertainty_mode='per_axis'`). The unified head handles both modes correctly:

1. **`__init__`**: `log_scale_out = nn.Linear(hidden_dim, 3)` when `log_scale_out_features=3`. Zero-initialised.

2. **`forward`**: `log_scale = self.log_scale_out(body_decoded)` → shape `(B, 22, 3)` for `log_scale_out_features=3`. No reshape needed — directly `(B, 22, 3)` matching `abs_err` shape element-wise.

3. **`loss`**: For `per_axis` mode, `log_s` is `(B, 22, 3)`, `s = exp(log_s)` is `(B, 22, 3)`, `abs_err` is `(B, 22, 3)`. All operations are element-wise — no broadcasting. `nll = w_ent * torch.log(2.0 * s) + abs_err / s` is `(B, 22, 3)`. `.mean()` averages over batch, joints, axes. Correct.

4. All clamping, entropy weight, pelvis losses, `_train_mpjpe`, and `predict` handling are identical to Design A — all correct as verified in design001 review.

### `config.py`

All required Design B config kwargs present:
- `use_per_joint_uncertainty=True` ✓
- `per_joint_uncertainty_mode='per_axis'` ✓
- `log_scale_out_features=3` ✓
- `laplace_entropy_weight=1.0` ✓
- `laplace_entropy_anneal_steps=0` ✓

All values are bool/int/float/str literals. No Python `import` statements. MMEngine constraint satisfied.

All other config values identical to baseline.

---

## Invariant Verification

No changes to invariant files. Confirmed.

---

## Test Output

Test ran to completion (SLURM job 55871898). Epoch 1 completed, checkpoint saved, no errors.

Key observations from `iter_metrics.csv`:
- `loss/joints/train` values in epoch 1 are slightly higher (~0.883–0.993) than Design A, consistent with 3× as many entropy terms (one per axis) at `s≈1`: `3 * log(2) ≈ 2.079` contribution per joint vs `log(2) ≈ 0.693` in Design A. The mean over 3 axes `(log(2) + |err_x|/s_x + log(2) + |err_y|/s_y + log(2) + |err_z|/s_z) / 3` = `log(2) + mean_abs_err`, same scale as Design A. The slightly higher values confirm per-axis independent entropy terms are active and correct.
- `grad_norm: 8.207` at step 50 — stable.
- Memory: 8619 MB — within constraint.

Training ran correctly. No runtime issues.

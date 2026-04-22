# Code Review — idea026/design001

**Verdict: APPROVED**

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea026/design001` — PASSED.

---

## Files Changed

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py`
- `code/config.py`

Both files are required by design001 (`pose3d_transformer_head.py`, `config.py`). No extra files changed. No invariant files modified.

---

## Design Fidelity

### `pose3d_transformer_head.py`

All design requirements are present and correct:

1. **`__init__` new kwargs**: All six new parameters (`use_per_joint_uncertainty`, `per_joint_uncertainty_mode`, `log_scale_out_features`, `laplace_entropy_weight`, `laplace_entropy_weight_start`, `laplace_entropy_weight_end`, `laplace_entropy_anneal_steps`) are added with correct baseline-preserving defaults (`False`, `'shared_scalar'`, `1`, `1.0`, `1.0`, `1.0`, `0`). All stored as instance attributes.

2. **`log_scale_out` creation**: `nn.Linear(hidden_dim, log_scale_out_features)` created when `use_per_joint_uncertainty=True`. Weight and bias zero-initialised via `nn.init.zeros_`. `_loss_call_count = 0` initialised.

3. **`_init_head_weights`**: No change — correct. `log_scale_out` is zero-initialised separately in `__init__`.

4. **`forward`**: When `use_per_joint_uncertainty=True`, `body_decoded = decoded[:, :22, :]` (B, 22, hidden_dim) and `log_scale = self.log_scale_out(body_decoded)` applied per-token, yielding (B, 22, 1) for `log_scale_out_features=1`. Stored as `pred['log_scale']`. Correct per-token application.

5. **`loss`**: Body-joint loss replaced with Laplace NLL when `use_per_joint_uncertainty=True`:
   - `log_s` clamped to `[-10.0, 5.0]` — AMP safety present.
   - `s = torch.exp(log_s)` then `s.clamp(min=1e-4)` — log(0) prevention present.
   - `abs_err = (pred['joints'][:, _BODY] - gt_joints[:, _BODY]).abs()` — correct body-only slice.
   - Entropy annealing branch: `laplace_entropy_anneal_steps=0` so `w_ent = self.laplace_entropy_weight = 1.0` — correct for Design A.
   - `nll = w_ent * torch.log(2.0 * s) + abs_err / s` — correct Laplace NLL formula.
   - `losses['loss/joints/train'] = nll.mean()` — scalar loss.
   - `(B, 22, 1)` broadcasts over `(B, 22, 3)` correctly.
   - Else branch retains original `loss_joints_module` path.

6. **Pelvis depth and UV losses**: Unchanged.
7. **`_train_mpjpe` and `_train_mpjpe_abs`**: Unchanged, computed with `torch.no_grad()`.
8. **`predict`**: No change — `pred['log_scale']` is present in the output dict but never read by `predict()`.

### `config.py`

All required Design A config kwargs present:
- `use_per_joint_uncertainty=True` ✓
- `per_joint_uncertainty_mode='shared_scalar'` ✓
- `log_scale_out_features=1` ✓
- `laplace_entropy_weight=1.0` ✓
- `laplace_entropy_anneal_steps=0` ✓

All values are bool/int/float/str literals. No Python `import` statements in config. MMEngine constraint satisfied.

All other config values (optimizer, LR, data pipeline, hooks, batch size, seed, etc.) are identical to baseline.

---

## Invariant Verification

No changes to: `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, transforms, data preprocessor, `train.py` wrapper, infra files. Confirmed by implementation_summary and direct file inspection (only two files changed, both are experimentable files).

---

## Test Output

All three designs ran as a single SLURM test job (job 55871896 for design001). Test ran to completion: 1 epoch of training completed, checkpoint saved, "Done training." observed. No errors or exceptions in the log.

Key observations from `iter_metrics.csv`:
- `loss/joints/train` values in epoch 1 are in the range ~0.876–0.957, consistent with Laplace NLL at `s≈1` (log(2) + L1 ≈ 0.693 + |error|). The values are slightly higher than pure L1 due to the entropy term, which is expected behaviour at init.
- Depth and UV losses are in expected ranges.
- `grad_norm: 8.848` at step 50 — training is not exploding.
- Memory: 8619 MB — within 24G constraint.

Training ran correctly. No runtime issues.

# Code Review — idea014 / design002

**Verdict: APPROVED**

## Implementation check
- `python scripts/cli.py review-check-implementation runs/idea014/design002` — PASS.

## Files changed vs design.md
- `code/pose3d_transformer_head.py` — required. Identical to Design 001's head (shared signature design, as stated in design.md).
- `code/config.py` — required.
- `code/pelvis_utils.py` — unchanged vs baseline. OK.
- No invariant files modified.

## Fidelity to design.md
- Design 002 = Design 001 + auxiliary SmoothL1 on the soft-argmax expectation with `depth_aux_reg_weight=0.3`.
- `forward()` uses fixed log-uniform `log_bin_centres` path (not adaptive) — correct for `classification_hybrid`.
- `loss()` CE path identical to Design 001; additionally, since `depth_aux_reg_weight > 0`, the aux branch executes: `L_depth_reg = F.smooth_l1_loss(pred['pelvis_depth'], gt_depth, reduction='mean', beta=0.05)` and emits `losses['loss/depth_reg/train'] = 0.3 * L_depth_reg`. Matches spec exactly.
- Target tensor detached so gradient flows only via `log_probs` (CE) and `pred['pelvis_depth']` (aux SmoothL1) — correct.
- `pelvis_depth` still `(B, 1)` expectation; `_compute_mpjpe_abs` consumes it unchanged.

## Config (`code/config.py`)
- Head kwargs: `depth_head_type='classification_hybrid'`, `num_depth_bins=64`, `depth_range_min=1.0`, `depth_range_max=15.0`, `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight=0.3`. All literal scalars.
- No Python `import` statements; uses `__import__('json')` pattern for splits. OK.
- All other config values match baseline/training-loop invariants.

## Test output
- SLURM job 55739453 completed successfully.
- SLURM log shows the expected four loss keys: `loss/joints/train`, `loss/depth/train`, `loss/depth_reg/train`, `loss/uv/train` — confirms the aux branch is active.
- `loss/depth_reg/train` ≈ 1.04 early (0.3 × SmoothL1 on ~3 m errors), consistent with hybrid formulation.
- `loss/depth/train` ≈ 4.15 (CE, same as Design 001) — confirms fixed-bins branch executed.
- No NaNs, no crashes.

## Notes
- `iter_metrics.csv` schema only has three loss columns (joints/depth/uv); `loss/depth_reg/train` is not in the CSV. This is an invariant of `infra/metrics_csv_hook.py` (not modified by design) — not a fidelity issue for this design.
- Shared head file strategy is explicitly declared in Design 002's design.md and implementation_summary.md. Not a violation.

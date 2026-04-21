# Code Review — idea014 / design001

**Verdict: APPROVED**

## Implementation check
- `python scripts/cli.py review-check-implementation runs/idea014/design001` — PASS.

## Files changed vs design.md
- `code/pose3d_transformer_head.py` — required by design. OK.
- `code/config.py` — required by design. OK.
- `code/pelvis_utils.py` — unchanged vs baseline (diff empty). OK.
- No modification to invariant files (dataset, transforms, backbone, metric, infra, train.py wrapper).

## Fidelity to design.md

1. Six new `__init__` kwargs appended before `init_cfg` with exact names, order, and defaults matching design — verified (`depth_head_type='regression'`, `num_depth_bins=64`, `depth_range_min=1.0`, `depth_range_max=15.0`, `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight=0.0`).
2. Assertion on `depth_head_type` and value-range assertions on classification kwargs present and match design spec exactly.
3. `self.depth_out` allocation is conditional: `Linear(hidden_dim, 1)` in regression mode and `Linear(hidden_dim, num_depth_bins)` in classification modes. Placement between `joints_out` and `uv_out` preserved.
4. `log_bin_centres` registered as non-persistent buffer (`persistent=False`), shape `(K,)`, computed via `torch.linspace(math.log(zmin), math.log(zmax), K)`.
5. `_init_head_weights` applies trunc-normal to all output modules including `depth_out` (width 64 in classification); bias zeroed. Adaptive-only `depth_bins_head` is correctly added to the init list only in adaptive mode.
6. `forward()` classification branch computes `depth_logits = depth_out(pelvis_token)`, `probs = softmax(logits)`, `expected_depth = (probs * bin_centres).sum(-1, keepdim=True)`; preserves `(B, 1)` output contract for `pelvis_depth`. Returned dict exposes `depth_logits` and `depth_bin_centres` as required.
7. `loss()` classification branch implements SORD: `target = softmax(-(log_bin_centres − log(z_gt))^2 / (2 σ^2))` with `sigma_log = depth_soft_label_sigma × bin_width_log`, `bin_width_log = (log_max − log_min) / max(K−1, 1)`. Target is detached. CE computed as `-(target * log_softmax(logits)).sum(-1).mean()`. GT depth clamped into `[zmin, zmax]` before `log()`. Matches design exactly.
8. For Design 001 (`depth_aux_reg_weight=0.0`), the aux regression branch is skipped — no `loss/depth_reg/train` key emitted. Confirmed in SLURM log (only `loss/joints/train`, `loss/depth/train`, `loss/uv/train` present).
9. MPJPE no-grad block unchanged; `_compute_mpjpe_abs` consumes the expectation scalar as `(B, 1)` tensor.
10. `predict()` unchanged; reads `pred['pelvis_depth']`.

## Config (`code/config.py`)
- Head kwargs block appended to `head=dict(...)` with exact Design-001 values: `depth_head_type='classification'`, `num_depth_bins=64`, `depth_range_min=1.0`, `depth_range_max=15.0`, `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight=0.0`. All literals, no Python `import` statements.
- Only differences vs baseline are the splits loader (standard project convention used in idea012/013), `output_dir` for this run, and the six head kwargs. No other training-loop values changed.

## Test output
- SLURM job 55739450 completed successfully.
- `metrics.csv` has one validated epoch; `iter_metrics.csv` has three loss columns (joints, depth, uv) — confirms the aux branch is inactive as required for Design 001.
- `loss/depth/train` early magnitude ≈ 4.17 (close to `log(64) ≈ 4.16`), consistent with near-uniform softmax at init.
- `loss/uv/train` and `loss/joints/train` within normal early-training ranges.
- No NaNs, no crashes.

## Notes
- Shared head file across all three idea014 designs is consistent with the design specs (all three design.md files declare the shared signature and behaviour). No issue.

# Code Review — idea014 / design003

**Verdict: APPROVED**

## Implementation check
- `python scripts/cli.py review-check-implementation runs/idea014/design003` — PASS.

## Files changed vs design.md
- `code/pose3d_transformer_head.py` — required. Shared head file (identical across idea014 designs, declared in design.md).
- `code/config.py` — required.
- `code/pelvis_utils.py` — unchanged. OK.
- No invariant files modified.

## Fidelity to design.md
- Design 003 = Design 002 + per-sample adaptive bin widths (AdaBins-style).
- `__init__` adaptive branch allocates `self.depth_bins_head = Linear(hidden_dim, num_depth_bins)` only when `depth_head_type == 'classification_adaptive'` — correct.
- `_init_head_weights` includes `depth_bins_head` in the trunc-normal init list in adaptive mode — correct.
- `forward()` adaptive branch:
  - `width_logits = depth_bins_head(pelvis_token)`; `widths = softmax(...) * (zmax − zmin)` — widths sum to range.
  - `edges = cumsum(widths)` prepended with zero column and shifted by `depth_range_min` to produce K+1 edges in `[zmin, zmax]`.
  - `bin_centres = 0.5 * (edges[:, :-1] + edges[:, 1:])` (B, K).
  - Soft-argmax expectation over per-sample centres. Matches spec.
- `loss()` adaptive branch:
  - `log_bin_centres_per_sample = bin_centres.clamp(min=zmin*1e-3).log()` — NaN-safe per design.
  - `sigma_log = depth_soft_label_sigma * median(|Δ log_centres|)` per sample — matches design's per-sample sigma from adjacent-centre spacing.
  - SORD target computed against per-sample log bin centres and detached so CE does not propagate into the width head — design explicitly requires detach so widths are trained only by SmoothL1 aux.
- Aux SmoothL1 active (`depth_aux_reg_weight=0.3`); emits `loss/depth_reg/train`. Gradient flows through `pelvis_depth` expectation into `depth_bins_head` (widths) and `depth_out` (probs) — matches design intent.
- `pelvis_depth` `(B, 1)` contract preserved.

## Config (`code/config.py`)
- Head kwargs: `depth_head_type='classification_adaptive'`, `num_depth_bins=64`, `depth_range_min=1.0`, `depth_range_max=15.0`, `depth_soft_label_sigma=1.5`, `depth_aux_reg_weight=0.3`. Literals only.
- No Python imports; standard splits loader. Other config values unchanged from baseline.

## Test output
- SLURM job 55739454 completed successfully.
- SLURM log shows all four expected loss keys: `loss/joints/train`, `loss/depth/train`, `loss/depth_reg/train`, `loss/uv/train`.
- `loss/depth_reg/train` ≈ 1.85 early (larger than Design 002's ~1.04 because adaptive widths at init produce less-calibrated expectations). Validation pelvis ≈ 5659 mm is higher than fixed-bins designs at epoch 1 — expected for AdaBins-style initialisation; not a correctness issue.
- No NaNs, no crashes.

## Notes
- Per-sample `sigma_log` computed from median of adjacent log-centre gaps (rather than strict pairwise widths); this is consistent with the design's "per-sample bin width in log-space" phrasing and avoids per-bin variance blowups — acceptable implementation choice.
- `bin_centres.clamp(min=zmin*1e-3).log()` guards `log(0)` if cumsum edges happen to collapse at the first bin — acceptable numerical guard; does not alter the forward path in practice because `edges` start at `zmin ≥ 1.0`.

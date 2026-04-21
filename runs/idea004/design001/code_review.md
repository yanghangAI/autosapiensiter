**Verdict: APPROVED**

**Reviewer:** Reviewer agent
**Date:** 2026-04-16
**Design:** idea004/design001 — Scalar depth per spatial token (linear projection)

---

## Automated Check

`python scripts/cli.py review-check-implementation runs/idea004/design001` — PASSED.

---

## Files Changed

`implementation_summary.md` lists:
- `code/pose3d_transformer_head.py` — required by design.md. Present and correct.
- `code/config.py` — required by design.md. Present and correct.

No file changed that was not specified by design.md. `pelvis_utils.py` correctly unchanged.

---

## Code vs Design Fidelity

### `pose3d_transformer_head.py`

1. **New imports** (`numpy`, `torch.nn.functional as F`): present at lines 28-31. Correct.

2. **`__init__` signature** — `depth_pos_enc_type: str = 'linear'` added after `loss_weight_uv`, before `init_cfg`. Exact match to design spec (line 163).

3. **`depth_proj = nn.Linear(1, hidden_dim)`**: present at line 192. Correct.

4. **Zero-init of weight and bias**:
   - `nn.init.zeros_(self.depth_proj.weight)` — line 194. Correct.
   - `nn.init.zeros_(self.depth_proj.bias)` — line 195. Correct.
   - Design required explicit zero-init of both. Both present.

5. **`depth_proj` NOT added to `_init_head_weights`**: confirmed — `_init_head_weights` (lines 211-218) only touches `joint_queries`, `joints_out`, `depth_out`, `uv_out`. Constraint satisfied.

6. **`_extract_depth_map` helper** (lines 229-287): matches design spec precisely — reads `depth_npy_path` from metainfo, handles NPZ/NPY, crops to `img_shape`, bilinearly resizes via `F.interpolate(..., mode='bilinear', align_corners=False)`, graceful zero fallback. All edge cases match spec.

7. **`forward` signature**: `depth_map: torch.Tensor | None = None` added. Correct.

8. **`forward` body**:
   - `pos_enc` via `_get_pos_enc` is still added to `spatial` before depth branch (line 317). Correct for Design A (depth is additive on top of 2D sincos).
   - Depth branch (lines 320-327): `depth_map is not None` guard, flatten, clamp [0,10]/10, `depth_proj`, add to `spatial`. Exact match to design spec.
   - `depth_map is None` path: branch is simply skipped — the depth contribution is zero. Correct per design (no-depth fallback uses `if depth_map is not None` guard, not a zero-padding path). Design spec says "If `depth_map is None`, the depth branch is skipped entirely" — this is satisfied.

9. **`loss` method**: calls `_extract_depth_map(batch_data_samples, feat_h, feat_w, feats[-1].device)` before `self.forward(feats, depth_map=depth_map)`. Exact match. Rest of loss (GT extraction, loss computation, MPJPE tracking) unchanged from baseline.

10. **`predict` method**: same pattern as `loss`. Exact match.

11. **Invariants**: body joint loss restricted to `_BODY = list(range(0, 22))` (line 386). Pelvis pathway via `decoded[:, 0, :]` (line 339). Both preserved.

### `config.py`

- `depth_pos_enc_type='linear'` present in head dict (line 146). Correct.
- All other config values (LR 1e-4, weight decay 0.03, batch 4, accum 8, warmup 3 epochs, cosine LR, seed 2026, persistent_workers=False) identical to baseline. Verified.
- No Python `import` statements in config. `__import__('json')` and `__import__('os')` usage is correct inline form. No top-level `import`.
- `depth_npy_path` present in `meta_keys` for both train and val pipeline (line 158, 165). Required for head to extract depth at runtime.

---

## Invariant File Check

No modifications to: `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, data preprocessor, `infra/constants.py`, `infra/metrics_csv_hook.py`, `train.py` wrapper, `tools/train.py`. Only `code/pose3d_transformer_head.py` and `code/config.py` changed. Correct.

---

## Test Output

- SLURM job `55635412` completed successfully with "[test] Finished."
- Training ran 1 epoch on GPU (GTX 1080 Ti), seed 2026 confirmed.
- Validation produced all expected metrics: `mpjpe/rel/val`, `mpjpe/body/val`, `mpjpe/hand/val`, `mpjpe/abs/val`, `mpjpe/pelvis/val`, `composite/val`.
- `metrics.csv` contains correct header and epoch-1 values.
- No runtime errors, exceptions, or NaN/Inf in losses.
- Epoch-1 train loss (1.69) and val composite (479.98 mm) are finite and plausible for epoch 1 of a 20-epoch run.

---

## Summary

All design requirements fully and correctly implemented. No deviations found. Test run clean.

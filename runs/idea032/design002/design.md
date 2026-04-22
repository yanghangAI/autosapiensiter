# Design 002 — Log-Space Auxiliary Depth Reconstruction, λ=0.3

**Design Description:** Design 001 in log-space: the aux head regresses `log1p(depth)` at each feature cell instead of raw metres, with a higher loss weight λ=0.3 to compensate for the smaller log-space magnitudes; compresses dynamic range and aligns the loss with relative depth error.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pelvis_utils.py` — add `downsample_depth_map` helper (same as design001).
2. `pose3d_transformer_head.py` — same global hook + `aux_depth_head` + loss scaffolding as design001; only the log-space branch is activated here.
3. `config.py` — add aux-depth kwargs with `aux_depth_log_space=True`, `aux_depth_loss_weight=0.3`.

No other files are modified. All invariant files are untouched.

---

## Algorithm

Identical to Design 001 except:
- The regression target is `log1p(depth_gt)` (equivalently, `log(depth + 1)`) rather than raw metric depth.
- The prediction is interpreted as predicting `log1p(depth)` directly (no transformation is applied to the aux head output; the target is log-transformed so the loss is computed in log-space).
- `SmoothL1(beta=0.1)` is applied on the log-space residuals.
- Loss weight is `0.3` to account for the ~5–10× reduction in per-pixel residual magnitudes in log-space (raw residual of 1 m at depth 5 m ≈ log-space residual of 0.15).

Masking rules are the same as Design 001 (`0.1 m < depth_gt < 30 m` evaluated on the *raw* metric depth, applied to both `log_pred[valid]` and `log_target[valid]`).

Zero-init behaviour unchanged: aux head outputs 0 at step 0; `log1p(depth_gt)` targets yield finite, well-behaved loss.

---

## 1. `pelvis_utils.py` Changes

Identical to Design 001: append `downsample_depth_map` at the end of the file and add `import torch.nn.functional as F` at the top.

```python
def downsample_depth_map(
    depth_map: torch.Tensor,
    feat_h: int,
    feat_w: int,
) -> torch.Tensor:
    return F.interpolate(
        depth_map, size=(feat_h, feat_w),
        mode='bilinear', align_corners=False,
    ).squeeze(1)
```

---

## 2. `pose3d_transformer_head.py` Changes

Identical scaffolding to Design 001:
- Add module-level RGBD-capture global `forward_pre_hook` (registered once, guarded by `_RGBD_CAPTURE_HOOK_REGISTERED`).
- Update the `pelvis_utils` import to include `downsample_depth_map`.
- Add the same nine `__init__` kwargs (`use_aux_depth`, `aux_depth_loss_weight`, `aux_depth_log_space`, `aux_depth_grad_weight`, `aux_depth_valid_min`, `aux_depth_valid_max`, `aux_depth_denorm_scale`, `feat_h`, `feat_w`) and store them on `self`.
- Instantiate `self.aux_depth_head = nn.Linear(hidden_dim, 1)` with zero-init weight and bias when `use_aux_depth=True`.
- In `forward()`, after `spatial = spatial + pos_enc`, compute `self._aux_depth_pred = self.aux_depth_head(spatial).squeeze(-1).view(B, feat_h, feat_w)` when enabled, else `None`.
- In `loss()`, after the UV-loss line and before the `with torch.no_grad():` block, run the full aux-depth loss snippet from Design 001 (see Design 001 §2d for the exact code). That snippet already branches on `self.aux_depth_log_space`:

```python
if self.aux_depth_log_space:
    target = torch.log1p(depth_gt)
else:
    target = depth_gt
pred = self._aux_depth_pred
```

For Design 002 the flags set by config are `aux_depth_log_space=True`, `aux_depth_grad_weight=0.0`, `aux_depth_loss_weight=0.3`, so the executed branch uses `log1p` targets and skips the gradient-consistency term.

No other changes to the head file. `predict()` is unchanged.

---

## 3. `config.py` Changes

In the `head=dict(...)` block inside `model=dict(...)`, append after `loss_weight_uv=1.0,`:

```python
use_aux_depth=True,
aux_depth_loss_weight=0.3,
aux_depth_log_space=True,
aux_depth_grad_weight=0.0,
aux_depth_valid_min=0.1,
aux_depth_valid_max=30.0,
aux_depth_denorm_scale=20.0,
feat_h=40,
feat_w=24,
```

All values are bool/int/float literals. No Python `import` statements. MMEngine config constraint satisfied. All other config blocks unchanged.

---

## Invariants the Builder Must Preserve

Same list as Design 001:

1. `persistent_workers=False`.
2. Body-only joint loss (indices 0–21).
3. `predict()` path unchanged.
4. Zero-init on `aux_depth_head`.
5. Absolute imports in `pose3d_transformer_head.py`.
6. No `import` in `config.py`.
7. `feat_h=40`, `feat_w=24`.
8. `aux_depth_denorm_scale=20.0` matches `_DEPTH_MAX_METERS` in `bedlam2_transforms.py`.
9. Global hook registration guard.
10. Mask computed on raw metric depth, applied to log-space residuals.
11. Empty-mask safety fallback.
12. AMP safety: `log1p` on positive bilinear-interpolated values is FP16-safe (inputs are in `[0, 20]`, `log1p([0, 20]) ∈ [0, ~3.04]`).

---

## Edge Cases and Risks

- **log1p numerical safety**: `depth_gt >= 0` everywhere (post-clamp, post-interpolation). `log1p(0) = 0`. No NaN risk.
- **Prediction sign**: the aux head is unconstrained real-valued; it can predict negative log-depths, which correspond to `depth < 0` (impossible). `SmoothL1` is robust to arbitrary-signed residuals. No stability issue.
- **Log-space residual scale**: at depth 10 m, a 1 m absolute error is ~0.10 in log-space; at depth 1 m, ~0.50. The log-space loss therefore emphasises relative error, which is the geometrically meaningful quantity for pelvis-depth regression.
- **Gradient scale to `spatial`**: at step 0 the gradient is zero (Linear weight = 0). After a few steps the effective gradient on `spatial` is bounded by `aux_depth_loss_weight * ||aux_depth_head.weight||`. With λ=0.3 and a small weight norm early in training, the perturbation to the backbone is small.
- Same preemption/resume, validation, and zero-fill handling as Design 001.

---

## Expected Behaviour

- **Step 0**: identical to baseline (zero aux contribution).
- **Steady state**: log-space aux loss in the range `[0.05, 0.5]` before λ; with λ=0.3 the aux term contributes `[0.015, 0.15]` to total loss.
- **Convergence property**: because log-space depth loss is roughly `|Δd/d|` for small residuals, the spatial tokens are incentivised to encode *relative* depth accuracy, which is the dominant error mode at large depths (where a 1 m error at 15 m matters less than a 1 m error at 2 m).
- **New CSV scalar key**: `loss/aux_depth/train`.

---

## Expected Metrics (Stage-1, Epoch 20)

- `mpjpe_pelvis_val < 600 mm` (improvement over best prior 608 mm). Driver: improved feature-level depth fidelity, especially at mid-range depths where log-space loss provides stronger gradient than raw-L1.
- `mpjpe_abs_val < 780 mm` (improvement over best prior 785 mm).
- `composite_val < 323 mm` (target beats best prior 323.75 mm from idea023).
- `mpjpe_body_val` mild improvement or neutral (backbone regularisation).

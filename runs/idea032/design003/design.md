# Design 003 — Log-Space Depth Reconstruction + Gradient Consistency, λ=0.3

**Design Description:** Design 002 (log-space aux depth reconstruction at λ=0.3) plus an edge-preserving first-order spatial gradient consistency term at sub-weight 0.5; the total aux loss is `recon_loss + 0.5 * grad_loss` times λ=0.3, encouraging both absolute per-cell depth fidelity and matching depth discontinuities at the body outline.

**Starting Point:** `baseline/`

---

## Files to Modify

1. `pelvis_utils.py` — add `downsample_depth_map` helper (same as design001/002).
2. `pose3d_transformer_head.py` — same scaffolding as design001/002; the gradient-consistency branch is activated here.
3. `config.py` — add aux-depth kwargs with `aux_depth_log_space=True`, `aux_depth_loss_weight=0.3`, `aux_depth_grad_weight=0.5`.

No other files are modified. All invariant files are untouched.

---

## Algorithm

Identical to Design 002 (log-space target, SmoothL1 recon, masked foreground, λ=0.3) plus an additional gradient-consistency term computed **on the log-space prediction and log-space target**:

```python
# pred, target are both shape (B, feat_h=40, feat_w=24), in log1p(depth) space
dx_pred = pred[:, :, 1:] - pred[:, :, :-1]          # (B, 40, 23)
dy_pred = pred[:, 1:, :] - pred[:, :-1, :]          # (B, 39, 24)
dx_tgt  = target[:, :, 1:] - target[:, :, :-1]
dy_tgt  = target[:, 1:, :] - target[:, :-1, :]
grad_loss = (dx_pred - dx_tgt).abs().mean() + (dy_pred - dy_tgt).abs().mean()

total_aux = recon_loss + self.aux_depth_grad_weight * grad_loss   # grad_weight = 0.5
losses['loss/aux_depth/train'] = self.aux_depth_loss_weight * total_aux
```

The gradient term penalises mismatch in first-order spatial differences — this is the standard edge-preserving depth loss. It is **not** masked (unlike the recon term) because differences across invalid-pixel boundaries still carry useful information about the foreground outline and masking would introduce variable-size tensor indexing.

The three hyperparameters are: `aux_depth_loss_weight = 0.3` (outer λ), `aux_depth_grad_weight = 0.5` (sub-weight on grad term relative to recon term), `aux_depth_log_space = True`.

Zero-init behaviour unchanged: at step 0, `pred = 0` → `dx_pred = dy_pred = 0` → `grad_loss = |dx_tgt|.mean() + |dy_tgt|.mean() = const(target)`. This is a non-zero constant independent of model parameters; its gradient w.r.t. model parameters is zero at step 0 (since the only model-dependent quantity is `pred` and `pred=0`). Main losses at step 0 are therefore identical to baseline.

---

## 1. `pelvis_utils.py` Changes

Identical to Design 001/002: append `downsample_depth_map` at the end of the file and add `import torch.nn.functional as F` at the top.

---

## 2. `pose3d_transformer_head.py` Changes

Identical scaffolding to Design 001 (see Design 001 §2a–§2d for the exact code):
- Module-level RGBD-capture global `forward_pre_hook` + registration guard.
- `pelvis_utils` import updated to include `downsample_depth_map`.
- Nine new `__init__` kwargs; stored on `self`.
- `self.aux_depth_head = nn.Linear(hidden_dim, 1)` zero-init when enabled.
- `forward()` computes `self._aux_depth_pred`.
- `loss()` runs the full aux-depth loss snippet, which already branches on `self.aux_depth_grad_weight > 0` to include the gradient term.

For Design 003 the flags set by config are `aux_depth_log_space=True`, `aux_depth_grad_weight=0.5`, `aux_depth_loss_weight=0.3`, so the executed code path uses log-space targets AND adds the grad consistency term.

No other changes to the head file. `predict()` is unchanged.

### Gradient-consistency code block (inside the Design 001 §2d snippet)

```python
if self.aux_depth_grad_weight > 0:
    dx_pred = pred[:, :, 1:] - pred[:, :, :-1]
    dy_pred = pred[:, 1:, :] - pred[:, :-1, :]
    dx_tgt = target[:, :, 1:] - target[:, :, :-1]
    dy_tgt = target[:, 1:, :] - target[:, :-1, :]
    grad_loss = (dx_pred - dx_tgt).abs().mean() + (
        dy_pred - dy_tgt).abs().mean()
    recon_loss = recon_loss + self.aux_depth_grad_weight * grad_loss
```

This block is already part of the Design 001 scaffolding; enabling it is purely a config choice.

---

## 3. `config.py` Changes

In the `head=dict(...)` block inside `model=dict(...)`, append after `loss_weight_uv=1.0,`:

```python
use_aux_depth=True,
aux_depth_loss_weight=0.3,
aux_depth_log_space=True,
aux_depth_grad_weight=0.5,
aux_depth_valid_min=0.1,
aux_depth_valid_max=30.0,
aux_depth_denorm_scale=20.0,
feat_h=40,
feat_w=24,
```

All values are bool/int/float literals. No Python `import`. MMEngine config constraint satisfied. All other config blocks unchanged.

---

## Invariants the Builder Must Preserve

Same list as Design 001/002 plus:

13. **Gradient term is computed on *log-space* tensors** (same space as the recon term), not on raw-metric depth. Mixing spaces would create an unprincipled scale mismatch.
14. **Gradient term is unmasked** (computed over all 40×24 cells) — the per-cell differences are cheap and the mean absorbs any boundary-induced noise; this matches the standard edge-preserving depth loss convention.
15. **`aux_depth_grad_weight = 0.5`** is the inner sub-weight; the outer λ is still `aux_depth_loss_weight = 0.3`. The total aux loss magnitude is `0.3 * (recon + 0.5 * grad)`.

---

## Edge Cases and Risks

- **Grad-term dominance at step 0**: at step 0, `pred = 0` ⇒ `dx_pred = 0` ⇒ `grad_loss = |dx_tgt|.mean() + |dy_tgt|.mean()`. This is a data-dependent constant with zero gradient w.r.t. parameters. No effect on training at step 0.
- **FP16 / AMP safety**: first-order differences of a bilinearly interpolated log-depth tensor are bounded and well-represented in FP16.
- **Sensitivity to grad-weight**: 0.5 (50% of the recon term) is a standard starting value in monocular-depth literature. If the grad term dominates (empirically: grad_loss ~ O(0.1), recon_loss ~ O(0.3) in log-space), the effective split is roughly 60% recon / 40% grad, which is balanced.
- Same preemption/resume, validation, and zero-fill handling as Design 001/002.

---

## Expected Behaviour

- **Step 0**: identical to baseline.
- **Steady state**: `recon_loss ≈ 0.1–0.5` (log-space), `grad_loss ≈ 0.05–0.3`. `total_aux ≈ 0.12–0.65`. With outer λ=0.3, the aux term contributes 0.04–0.20 to total loss.
- **New CSV scalar key**: `loss/aux_depth/train` (single combined scalar).

---

## Expected Metrics (Stage-1, Epoch 20)

- `mpjpe_body_val` improvement expected largest here — depth edges at body outline are geometrically informative for body-joint localisation.
- `mpjpe_pelvis_val` and `mpjpe_abs_val`: similar to Design 002 (pelvis benefit primarily from recon fidelity; grad term is a secondary regulariser).
- `composite_val < 322 mm` target (best case across the three designs).

# Design 002 — Absolute Body Joint Loss with Depth-Axis Upweighting (λ=0.5, axis_weights=[2.0, 1.0, 1.0])

**Design Description:** Add smooth-L1 absolute body joint loss (λ=0.5) with X-axis (forward/depth) residuals weighted 2× vs. Y/Z axes, amplifying gradient to the depth head for the dominant depth-axis absolute error.

**Starting Point:** `baseline/`

---

## Algorithm

The core algorithm is the same as Design 001 (absolute body joint smooth-L1 loss via `recover_abs_joints_batched`), with one modification: before averaging, the element-wise smooth-L1 residuals are multiplied by per-axis weights `[2.0, 1.0, 1.0]` corresponding to `[X, Y, Z]`. Since BEDLAM2 uses X=forward (depth direction), this doubles the gradient signal for depth errors at both the relative-joint head and the pelvis depth head. The algorithm otherwise follows the same coupling mechanism: both prediction pathways receive gradient through the compound absolute-space residual.

## Summary of Changes

Three files change: `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py`.

The `pelvis_utils.py` and `pose3d_transformer_head.py` changes are **identical to Design 001** with one difference in `config.py`: the `abs_joint_axis_weights=[2.0, 1.0, 1.0]` kwarg is added.

---

## 1. `pelvis_utils.py`

**Identical to Design 001.** Add `recover_abs_joints_batched` at the end of the file:

```python
def recover_abs_joints_batched(
    pred_joints_rel: torch.Tensor,
    gt_joints_rel: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    pred_uv: torch.Tensor,
    gt_uv: torch.Tensor,
    batch_data_samples,
    num_body_joints: int = 22,
):
    """Compute predicted and GT absolute joint positions (with gradients).

    Returns:
        pred_abs: (B, num_body_joints, 3) predicted absolute body joints (metres).
        gt_abs:   (B, num_body_joints, 3) GT absolute body joints (metres).
    """
    B = pred_joints_rel.size(0)
    pred_abs_list = []
    gt_abs_list = []
    for i in range(B):
        ds = batch_data_samples[i]
        K = np.asarray(ds.metainfo.get('K'), dtype=np.float32)
        img_shape = ds.metainfo.get('img_shape', (640, 384))
        crop_h, crop_w = int(img_shape[0]), int(img_shape[1])

        pred_pelvis = recover_pelvis_3d(
            pred_depth[i:i+1], pred_uv[i:i+1], K, crop_h, crop_w)  # (1, 3)
        gt_pelvis = recover_pelvis_3d(
            gt_depth[i:i+1], gt_uv[i:i+1], K, crop_h, crop_w)      # (1, 3)

        pred_abs_list.append(pred_joints_rel[i, :num_body_joints] + pred_pelvis)  # (J, 3)
        gt_abs_list.append(gt_joints_rel[i, :num_body_joints] + gt_pelvis)        # (J, 3)

    return torch.stack(pred_abs_list), torch.stack(gt_abs_list)  # (B, J, 3) each
```

**Constraints (same as Design 001):**
- No `.detach()`, no `.norm()`, no `* 1000.0`.
- Existing function `compute_mpjpe_abs` is unchanged.

---

## 2. `pose3d_transformer_head.py`

**Identical to Design 001.** All changes described below must be applied exactly as in Design 001 — the `abs_axis_weights` buffer path is exercised here (where Design 001 does not use it).

### 2a. Import addition (after existing `from pelvis_utils import compute_mpjpe_abs as _compute_mpjpe_abs`):

```python
from pelvis_utils import recover_abs_joints_batched as _recover_abs_joints_batched
```

### 2b. `__init__` signature additions (after `loss_weight_uv: float = 1.0,`, before `init_cfg`):

```python
abs_joint_loss_weight: float = 0.0,
abs_joint_indices: int = 22,
abs_joint_axis_weights=None,
abs_joint_pelvis_grad_scale: float = 1.0,
```

### 2c. `__init__` body additions (after `self.loss_weight_uv = loss_weight_uv`):

```python
self.abs_joint_loss_weight = abs_joint_loss_weight
self.abs_joint_indices = abs_joint_indices
self.abs_joint_pelvis_grad_scale = abs_joint_pelvis_grad_scale

if abs_joint_axis_weights is not None:
    w = torch.tensor(abs_joint_axis_weights, dtype=torch.float32)  # (3,)
    self.register_buffer('abs_axis_weights', w)
else:
    self.abs_axis_weights = None
```

**For this design**, `abs_joint_axis_weights=[2.0, 1.0, 1.0]` is passed from config, so `self.abs_axis_weights` will be a `(3,)` buffer with values `[2.0, 1.0, 1.0]`.

### 2d. `loss()` method additions (after `losses['loss/uv/train'] = ...`, before `with torch.no_grad():`):

```python
        # ── Absolute Body Joint Consistency Loss ────────────────────────────────
        if self.abs_joint_loss_weight > 0.0:
            if self.abs_joint_pelvis_grad_scale < 1.0:
                alpha = self.abs_joint_pelvis_grad_scale
                pred_abs_full, gt_abs = _recover_abs_joints_batched(
                    pred['joints'], gt_joints,
                    pred['pelvis_depth'], gt_depth,
                    pred['pelvis_uv'], gt_uv,
                    batch_data_samples, self.abs_joint_indices)
                pred_abs_det, _ = _recover_abs_joints_batched(
                    pred['joints'], gt_joints,
                    pred['pelvis_depth'].detach(), gt_depth,
                    pred['pelvis_uv'].detach(), gt_uv,
                    batch_data_samples, self.abs_joint_indices)
                pred_abs = alpha * pred_abs_full + (1.0 - alpha) * pred_abs_det
            else:
                pred_abs, gt_abs = _recover_abs_joints_batched(
                    pred['joints'], gt_joints,
                    pred['pelvis_depth'], gt_depth,
                    pred['pelvis_uv'], gt_uv,
                    batch_data_samples, self.abs_joint_indices)

            beta_abs = 0.05
            abs_diff = (pred_abs - gt_abs).abs()
            abs_loss_raw = torch.where(
                abs_diff < beta_abs,
                0.5 * abs_diff ** 2 / beta_abs,
                abs_diff - 0.5 * beta_abs,
            )  # (B, abs_joint_indices, 3)

            if self.abs_axis_weights is not None:
                abs_loss_raw = abs_loss_raw * self.abs_axis_weights.view(1, 1, 3)

            losses['loss/abs_joints/train'] = self.abs_joint_loss_weight * abs_loss_raw.mean()
```

**For this design**, `self.abs_joint_pelvis_grad_scale = 1.0` (not < 1.0), so the `else` branch executes (no detach). `self.abs_axis_weights` is `[2.0, 1.0, 1.0]` so the per-axis weighting line executes.

**Effect of per-axis weighting:**
- `abs_loss_raw` shape: `(B=4, 22, 3)`
- After `* self.abs_axis_weights.view(1, 1, 3)`: X-column scaled ×2.0, Y and Z columns unchanged.
- `.mean()` averages over all B×22×3 = 264 elements. The effective weight on the X-axis residual is 2/4 (2× weight / 4 total elements per joint when averaging), making depth errors twice as impactful in the loss as Y/Z errors.
- The buffer `abs_axis_weights` must be on the correct device. `register_buffer` handles device placement automatically during `.to(device)` or `.cuda()` calls.

---

## 3. `config.py`

In `head=dict(...)`, add three kwargs after `loss_weight_uv=1.0,`:

```python
        abs_joint_loss_weight=0.5,
        abs_joint_indices=22,
        abs_joint_axis_weights=[2.0, 1.0, 1.0],
```

All values are float/int/list-of-float literals. No Python imports. Full head dict:

```python
    head=dict(
        type='Pose3dTransformerHead',
        in_channels=embed_dim,
        hidden_dim=256,
        num_joints=num_joints,
        num_heads=8,
        dropout=0.1,
        loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                         loss_weight=1.0),
        loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                        loss_weight=1.0),
        loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                     loss_weight=1.0),
        loss_weight_depth=1.0,
        loss_weight_uv=1.0,
        abs_joint_loss_weight=0.5,
        abs_joint_indices=22,
        abs_joint_axis_weights=[2.0, 1.0, 1.0],
    ),
```

---

## Invariants to Preserve

Same as Design 001:
- `persistent_workers=False`, seed `2026`, batch 4/accum 8.
- Joint loss restricted to body joints `0-21`.
- `_compute_mpjpe_abs` inside `with torch.no_grad():` unchanged.
- AMP, `resume=True`, `max_keep_ckpts=1`.

---

## Rationale for X-Axis Upweighting

BEDLAM2 coordinate system: X = forward (depth direction). The absolute position of joint `i` along X is:

```
pred_abs[i, X] = pred_joints_rel[i, X] + pred_pelvis_depth
```

The depth/distance uncertainty is the dominant source of absolute error in BEDLAM2 (subjects at varying 1–10m distances). Y and Z coordinates are constrained by image appearance and intrinsics; X is not. By weighting the X-axis residual 2× in the smooth-L1 loss, the gradient magnitude to `pred_pelvis_depth` (via `d(pred_abs[i,X])/d(pred_depth) = 1.0`) is doubled relative to Y/Z, providing a stronger depth learning signal.

---

## Expected Behaviour

- `loss/abs_joints/train` appears in training log.
- The X-axis component of `loss/abs_joints/train` is 2× larger than in Design 001 for the same absolute errors.
- `mpjpe_pelvis_val` target: < 580mm at stage-1 (same target as Design 001, but depth upweighting expected to produce stronger gains).
- `mpjpe_abs_val` target: < 750mm at stage-1.
- `composite_val` target: < 330 at stage-1 (slightly more ambitious than Design 001's < 335).

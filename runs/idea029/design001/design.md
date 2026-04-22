# Design 001 — Absolute Body Joint Loss, Uniform Weight λ=0.5

**Design Description:** Add smooth-L1 loss on predicted absolute 3D body joint positions (joints_rel + unproject(pelvis)) with uniform weight 0.5, coupling gradient to both relative-joint head and pelvis depth/UV heads.

**Starting Point:** `baseline/`

---

## Algorithm

The core algorithm adds a smooth-L1 loss on the **absolute** 3D body joint positions: for each training sample, the predicted pelvis 3D position (recovered via `recover_pelvis_3d`) is added to the predicted root-relative joints to form predicted absolute joints; the same is done for GT. Smooth-L1 with `beta=0.05` is applied over all 22 body joints × 3 coordinates and averaged. This provides end-to-end gradient coupling between the relative-joint prediction head and the pelvis depth/UV heads — both pathways receive gradient from the same compound absolute-space residual.

## Summary of Changes

Three files change: `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py`.

---

## 1. `pelvis_utils.py`

Add a new helper function `recover_abs_joints_batched` **after** the existing `compute_mpjpe_abs` function (around line 98). This function mirrors `compute_mpjpe_abs` but returns raw gradient-carrying tensors instead of a scalar MPJPE value.

### Exact code to append at the end of `pelvis_utils.py`:

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

**Key constraints:**
- Do NOT apply `.norm()`, `* 1000.0`, or `.detach()` anywhere in this function — gradients must flow through `pred_abs_list`.
- `gt_pelvis` is computed with `gt_depth` and `gt_uv` — this is a constant w.r.t. model parameters (no gradient needed for GT).
- The function signature type hints use bare names (not `torch.Tensor` annotation style with imports from `typing`) so no additional imports are needed beyond what is already in `pelvis_utils.py`.

---

## 2. `pose3d_transformer_head.py`

### 2a. Import addition (line 36, after existing `from pelvis_utils import compute_mpjpe_abs as _compute_mpjpe_abs`):

```python
from pelvis_utils import recover_abs_joints_batched as _recover_abs_joints_batched
```

### 2b. `__init__` signature: add four new keyword arguments with defaults that reproduce baseline behaviour:

After `loss_weight_uv: float = 1.0,` and before `init_cfg: OptConfigType = None,`, add:

```python
abs_joint_loss_weight: float = 0.0,
abs_joint_indices: int = 22,
abs_joint_axis_weights=None,
abs_joint_pelvis_grad_scale: float = 1.0,
```

Default `abs_joint_loss_weight=0.0` means the absolute loss is off unless explicitly set — preserves backward compatibility with baseline.

### 2c. `__init__` body: store new attributes after `self.loss_weight_uv = loss_weight_uv`:

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

### 2d. `loss()` method: add absolute joint loss block after the three existing loss lines and before the `with torch.no_grad():` block.

Insert the following block (after `losses['loss/uv/train'] = ...` line, before `with torch.no_grad():`):

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

**Important notes on gt_depth / gt_uv tensors:**
- The existing `loss()` already extracts `gt_depth` (shape `(B, 1)`) and `gt_uv` (shape `(B, 2)`) from `batch_data_samples` earlier in the method. Reuse those exact tensors — do NOT re-extract them from `batch_data_samples` inside the absolute loss block.
- `pred['joints']` is `(B, 70, 3)`. The helper slices `[:num_body_joints]` = `[:22]` internally.
- `gt_joints` is already assembled as `(B, 70, 3)` above the existing loss lines. Reuse it.

**Constraints:**
- The `loss/abs_joints/train` key must be a differentiable tensor (no `.detach()` on `pred_abs`). MMEngine will call `.backward()` on the summed losses dict.
- Do not change anything in the `predict()` method.
- Do not change `_BODY`, the joint/depth/UV loss lines, or the `with torch.no_grad():` block.

---

## 3. `config.py`

In the `model` dict, inside `head=dict(...)`, add the following two kwargs after `loss_weight_uv=1.0,`:

```python
        abs_joint_loss_weight=0.5,
        abs_joint_indices=22,
```

All other head kwargs remain unchanged. Full head dict after edit:

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
    ),
```

No Python import statements are added. All values are float/int literals. MMEngine config constraint satisfied.

---

## Invariants to Preserve

- `persistent_workers=False` — unchanged.
- Seed `2026` — unchanged.
- Batch 4, accum 8 — unchanged.
- `loss_joints` restricted to body joints `0-21` (`_BODY = list(range(0, 22))`) — unchanged.
- `_compute_mpjpe_abs` usage inside `with torch.no_grad():` — unchanged.
- AMP via `FixedAmpOptimWrapper` — unchanged.
- `resume=True`, `max_keep_ckpts=1` — unchanged.

---

## Expected Behaviour

- A new loss term `loss/abs_joints/train` appears in the training log.
- At early training (epoch 1), `loss/abs_joints/train` will be in the linear smooth-L1 regime (absolute errors >> 0.05m) and have magnitude roughly equal to `0.5 * mean_abs_error_metres`. Typical early absolute errors are 0.5–3m, so expect `loss/abs_joints/train ≈ 0.25–1.5`.
- Gradient flows from `loss/abs_joints/train` to both `pred['joints'][:, :22]` (relative joint head) and `pred['pelvis_depth']`, `pred['pelvis_uv']` (pelvis heads).
- `mpjpe_pelvis_val` should improve vs. baseline (target: < 580mm at stage-1, baseline 652mm).
- `composite_val` target: < 335 at stage-1.
- No change to `mpjpe_rel_val` metric (root-relative body MPJPE is not directly affected by the absolute loss, but indirect gradient through shared backbone features may help).

---

## Design Variants Not Included Here

This design uses `abs_joint_axis_weights=None` (uniform, no per-axis weighting) and `abs_joint_pelvis_grad_scale=1.0` (full gradient to pelvis branch). Designs 002 and 003 add per-axis weighting and stop-gradient respectively.

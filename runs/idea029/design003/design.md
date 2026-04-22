# Design 003 — Absolute Body Joint Loss with Pelvis Gradient Scaling (λ=0.5, pelvis_grad_scale=0.5)

**Design Description:** Add smooth-L1 absolute body joint loss (λ=0.5) where gradient to the pelvis depth/UV heads is scaled to 0.5 via selective stop-gradient, preventing the absolute loss from over-competing with the direct L_depth and L_uv supervision signals.

**Starting Point:** `baseline/`

---

## Algorithm

The core algorithm is the same as Design 001 (absolute body joint smooth-L1 loss), with a selective stop-gradient mechanism: the absolute loss is computed twice — once with the full pelvis 3D tensor (gradient flows to both relative joints and pelvis heads) and once with the pelvis 3D tensor detached (gradient flows to relative joints only). The final `pred_abs` used for loss computation is `alpha * pred_abs_full + (1 - alpha) * pred_abs_det` with `alpha=0.5`. This implements per-branch gradient scaling: the relative joint head receives full gradient (1.0) from the absolute loss, while the pelvis depth/UV heads receive half gradient (0.5), preventing the absolute loss from over-competing with the direct `L_depth` and `L_uv` supervision.

## Summary of Changes

Three files change: `pelvis_utils.py`, `pose3d_transformer_head.py`, `config.py`.

The `pelvis_utils.py` changes are **identical to Designs 001/002**. The `pose3d_transformer_head.py` changes are identical to Designs 001/002 (same code block handles all three designs). The `config.py` difference: `abs_joint_pelvis_grad_scale=0.5` is added instead of `abs_joint_axis_weights`.

---

## 1. `pelvis_utils.py`

**Identical to Designs 001/002.** Add `recover_abs_joints_batched` at the end of the file:

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

**Constraints (same as Designs 001/002):**
- No `.detach()`, no `.norm()`, no `* 1000.0`.
- Existing `compute_mpjpe_abs` unchanged.

---

## 2. `pose3d_transformer_head.py`

The code changes are **identical to Designs 001/002** — the same `loss()` block handles the pelvis-grad-scale logic via the `if self.abs_joint_pelvis_grad_scale < 1.0:` branch.

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

**For this design**, `abs_joint_axis_weights=None`, so `self.abs_axis_weights = None` (no per-axis weighting). `abs_joint_pelvis_grad_scale=0.5` so `self.abs_joint_pelvis_grad_scale = 0.5`.

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

**For this design**, `self.abs_joint_pelvis_grad_scale = 0.5 < 1.0`, so the `if` branch executes.

**Gradient analysis for this design:**

With `alpha = 0.5`:

```
pred_abs = 0.5 * pred_abs_full + 0.5 * pred_abs_det
         = 0.5 * (pred_joints_rel + pred_pelvis) + 0.5 * (pred_joints_rel + pred_pelvis.detach())
         = pred_joints_rel + 0.5 * pred_pelvis + 0.5 * pred_pelvis.detach()
```

Gradient w.r.t. `pred_joints_rel`: coefficient is `0.5 + 0.5 = 1.0` (full gradient, same as Design 001).
Gradient w.r.t. `pred_pelvis_depth` / `pred_pelvis_uv` (via `recover_pelvis_3d`): coefficient is `0.5 * alpha = 0.5 * 0.5 = 0.25`... 

Wait — more precisely: the `pred_abs_full` term carries gradient to the pelvis with coefficient 0.5 (from the `alpha *` scaling). The `pred_abs_det` term carries zero gradient to the pelvis (detached). So:

- Gradient to `pred_pelvis` from absolute loss: `0.5 × (full gradient)` — half the gradient compared to Design 001.
- Gradient to `pred_joints_rel` from absolute loss: `(0.5 + 0.5) = 1.0 × (full gradient)` — same as Design 001.

This implements the intent: the relative joint branch benefits fully from the absolute loss signal, while the pelvis branch receives only half the absolute loss gradient (the other half coming from `L_depth` and `L_uv` already).

**Important:** Both `_recover_abs_joints_batched` calls must use the already-assembled `gt_joints`, `gt_depth`, `gt_uv` tensors from earlier in `loss()`. Do NOT re-extract from `batch_data_samples`. The second call returns `_` for `gt_abs` (discarded) since `gt_abs` is identical between the two calls.

---

## 3. `config.py`

In `head=dict(...)`, add two kwargs after `loss_weight_uv=1.0,`:

```python
        abs_joint_loss_weight=0.5,
        abs_joint_indices=22,
        abs_joint_pelvis_grad_scale=0.5,
```

All values are float/int literals. No Python imports. Full head dict:

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
        abs_joint_pelvis_grad_scale=0.5,
    ),
```

---

## Invariants to Preserve

Same as Designs 001/002:
- `persistent_workers=False`, seed `2026`, batch 4/accum 8.
- Joint loss restricted to body joints `0-21`.
- `_compute_mpjpe_abs` inside `with torch.no_grad():` unchanged.
- AMP, `resume=True`, `max_keep_ckpts=1`.

---

## Rationale for Pelvis Gradient Scaling

The depth/UV heads are already directly supervised by `L_depth` and `L_uv` (clean, per-target scalar losses). If the absolute joint loss also propagates full gradient to the depth/UV parameters, there are now two gradient sources of comparable scale pointing to the same parameters:

1. `L_depth`: direct, clean, single-target signal.
2. `L_abs → pelvis branch`: indirect, 22-joint-summed signal via `recover_pelvis_3d`.

If both are at full strength, the pelvis head may receive conflicting gradient directions (when relative joint errors and pelvis depth errors partially compensate each other). Scaling the absolute loss gradient to 0.5 for the pelvis branch gives the direct depth/UV losses priority while still providing the coupling gradient for absolute consistency.

The relative joint branch, in contrast, has no direct absolute-space supervision — it only receives gradient from `L_joints` in root-relative space. For this branch, full absolute-loss gradient (scale 1.0) is beneficial because there is no risk of competing with an existing absolute signal.

---

## Expected Behaviour

- `loss/abs_joints/train` appears in training log.
- Training stability should be highest among the three designs because the pelvis head is not over-driven by the absolute loss.
- `mpjpe_pelvis_val` target: < 580mm at stage-1 (baseline 652mm).
- `mpjpe_body_val` target: < 185mm at stage-1 (benefit from full absolute gradient on joint branch).
- `composite_val` target: < 330 at stage-1.
- Stage-2: `composite_val` target < 222 (vs. best prior 224.52).

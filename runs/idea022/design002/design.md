**Design Description:** 2-layer cascaded decoder with dynamic Gaussian reprojection bias (fixed σ=4, γ=2) and auxiliary body-joint loss (weight=0.4) on layer-1 output.

**Starting Point:** `baseline/`

---

## Overview

**Algorithm:** Two-stage cascaded decoding with geometry-guided cross-attention and auxiliary intermediate supervision. Layer 1 produces an intermediate 3D pose estimate; those predicted joints are reprojected to 2D feature-grid coordinates using camera intrinsics K; a per-sample Gaussian additive bias is constructed over the 40×24 feature grid centred on each projected 2D joint location and injected into layer 2's cross-attention logits. An auxiliary body-joint loss (weight=0.4) on layer-1's output directly supervises layer-1 predictions, bootstrapping reprojection bias quality from epoch 1.

Same as Design A (idea022/design001) — two transformer decoder layers with dynamic Gaussian cross-attention bias fed from layer-1's intermediate 3D predictions — but adds an **auxiliary joint regression loss** on the layer-1 output with weight 0.4. The auxiliary loss directly supervises layer-1 joint predictions, improving the quality of the reprojection-derived attention bias from the earliest training epochs. No intermediate depth or UV loss on layer-1 (to avoid pelvis regression degradation seen in prior multi-layer ideas).

The auxiliary loss also provides an independent gradient path into layer-1 weights that does not require backpropagation through layer-2.

---

## Files to Change

### 1. `pelvis_utils.py`

Add the same `project_joints_to_feat_grid` helper as in design001 (identical code). If design001 was built first, this function is already present — the Builder must check and skip if already added.

```python
def project_joints_to_feat_grid(
    joints_abs: torch.Tensor,
    K,
    crop_h: int,
    crop_w: int,
    feat_h: int = 40,
    feat_w: int = 24,
) -> torch.Tensor:
    """Project absolute 3D joints (camera frame) to feature-grid coordinates.

    BEDLAM2 convention: X=forward (depth), Y=left, Z=up.
    Projection: u_px = fx*(-Y/X) + cx,  v_px = fy*(-Z/X) + cy

    Args:
        joints_abs: (B, J, 3) absolute camera-frame joints in metres.
        K: (3, 3) crop intrinsic matrix (numpy array).
        crop_h: Crop height in pixels.
        crop_w: Crop width in pixels.
        feat_h: Feature grid height (default 40).
        feat_w: Feature grid width (default 24).

    Returns:
        (B, J, 2) float tensor: (h_frac, w_frac) in feature grid units,
        clamped to [0, feat_h) x [0, feat_w).
    """
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    X = joints_abs[..., 0].clamp(min=0.01)  # (B, J)
    Y = joints_abs[..., 1]                   # (B, J)
    Z = joints_abs[..., 2]                   # (B, J)

    u_px = -Y / X * fx + cx                  # (B, J)
    v_px = -Z / X * fy + cy                  # (B, J)

    h_frac = v_px / crop_h * feat_h
    w_frac = u_px / crop_w * feat_w

    h_frac = h_frac.clamp(0.0, float(feat_h) - 1e-4)
    w_frac = w_frac.clamp(0.0, float(feat_w) - 1e-4)

    return torch.stack([h_frac, w_frac], dim=-1)  # (B, J, 2)
```

### 2. `pose3d_transformer_head.py`

All structural changes from design001 apply here identically:
- Import `F`, `np`, `recover_pelvis_3d`, `project_joints_to_feat_grid` (2b in design001).
- Add `_build_gaussian_bias` module-level function (2b in design001, identical).
- Modify `_DecoderLayer.forward` to accept optional `cross_attn_bias` (2c in design001, identical).
- Extend `__init__` signature with the new parameters (2d in design001) — same parameter set.
- Replace single `decoder_layer` with `decoder_layers` ModuleList (2d in design001, identical).
- Modify `forward()` with multi-layer loop and bias injection (2e in design001, identical).

**Key difference from design001: `loss()` changes**

Design B differs from Design A in two ways in `loss()`:

**Difference 1: Intermediate layer-1 forward with gradient enabled.**

In design001, the intermediate layer-0 forward uses `torch.no_grad()`. In Design B, the intermediate layer-0 forward uses normal autograd (no `torch.no_grad()`), because the auxiliary loss must backpropagate through layer-1.

Replace the `with torch.no_grad():` block:
```python
with torch.no_grad():
    decoded_l1 = self.decoder_layers[0](queries_tmp, spatial_tmp)
```
With (no `torch.no_grad()`):
```python
decoded_l1 = self.decoder_layers[0](queries_tmp, spatial_tmp)
```

All other bias construction code (recovering pelvis, projecting to feature grid, building Gaussian bias) is identical to design001 — fix sigma/gamma as full tensors with `reproj_bias_sigma` and `reproj_bias_gamma`.

**Difference 2: Add auxiliary joint loss after `pred = self.forward(feats)`.**

After `pred = self.forward(feats)` and after collecting `gt_joints` (which is already done in baseline `loss()`), add:

```python
# Auxiliary intermediate loss on layer-1 output (body joints only)
if self.aux_loss_weight > 0.0:
    _BODY = list(range(0, 22))
    losses['loss/joints_aux/train'] = (
        self.aux_loss_weight * self.loss_joints_module(
            layer1_joints[:, _BODY], gt_joints[:, _BODY]))
```

Note: `layer1_joints` is computed inside the `if self.use_reproj_bias` block before calling `self.forward(feats)`. The Builder must ensure `layer1_joints` is in scope when the auxiliary loss is added (i.e., the auxiliary loss block is placed after the bias construction block, before or after the main loss block — either is fine as long as `layer1_joints` is accessible).

**Placing `layer1_joints` in scope outside the `if` block**: to keep the code clean, move the `layer1_joints` variable to be accessible for the auxiliary loss. One clean approach is:

```python
layer1_joints = None  # will be set if use_reproj_bias
layer1_decoded = None

if self.use_reproj_bias and self.num_decoder_layers > 1:
    # ... compute spatial_tmp, queries_tmp ...
    layer1_decoded = self.decoder_layers[0](queries_tmp, spatial_tmp)
    layer1_joints = self.joints_out(layer1_decoded)   # (B, J, 3)
    layer1_depth  = self.depth_out(layer1_decoded[:, 0])
    layer1_uv     = self.uv_out(layer1_decoded[:, 0])
    # ... bias construction ...

pred = self.forward(feats)

# Collect GT joints (as in baseline)
gt_joints = torch.cat([d.gt_instances.lifting_target
                        for d in batch_data_samples], dim=0)
if gt_joints.dim() == 4:
    gt_joints = gt_joints.squeeze(1)
gt_joints = gt_joints.to(pred['joints'].device)

# ... collect gt_depth, gt_uv ...

_BODY = list(range(0, 22))
losses = dict()
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])
losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
    pred['pelvis_depth'], gt_depth)
losses['loss/uv/train'] = self.loss_weight_uv * self.loss_uv_module(
    pred['pelvis_uv'], gt_uv)

# Auxiliary intermediate loss (Design B/C)
if self.aux_loss_weight > 0.0 and layer1_joints is not None:
    losses['loss/joints_aux/train'] = (
        self.aux_loss_weight * self.loss_joints_module(
            layer1_joints[:, _BODY], gt_joints[:, _BODY]))
```

This pattern cleanly separates the concerns: `layer1_joints` is computed in the bias block and reused in the auxiliary loss block.

---

## Config Changes (`config.py`)

Identical to the baseline except for the head dict, which sets `aux_loss_weight=0.4`:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    num_decoder_layers=2,
    use_reproj_bias=True,
    reproj_bias_sigma=4.0,
    reproj_bias_gamma=2.0,
    reproj_bias_learnable=False,
    aux_loss_weight=0.4,
    feat_h=40,
    feat_w=24,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config sections (optimizer, LR schedule, data pipeline, hooks) are identical to the baseline.

---

## Invariants the Builder Must Preserve

1. Same as design001 invariants 1–9.
2. The auxiliary loss uses the **same** `loss_joints_module` (SoftWeightSmoothL1Loss with beta=0.05) as the main joint loss — no separate loss module needed.
3. The auxiliary loss key `'loss/joints_aux/train'` must differ from the main joint loss key `'loss/joints/train'` so MMEngine logs both separately.
4. No intermediate depth or UV losses on layer-1 output (to avoid pelvis regression degradation).
5. The intermediate layer-1 forward in `loss()` uses normal autograd (no `torch.no_grad()`), so gradients from the auxiliary loss flow back through layer-1 weights.
6. `gt_joints` must be on the same device as `layer1_joints` when computing the auxiliary loss. Since `layer1_joints` comes from layer-1 applied to the same feature tensor as `pred`, and `gt_joints` is moved to `pred['joints'].device`, they will be on the same device.

---

## Expected Behaviour

- Stage-1 `composite_val` target: < 332 (better than design001's < 340 target due to auxiliary supervision bootstrapping early training).
- Stage-2 `composite_val` target: < 222 (competitive with or better than idea001/design001's 224.52).
- `mpjpe_body_val` stage-1 target: < 190 mm.
- The auxiliary loss should improve layer-1 prediction accuracy from early epochs, producing better-quality reprojection biases for layer-2 from epoch 1 onward.
- Design B is the primary hypothesis candidate: the combination of auxiliary supervision + dynamic geometric feedback is expected to produce the clearest improvement.

**Design Description:** 2-layer cascaded decoder with dynamic Gaussian reprojection bias, auxiliary loss (weight=0.4), and learnable per-joint σ and γ initialized to (4.0, 2.0).

**Starting Point:** `baseline/`

---

## Overview

**Algorithm:** Two-stage cascaded decoding with geometry-guided cross-attention, auxiliary intermediate supervision, and learnable per-joint Gaussian bandwidth. Layer 1 produces an intermediate 3D pose estimate; those predicted joints are reprojected to feature-grid coordinates; per-joint Gaussian additive biases (with learnable σ and γ) are injected into layer 2's cross-attention. Learnable σ allows narrow focus for distal joints and broad focus for proximal joints. Auxiliary body-joint loss (weight=0.4) on layer-1 output supervises the intermediate predictions.

Same as Design B (idea022/design002) — two decoder layers, dynamic Gaussian reprojection bias from layer-1 predictions, auxiliary joint loss (weight=0.4) — but replaces the fixed scalar `reproj_bias_sigma` and `reproj_bias_gamma` with **learnable per-joint `nn.Parameter` tensors**:

- `self.bias_sigma = nn.Parameter(torch.ones(num_joints) * 4.0)` — per-joint Gaussian bandwidth in grid cells.
- `self.bias_gamma = nn.Parameter(torch.ones(num_joints) * 2.0)` — per-joint Gaussian amplitude.

`bias_sigma` is passed through `F.softplus` before use to ensure positivity. The model can learn narrow focus for precisely-locatable distal joints (wrists, ankles) and broad focus for proximal joints with higher positional uncertainty (spine, pelvis). Initialization at (σ=4, γ=2) matches Design B's fixed values so early training behaviour is identical.

---

## Files to Change

### 1. `pelvis_utils.py`

Add the same `project_joints_to_feat_grid` helper as in design001 (identical code). Builder must check and skip if already present.

### 2. `pose3d_transformer_head.py`

All structural changes from design001 and design002 apply here with the following differences:

#### 2a. Imports

Same as design002 — add `F`, `np`, `recover_pelvis_3d`, `project_joints_to_feat_grid`.

#### 2b. `_build_gaussian_bias` module-level function

Same as design001/002 — identical implementation.

#### 2c. `_DecoderLayer.forward`

Same as design001/002 — accept optional `cross_attn_bias`.

#### 2d. `Pose3dTransformerHead.__init__` — add learnable bias parameters

New parameters (identical signature to design002):
```python
num_decoder_layers: int = 1,
use_reproj_bias: bool = False,
reproj_bias_sigma: float = 4.0,
reproj_bias_gamma: float = 2.0,
reproj_bias_learnable: bool = False,
aux_loss_weight: float = 0.0,
feat_h: int = 40,
feat_w: int = 24,
```

Store same attributes as design002.

After storing attributes, add conditional creation of learnable parameters:

```python
if reproj_bias_learnable:
    self.bias_sigma = nn.Parameter(
        torch.ones(num_joints) * reproj_bias_sigma)  # (J,) initialized to 4.0
    self.bias_gamma = nn.Parameter(
        torch.ones(num_joints) * reproj_bias_gamma)  # (J,) initialized to 2.0
```

These are `nn.Parameter` instances registered on the module; they will be included in `model.parameters()` and updated by AdamW automatically.

#### 2e. `Pose3dTransformerHead.forward`

Identical to design002 — multi-layer loop with `getattr(self, '_reproj_bias', None)`.

#### 2f. `Pose3dTransformerHead.loss` — use learnable σ and γ

The bias-construction block is the same as design002 except for how `sigma` and `gamma` are computed:

Replace the fixed-tensor sigma/gamma lines in design002:
```python
sigma = torch.full(
    (self.num_joints,), self.reproj_bias_sigma,
    device=feat_coords.device, dtype=feat_coords.dtype)
gamma = torch.full(
    (self.num_joints,), self.reproj_bias_gamma,
    device=feat_coords.device, dtype=feat_coords.dtype)
```

With learnable versions (conditional on `reproj_bias_learnable`):
```python
if self.reproj_bias_learnable:
    # Apply softplus to ensure sigma > 0; cast to feat_coords dtype
    sigma = F.softplus(self.bias_sigma).to(
        device=feat_coords.device, dtype=feat_coords.dtype)  # (J,)
    gamma = self.bias_gamma.to(
        device=feat_coords.device, dtype=feat_coords.dtype)  # (J,)
else:
    sigma = torch.full(
        (self.num_joints,), self.reproj_bias_sigma,
        device=feat_coords.device, dtype=feat_coords.dtype)
    gamma = torch.full(
        (self.num_joints,), self.reproj_bias_gamma,
        device=feat_coords.device, dtype=feat_coords.dtype)
```

This conditional block ensures that the same `pose3d_transformer_head.py` code supports all three designs (A, B, C) based on the `reproj_bias_learnable` flag.

The `_build_gaussian_bias` call is unchanged — it receives `sigma` and `gamma` as `(J,)` tensors regardless of whether they are fixed or learnable.

#### 2g. Auxiliary loss in `loss()`

Identical to design002:
```python
if self.aux_loss_weight > 0.0 and layer1_joints is not None:
    losses['loss/joints_aux/train'] = (
        self.aux_loss_weight * self.loss_joints_module(
            layer1_joints[:, _BODY], gt_joints[:, _BODY]))
```

---

## Config Changes (`config.py`)

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
    reproj_bias_learnable=True,
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

Only `reproj_bias_learnable=True` differs from design002's config. All other config sections (optimizer, LR schedule, data pipeline, hooks) are identical to baseline.

---

## Invariants the Builder Must Preserve

1. All invariants from design002 (items 1–6) apply here.
2. `self.bias_sigma` and `self.bias_gamma` are `nn.Parameter` tensors — they must be created only when `reproj_bias_learnable=True`, to avoid adding unused parameters to the model in Design A/B configurations.
3. `F.softplus(self.bias_sigma)` guarantees sigma > 0. The `_build_gaussian_bias` function also clamps sigma to `>= 0.5` inside the Gaussian computation — both safeguards are present.
4. The learnable parameters `bias_sigma` and `bias_gamma` are initialized to the same values as Design B's fixed parameters (σ=4.0, γ=2.0), so the model starts from an identical state to Design B at epoch 0.
5. `bias_gamma` has no positivity constraint applied (negative gamma would invert the bias to an attention suppressor). The AdamW optimizer with `weight_decay=0.03` provides mild regularization. If the Builder observes gamma going negative during training (unlikely but possible), a `F.softplus` or `torch.abs` can be added — but do not add this proactively; preserve the unconstrained design as specified.
6. The `bias_sigma` and `bias_gamma` parameters are subject to the same optimizer and LR schedule as the decoder head parameters (not the backbone, which has `lr_mult=0.1`). This is correct — they are part of the head.
7. AMP compatibility: `self.bias_sigma` and `self.bias_gamma` are float32 parameters. They are cast to `feat_coords.dtype` (which may be float16 under AMP) before passing to `_build_gaussian_bias`. The `_build_gaussian_bias` function itself operates in the dtype of `joint_feat_coords`, so the cast ensures consistency.

---

## Expected Behaviour

- Stage-1 `composite_val` target: < 328.
- Stage-2 `composite_val` target: < 220 (primary target; improvement over best prior 224.52).
- `mpjpe_body_val` stage-2 target: < 168 mm (matching or improving on best body MPJPE of 168.79 from idea010/design002).
- Learnable per-joint bandwidths expected to converge such that: distal joints (wrists=joints 13,14; ankles=joints 7,8) have smaller σ values (narrower Gaussian focus), while proximal joints (spine, hips) have larger σ values (broader attention region).
- Design C is expected to outperform Design B by 3–8 mm body MPJPE at stage-2, driven by the joint-specific attention focus.

# Design 003 — Uncertainty Weighting on Depth/UV + Composite-Proportional Joint Anchor

**Design Description:** Depth and UV tasks get learnable uncertainty weights; joint loss uses a fixed multiplier of 2.0 (= 0.67/0.33) to reflect the composite metric's intended task weighting as a prior before uncertainty adaptation.

**Starting Point:** `baseline/`

---

## Algorithm Overview

Extends the Design B algorithm by replacing the fixed joint loss weight of 1.0 with a composite-proportional value of 2.0. The composite metric is defined as `0.67 * mpjpe_body + 0.33 * mpjpe_pelvis`, so the body task is weighted approximately twice as much as the pelvis task. Setting the joint loss multiplier to 2.0 (= 0.67/0.33 ≈ 2.03, rounded to 2.0) encodes this domain knowledge directly into the starting-point gradient balance, before the uncertainty mechanism adapts the depth/UV weights.

The rationale: if the composite metric was designed with genuine domain knowledge about task importance, incorporating its weighting as an inductive bias may accelerate convergence toward good composite scores within the 20-epoch training budget. The uncertainty mechanism on depth and UV then self-tunes the pelvis sub-tasks relative to this fixed joint anchor.

A new constructor argument `joint_loss_scale: float = 1.0` controls the fixed multiplier; in this design it is set to 2.0 via config.

---

## Files to Modify

### 1. `pose3d_transformer_head.py`

#### `__init__` additions

Add constructor parameters:

```python
uncertainty_pelvis_only: bool = False,   # may already exist from design002
joint_loss_scale: float = 1.0,
```

Store as `self.joint_loss_scale = joint_loss_scale`.

The `uncertainty_pelvis_only` logic and associated `nn.Parameter` registrations are identical to design002. If the Builder is working from a file that already has `uncertainty_pelvis_only` from design002, only `joint_loss_scale` needs to be added.

When `uncertainty_pelvis_only=True`, register:

```python
if self.uncertainty_pelvis_only:
    self.log_var_depth = nn.Parameter(torch.zeros(1))
    self.log_var_uv    = nn.Parameter(torch.zeros(1))
```

#### `loss()` modifications

Apply the joint loss scale and uncertainty weighting:

```python
raw_joints = self.joint_loss_scale * self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])
raw_depth  = self.loss_weight_depth * self.loss_depth_module(
    pred['pelvis_depth'], gt_depth)
raw_uv     = self.loss_weight_uv * self.loss_uv_module(
    pred['pelvis_uv'], gt_uv)

if self.uncertainty_pelvis_only:
    lv_d = self.log_var_depth.clamp(-4.0, 4.0)
    lv_u = self.log_var_uv.clamp(-4.0, 4.0)
    losses['loss/joints/train'] = raw_joints
    losses['loss/depth/train']  = torch.exp(-lv_d) * raw_depth + lv_d
    losses['loss/uv/train']     = torch.exp(-lv_u) * raw_uv    + lv_u
else:
    losses['loss/joints/train'] = raw_joints
    losses['loss/depth/train']  = raw_depth
    losses['loss/uv/train']     = raw_uv
```

Key points:
- `joint_loss_scale` is applied to `raw_joints` before the conditional — it takes effect whether or not `uncertainty_pelvis_only` is True.
- When `joint_loss_scale=1.0` (default) and `uncertainty_pelvis_only=False`, the code is exactly equivalent to baseline.
- The `_train_mpjpe` computation below the losses is NOT scaled by `joint_loss_scale` — it remains a plain MPJPE diagnostic in millimetres.

#### `__init__` signature (full, after change)

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    loss_joints: ConfigType = dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth:  ConfigType = dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv:     ConfigType = dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth: float = 1.0,
    loss_weight_uv:    float = 1.0,
    use_uncertainty_weighting: bool = False,
    uncertainty_pelvis_only:   bool = False,
    joint_loss_scale:          float = 1.0,
    init_cfg: OptConfigType = None,
):
```

### 2. `config.py`

Add `uncertainty_pelvis_only=True` and `joint_loss_scale=2.0` to the head dict:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss',  beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss',     beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
    uncertainty_pelvis_only=True,
    joint_loss_scale=2.0,
),
```

`use_uncertainty_weighting` is NOT set (defaults to False). Do not set `use_uncertainty_weighting=True` and `uncertainty_pelvis_only=True` simultaneously.

### 3. `pelvis_utils.py`

No changes.

---

## Exact Hyperparameters

All other hyperparameters are inherited from the baseline:

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Base LR | 1e-4 |
| Backbone LR mult | 0.1 |
| Weight decay | 0.03 |
| Gradient clip max_norm | 1.0 |
| Accumulative counts | 8 |
| Warmup epochs | 3 (LinearLR, factor 0.333) |
| LR schedule | CosineAnnealingLR epochs 3–20 |
| Batch size | 4 |
| Seed | 2026 |
| `log_var_depth` init | 0.0 (`torch.zeros(1)`) |
| `log_var_uv` init | 0.0 (`torch.zeros(1)`) |
| `log_var_*` clamp range | [-4.0, 4.0] |
| `joint_loss_scale` | 2.0 (fixed, not learned) |

The `log_var_depth` and `log_var_uv` parameters inherit the full base LR (1e-4). The `joint_loss_scale=2.0` is a scalar Python float — not a learnable parameter and not an `nn.Parameter`.

---

## Invariants to Preserve

- `persistent_workers=False` in both dataloaders — do not change.
- No Python `import` statements in `config.py`.
- `pose3d_transformer_head.py` uses absolute imports.
- Joint loss restricted to body joints indices 0–21.
- `_train_mpjpe` and `_train_mpjpe_abs` computations unchanged and not affected by `joint_loss_scale`.
- `log_var` parameters must be `nn.Parameter` (not buffers).
- Baseline behaviour fully preserved when `uncertainty_pelvis_only=False` and `joint_loss_scale=1.0`.
- `joint_loss_scale` is a pure Python float multiplier on the loss value, not a config for the loss module.

---

## Expected Behaviour

At initialisation, `log_var_depth=0`, `log_var_uv=0`, so the effective loss is:
- `loss/joints/train` = 2.0 × raw_joints
- `loss/depth/train` = 1.0 × raw_depth
- `loss/uv/train` = 1.0 × raw_uv

This immediately biases the gradient toward joint regression in proportion to the composite metric's 0.67:0.33 weighting. As training proceeds, the uncertainty mechanism adapts the depth and UV weights. If depth is too dominant, `log_var_depth` increases to down-weight it further, while the joint gradient stays at 2× throughout.

The composite target is `composite_val < 163` (baseline 169.99). Compared to Design B (design002), this variant provides a stronger bias toward joint regression from the very first iteration, which may be beneficial given the 20-epoch training budget.

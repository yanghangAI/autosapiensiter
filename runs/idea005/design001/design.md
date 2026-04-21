# Design 001 — Uncertainty Weighting on All Three Tasks (Full)

**Design Description:** Add three learnable log-variance parameters (`log_var_joints`, `log_var_depth`, `log_var_uv`) to the head; replace fixed-weight loss terms with Kendall & Gal homoscedastic uncertainty weighting on all three tasks.

**Starting Point:** `baseline/`

---

## Algorithm Overview

Apply Kendall & Gal (2018) uncertainty-weighted multi-task loss algorithm to all three regression heads simultaneously. Each task's raw loss is scaled by `exp(-log_var_i)` and a regularisation term `log_var_i` is added to prevent the parameters from collapsing to +∞. All three `log_var` parameters are `nn.Parameter` initialised to 0, recovering the baseline equal-weight loss at the start of training. The `log_var` parameters are part of the head module and are updated by AdamW at the full (unscaled) learning rate.

---

## Files to Modify

### 1. `pose3d_transformer_head.py`

#### `__init__` additions

Add the following constructor parameter:

```python
use_uncertainty_weighting: bool = False,
```

Store it as `self.use_uncertainty_weighting = use_uncertainty_weighting`.

When `use_uncertainty_weighting=True`, register three scalar `nn.Parameter` objects:

```python
if self.use_uncertainty_weighting:
    self.log_var_joints = nn.Parameter(torch.zeros(1))
    self.log_var_depth  = nn.Parameter(torch.zeros(1))
    self.log_var_uv     = nn.Parameter(torch.zeros(1))
```

Do NOT add these parameters when `use_uncertainty_weighting=False` (baseline compatibility).

#### `loss()` modifications

After computing the three raw losses:

```python
raw_joints = self.loss_joints_module(pred['joints'][:, _BODY], gt_joints[:, _BODY])
raw_depth  = self.loss_weight_depth * self.loss_depth_module(pred['pelvis_depth'], gt_depth)
raw_uv     = self.loss_weight_uv    * self.loss_uv_module(pred['pelvis_uv'], gt_uv)
```

Apply the uncertainty weighting branch:

```python
if self.use_uncertainty_weighting:
    lv_j = self.log_var_joints.clamp(-4.0, 4.0)
    lv_d = self.log_var_depth.clamp(-4.0, 4.0)
    lv_u = self.log_var_uv.clamp(-4.0, 4.0)
    losses['loss/joints/train'] = torch.exp(-lv_j) * raw_joints + lv_j
    losses['loss/depth/train']  = torch.exp(-lv_d) * raw_depth  + lv_d
    losses['loss/uv/train']     = torch.exp(-lv_u) * raw_uv     + lv_u
else:
    losses['loss/joints/train'] = raw_joints
    losses['loss/depth/train']  = raw_depth
    losses['loss/uv/train']     = raw_uv
```

Clamp range `[-4, 4]`: at `log_var=4`, `exp(-4) ≈ 0.018` (strong down-weighting); at `log_var=-4`, `exp(4) ≈ 54.6` (strong up-weighting, capped). This prevents numeric blow-up without preventing meaningful adaptation.

The clamp is applied to a local variable (`lv_j` etc.), NOT in-place on the parameter itself, so gradients flow through the `clamp` operation correctly.

No changes to `_train_mpjpe` or `_train_mpjpe_abs` computation.

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
    init_cfg: OptConfigType = None,
):
```

### 2. `config.py`

Add `use_uncertainty_weighting=True` to the head dict. Keep `loss_weight_depth=1.0` and `loss_weight_uv=1.0` (they become effective no-ops; raw losses enter the uncertainty formula already scaled by 1.0).

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
    use_uncertainty_weighting=True,
),
```

No other changes to `config.py`.

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
| `log_var_*` init | 0.0 (`torch.zeros(1)`) |
| `log_var_*` clamp range | [-4.0, 4.0] |

The `log_var` parameters are **not** under `paramwise_cfg`'s `backbone` key, so they inherit the full base learning rate of 1e-4 (not the 0.1× backbone multiplier). No special paramwise entry is needed for them.

---

## Invariants to Preserve

- `persistent_workers=False` in both dataloaders — do not change.
- No Python `import` statements in `config.py` — use only `__import__()` or literals.
- `pose3d_transformer_head.py` uses absolute imports (e.g., `from mmpose.models.heads.base_head import BaseHead`).
- Joint loss restricted to body joints indices 0–21 (`_BODY = list(range(0, 22))`).
- `_train_mpjpe` and `_train_mpjpe_abs` computations unchanged (they are diagnostic, not loss).
- Baseline behaviour is fully preserved when `use_uncertainty_weighting=False`.
- The `log_var` parameters must be `nn.Parameter` (not buffers) so AdamW updates them.

---

## Expected Behaviour

At initialisation all `log_var=0`, so `exp(0)=1` and the total loss equals `raw_loss + 0 = raw_loss`, identical to baseline. As training progresses, the model learns to down-weight tasks with high loss variance by increasing the corresponding `log_var`, and up-weight tasks that are easier or more important by decreasing it. The composite target is `composite_val < 163` (baseline 169.99).

---

## Risk Notes

- This design allows joint loss to be down-weighted if it is harder than depth/UV early in training. Monitor `log_var_joints` in training logs.
- Design B (design002) is the conservative variant that anchors joint loss to avoid this risk.

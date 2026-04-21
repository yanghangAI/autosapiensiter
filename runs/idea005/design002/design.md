# Design 002 — Uncertainty Weighting on Depth and UV Only (Anchored Joint Loss)

**Design Description:** Add learnable log-variance parameters for depth and UV tasks only; joint loss kept at fixed weight 1.0, preventing the primary task from being spuriously down-weighted.

**Starting Point:** `baseline/`

---

## Algorithm Overview

A conservative variant of the uncertainty-weighting algorithm. The joint regression loss is the primary task and its weight is the natural anchor for the optimisation. Only the two pelvis sub-tasks (`loss/depth/train` and `loss/uv/train`) receive learnable uncertainty weights. This targets the known weak point in the composite metric (pelvis MPJPE) while protecting body MPJPE from degradation caused by the joint loss being down-weighted during difficult early-training periods.

The pelvis depth loss (metres, ~2–8 m range) and pelvis UV loss (normalised [-1,1]) operate at very different scales from each other and from the joint regression loss. Allowing only these two to self-balance against each other (and against the fixed joint anchor) is the most targeted intervention.

---

## Files to Modify

### 1. `pose3d_transformer_head.py`

#### `__init__` additions

Add constructor parameter:

```python
uncertainty_pelvis_only: bool = False,
```

Store as `self.uncertainty_pelvis_only = uncertainty_pelvis_only`.

When `uncertainty_pelvis_only=True`, register only two scalar `nn.Parameter` objects:

```python
if self.uncertainty_pelvis_only:
    self.log_var_depth = nn.Parameter(torch.zeros(1))
    self.log_var_uv    = nn.Parameter(torch.zeros(1))
```

Do NOT register `log_var_joints` in this design. Do NOT add any parameter when `uncertainty_pelvis_only=False`.

#### `loss()` modifications

After computing raw losses:

```python
raw_joints = self.loss_joints_module(pred['joints'][:, _BODY], gt_joints[:, _BODY])
raw_depth  = self.loss_weight_depth * self.loss_depth_module(pred['pelvis_depth'], gt_depth)
raw_uv     = self.loss_weight_uv    * self.loss_uv_module(pred['pelvis_uv'], gt_uv)
```

Apply the partial uncertainty weighting branch:

```python
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

Clamp applied to local variable (not in-place) so gradients flow correctly.

No changes to `_train_mpjpe` or `_train_mpjpe_abs`.

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
    init_cfg: OptConfigType = None,
):
```

Note: `use_uncertainty_weighting` from design001 may already be present if the Builder applies designs sequentially to the same file. If so, the `uncertainty_pelvis_only` parameter is simply added alongside it; the two flags are independent and non-overlapping branches.

### 2. `config.py`

Add `uncertainty_pelvis_only=True` to the head dict. Keep `loss_weight_depth=1.0` and `loss_weight_uv=1.0` as before.

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
),
```

`use_uncertainty_weighting` is NOT set (defaults to False). Do not set both flags to True simultaneously.

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
| Joint loss fixed weight | 1.0 (unchanged from baseline) |

The `log_var_depth` and `log_var_uv` parameters inherit the full base LR (1e-4); they are not under the backbone `paramwise_cfg` key.

---

## Invariants to Preserve

- `persistent_workers=False` in both dataloaders — do not change.
- No Python `import` statements in `config.py`.
- `pose3d_transformer_head.py` uses absolute imports.
- Joint loss restricted to body joints indices 0–21.
- `_train_mpjpe` and `_train_mpjpe_abs` computations unchanged.
- `log_var` parameters must be `nn.Parameter` (not buffers).
- Baseline behaviour fully preserved when `uncertainty_pelvis_only=False`.

---

## Expected Behaviour

At initialisation, `log_var_depth=0` and `log_var_uv=0`, so all three losses are applied with weight 1.0 — identical to baseline. During training:
- If depth loss magnitude dominates (its ~2–8 m scale vs. joints ~0–1 m), `log_var_depth` will increase, down-weighting depth and up-weighting joint regression by proxy (since joint loss is the fixed anchor).
- The body MPJPE is protected from degradation because joint loss weight is constant at 1.0.
- The composite target is `composite_val < 163` (baseline 169.99).

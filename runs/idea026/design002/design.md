# Design 002 — Per-Joint Per-Axis Laplace NLL (3 Scale Parameters per Joint)

**Design Description:** Add per-token `log_scale_out: Linear(256, 3)` applied to each body query independently; replace SoftWeightSmoothL1 body-joint loss with Laplace NLL using per-joint per-axis uncertainty (separate scale for x, y, z); zero-init → exact L1 baseline at training start.

**Starting Point:** `baseline/`

---

## Overview

Extend Design A by allowing the model to express separate scale (uncertainty) for each of the three spatial axes (x, y, z) at each joint.

**Algorithm summary:** For each of the 22 body query tokens, a `Linear(hidden_dim, 3)` head predicts `log_s` (one log-scale per axis). The algorithm computes `s = exp(log_s.clamp(-10, 5)).clamp(min=1e-4)` giving shape `(B, 22, 3)`, then minimises the element-wise Laplace NLL: `mean(log(2s) + |pred_joint - gt_joint| / s)` where both `s` and `abs_err` are `(B, 22, 3)` — no broadcasting needed. The algorithm allows depth-axis (X) uncertainty to differ from Y/Z uncertainty per joint, matching the anisotropic difficulty structure of 3D pose estimation from RGBD data. In BEDLAM2's coordinate system where X is the forward/depth axis, the model is expected to learn higher uncertainty for the depth axis (X) at end-effector joints (wrists, ankles) that suffer from depth ambiguity. This provides a richer inductive bias than shared-scalar uncertainty.

---

## Files to Change

1. `pose3d_transformer_head.py`
2. `config.py`

No changes to `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, or training infrastructure.

---

## `pose3d_transformer_head.py` — Exact Changes

### 1. `__init__` — new parameters and module

Add the following new keyword arguments to `__init__`, with defaults preserving baseline behaviour:

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    loss_joints: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                   beta=0.05, loss_weight=1.0),
    loss_depth: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                  beta=0.05, loss_weight=1.0),
    loss_uv: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                               beta=0.05, loss_weight=1.0),
    loss_weight_depth: float = 1.0,
    loss_weight_uv: float = 1.0,
    # NEW: per-joint Laplace uncertainty parameters
    use_per_joint_uncertainty: bool = False,
    per_joint_uncertainty_mode: str = 'shared_scalar',  # 'shared_scalar' or 'per_axis'
    log_scale_out_features: int = 1,                    # 1 for shared_scalar, 3 for per_axis
    laplace_entropy_weight: float = 1.0,
    laplace_entropy_weight_start: float = 1.0,
    laplace_entropy_weight_end: float = 1.0,
    laplace_entropy_anneal_steps: int = 0,
    init_cfg: OptConfigType = None,
):
```

Store all new args as instance attributes:
```python
self.use_per_joint_uncertainty = use_per_joint_uncertainty
self.per_joint_uncertainty_mode = per_joint_uncertainty_mode
self.log_scale_out_features = log_scale_out_features
self.laplace_entropy_weight = laplace_entropy_weight
self.laplace_entropy_weight_start = laplace_entropy_weight_start
self.laplace_entropy_weight_end = laplace_entropy_weight_end
self.laplace_entropy_anneal_steps = laplace_entropy_anneal_steps
```

Conditionally create the log-scale output head:
```python
if self.use_per_joint_uncertainty:
    self.log_scale_out = nn.Linear(hidden_dim, log_scale_out_features)
    nn.init.zeros_(self.log_scale_out.weight)
    nn.init.zeros_(self.log_scale_out.bias)
    self._loss_call_count = 0
```

`log_scale_out_features=3` for this design (Design B). Applied per body query token, it outputs `(B, 22, 3)` — one scale per axis per joint.

### 2. `_init_head_weights` — no change

The existing `_init_head_weights` initialises `joints_out`, `depth_out`, `uv_out`. The `log_scale_out` is zero-initialised immediately after creation in `__init__`, so it does not need to be added to `_init_head_weights`.

### 3. `forward` — add log_scale to output dict

After the existing output projections, add:

```python
if self.use_per_joint_uncertainty:
    # Apply log_scale_out to each body query token independently
    # decoded[:, :22, :] has shape (B, 22, hidden_dim)
    body_decoded = decoded[:, :22, :]  # (B, 22, hidden_dim)
    log_scale = self.log_scale_out(body_decoded)  # (B, 22, log_scale_out_features)
    # For Design B (per_axis): log_scale shape is (B, 22, 3)
    pred['log_scale'] = log_scale
```

The `nn.Linear(hidden_dim, 3)` is applied token-by-token via PyTorch broadcasting: input `(B, 22, hidden_dim)` → output `(B, 22, 3)`. No reshape needed since `log_scale_out_features=3` matches the axis count.

Return dict now optionally contains `'log_scale'` key when `use_per_joint_uncertainty=True`.

### 4. `loss` — replace body-joint loss with Laplace NLL (per-axis)

Replace:
```python
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])
```

With:
```python
if self.use_per_joint_uncertainty:
    log_s = pred['log_scale']                          # (B, 22, 3) for per_axis
    log_s = log_s.clamp(-10.0, 5.0)                   # AMP safety: s in [4.5e-5, 148]
    s = torch.exp(log_s)                               # (B, 22, 3)
    s = s.clamp(min=1e-4)                              # prevent log(2*0)
    abs_err = (pred['joints'][:, _BODY] - gt_joints[:, _BODY]).abs()  # (B, 22, 3)
    # Entropy weight (no annealing in Design B)
    if self.laplace_entropy_anneal_steps > 0:
        self._loss_call_count += 1
        progress = min(1.0, self._loss_call_count / float(self.laplace_entropy_anneal_steps))
        w_ent = self.laplace_entropy_weight_start + progress * (
            self.laplace_entropy_weight_end - self.laplace_entropy_weight_start)
    else:
        w_ent = self.laplace_entropy_weight
    # Laplace NLL: log(2s) + |mu - y| / s — element-wise for all (B, 22, 3)
    nll = w_ent * torch.log(2.0 * s) + abs_err / s    # (B, 22, 3)
    losses['loss/joints/train'] = nll.mean()
else:
    losses['loss/joints/train'] = self.loss_joints_module(
        pred['joints'][:, _BODY], gt_joints[:, _BODY])
```

Key differences from Design A:
- `log_s` shape is `(B, 22, 3)` instead of `(B, 22, 1)`.
- `s` shape is `(B, 22, 3)` — no broadcasting needed, `s` and `abs_err` are element-wise.
- All three axes have independent scale predictions for every joint.
- The entropy term `log(2s)` is applied independently per axis — the model must pay the entropy cost for each axis separately, preventing it from freely inflating one axis to compensate for another.

### 5. `predict` — no change

`predict()` reads only `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']`. The `pred['log_scale']` key is present but never accessed. No modification needed.

### 6. Invariants preserved

- `_BODY = list(range(0, 22))` remains defined in `loss()`, unchanged.
- Pelvis depth and UV losses unchanged.
- `_train_mpjpe` and `_train_mpjpe_abs` computed with `torch.no_grad()` unchanged.
- `predict()` output structure unchanged.
- All existing `__init__` parameters retain defaults — baseline preserved when `use_per_joint_uncertainty=False`.

---

## `config.py` — Exact Changes

In `model.head` dict, add the following new keys (all literals, no imports):

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
    # Design B: per-joint per-axis Laplace NLL
    use_per_joint_uncertainty=True,
    per_joint_uncertainty_mode='per_axis',
    log_scale_out_features=3,
    laplace_entropy_weight=1.0,
    laplace_entropy_anneal_steps=0,
),
```

All other config values (optimizer, LR schedule, data pipeline, hooks, etc.) remain identical to baseline.

---

## Expected Behaviour

- At training start: `log_scale_out` is zero-initialised → all `log_s = 0` → all `s = 1` → Laplace NLL = `log(2) + |μ - y|`. Gradient identical to L1 baseline.
- After early training: the model learns to assign higher uncertainty (larger `s`) to the depth axis (X component, index 0) for end-effector joints, since depth is systematically harder to predict from RGBD crops. The Y and Z components (horizontal/vertical in image plane) are expected to have lower uncertainty, maintaining stronger L1 gradients.
- Independent per-axis scales allow the model to express anisotropic difficulty per joint, which is a well-motivated inductive bias for 3D pose estimation.
- `log_scale_out` adds `256 * 3 + 3 = 771` parameters — negligible.

---

## Constraints and Edge Cases

1. **AMP / float16**: `log_s.clamp(-10, 5)` before `torch.exp()` is mandatory. Without clamping, float16 can overflow.
2. **Scale clamp after exp**: `s.clamp(min=1e-4)` prevents `log(2 * 0) = -inf`.
3. **No broadcast**: for `per_axis` mode, `s` is `(B, 22, 3)` and `abs_err` is `(B, 22, 3)` — element-wise multiplication, no broadcast needed. This is the key difference from Design A.
4. **Entropy cost per axis**: the entropy term is `log(2 * s_x) + log(2 * s_y) + log(2 * s_z)` per joint — the model cannot inflate one axis scale without paying the entropy cost for that axis.
5. **Baseline fallback**: when `use_per_joint_uncertainty=False`, original `loss_joints_module` path taken exactly.
6. **No changes to `pelvis_utils.py`**.
7. **MMEngine config compliance**: all new config values are bool/int/float/str literals. No Python `import` statements in config.
8. **Metric invariance**: `BedlamMPJPEMetric` receives only `pred['joints']` (shape `(B, 70, 3)`, unchanged). `log_scale` never reaches the metric.
9. **Implementation note**: the same `__init__` signature as Design A is used — the Builder can implement a single unified `__init__` and `forward`/`loss` that handles both `shared_scalar` and `per_axis` modes. Design A and Design B differ only in the `log_scale_out_features` config value (1 vs 3) and the shape of `s` in the loss computation. When `log_scale_out_features=1`, `s` broadcasts; when `log_scale_out_features=3`, `s` is element-wise.

# Design 001 — Per-Joint Scalar Laplace NLL (Shared Scalar per Joint)

**Design Description:** Add per-token `log_scale_out: Linear(256, 1)` applied to each body query independently; replace SoftWeightSmoothL1 body-joint loss with Laplace NLL using per-joint shared-scalar uncertainty; zero-init scale → exact L1 baseline at training start.

**Starting Point:** `baseline/`

---

## Overview

Introduce a learned per-joint scalar scale (one value per joint, shared across x/y/z) through a second output head applied independently to each of the 22 body query tokens. Replace the fixed SoftWeightSmoothL1 body-joint loss with Laplace negative log-likelihood (NLL), allowing the model to adaptively down-weight hard/ambiguous joints.

**Algorithm summary:** For each body query token (22 total), a `Linear(hidden_dim, 1)` head predicts `log_s` (log scale). The algorithm computes `s = exp(log_s.clamp(-10, 5)).clamp(min=1e-4)`, then minimises the Laplace NLL: `mean(log(2s) + |pred_joint - gt_joint| / s)` over all batch items, joints, and axes. At init (`log_s=0, s=1`), this is equivalent to pure L1 regression. Over training, the algorithm drives `s` toward each joint's empirical prediction error — high-error joints get higher `s`, lower-error joints get lower `s`, naturally routing gradient signal to correctable errors.

---

## Files to Change

1. `pose3d_transformer_head.py`
2. `config.py`

No changes to `pelvis_utils.py`, `bedlam_metric.py`, backbone, data pipeline, or training infrastructure.

---

## `pose3d_transformer_head.py` — Exact Changes

### 1. `__init__` — new parameters and module

Add the following new keyword arguments to `__init__`, with defaults that preserve baseline behaviour (all False/1.0/0):

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

`log_scale_out_features=1` for this design (Design A). Applied per body query token, it outputs `(B, 22, 1)` — one scalar scale per joint.

### 2. `_init_head_weights` — no change

The existing `_init_head_weights` initialises `joints_out`, `depth_out`, `uv_out`. The `log_scale_out` is zero-initialised immediately after creation in `__init__`, so it does not need to be added to `_init_head_weights`.

### 3. `forward` — add log_scale to output dict

After the existing output projections (`joints`, `pelvis_depth`, `pelvis_uv`), add:

```python
if self.use_per_joint_uncertainty:
    # Apply log_scale_out to each body query token independently
    # decoded[:, :22, :] has shape (B, 22, hidden_dim)
    body_decoded = decoded[:, :22, :]  # (B, 22, hidden_dim)
    log_scale = self.log_scale_out(body_decoded)  # (B, 22, log_scale_out_features)
    # log_scale shape: (B, 22, 1) for shared_scalar; (B, 22, 3) for per_axis
    pred['log_scale'] = log_scale
```

The `nn.Linear(hidden_dim, log_scale_out_features)` is applied token-by-token via PyTorch broadcasting: input `(B, 22, hidden_dim)` → output `(B, 22, log_scale_out_features)`. This is standard and correct.

Return dict now optionally contains `'log_scale'` key when `use_per_joint_uncertainty=True`.

### 4. `loss` — replace body-joint loss with Laplace NLL

Replace:
```python
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])
```

With:
```python
if self.use_per_joint_uncertainty:
    log_s = pred['log_scale']                          # (B, 22, 1)
    log_s = log_s.clamp(-10.0, 5.0)                   # AMP safety: s in [4.5e-5, 148]
    s = torch.exp(log_s)                               # (B, 22, 1)
    s = s.clamp(min=1e-4)                              # prevent log(2*0)
    abs_err = (pred['joints'][:, _BODY] - gt_joints[:, _BODY]).abs()  # (B, 22, 3)
    # Entropy weight (no annealing in Design A)
    if self.laplace_entropy_anneal_steps > 0:
        self._loss_call_count += 1
        progress = min(1.0, self._loss_call_count / float(self.laplace_entropy_anneal_steps))
        w_ent = self.laplace_entropy_weight_start + progress * (
            self.laplace_entropy_weight_end - self.laplace_entropy_weight_start)
    else:
        w_ent = self.laplace_entropy_weight
    # Laplace NLL: log(2s) + |mu - y| / s
    nll = w_ent * torch.log(2.0 * s) + abs_err / s    # (B, 22, 3)
    losses['loss/joints/train'] = nll.mean()
else:
    losses['loss/joints/train'] = self.loss_joints_module(
        pred['joints'][:, _BODY], gt_joints[:, _BODY])
```

Key details:
- `_BODY = list(range(0, 22))` remains unchanged (defined earlier in `loss()`).
- `log_s.clamp(-10.0, 5.0)` before `exp` prevents float16 overflow in AMP (exp(5)≈148, exp(-10)≈4.5e-5).
- `s.clamp(min=1e-4)` prevents `log(2 * 0)` from producing `-inf`.
- `w_ent = self.laplace_entropy_weight` (= 1.0 for Design A since `laplace_entropy_anneal_steps=0`).
- `nll` broadcast: `s` is `(B, 22, 1)`, `abs_err` is `(B, 22, 3)` — PyTorch broadcasts correctly to `(B, 22, 3)`.
- `.mean()` averages over batch, joints, and axes — scalar loss.

### 5. `predict` — no change

`predict()` reads only `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']` from the output dict. The `pred['log_scale']` key is present but never accessed in `predict()`. No modification needed.

### 6. Invariants preserved

- `_BODY = list(range(0, 22))` remains defined in `loss()`, unchanged.
- Pelvis depth and UV losses are unchanged.
- `_train_mpjpe` and `_train_mpjpe_abs` attributes computed with `torch.no_grad()` remain unchanged.
- `predict()` output structure (InstanceData fields) unchanged.
- All existing `__init__` parameters retain their defaults — baseline behaviour preserved when `use_per_joint_uncertainty=False`.

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
    # Design A: per-joint shared-scalar Laplace NLL
    use_per_joint_uncertainty=True,
    per_joint_uncertainty_mode='shared_scalar',
    log_scale_out_features=1,
    laplace_entropy_weight=1.0,
    laplace_entropy_anneal_steps=0,
),
```

All other config values (optimizer, LR schedule, data pipeline, hooks, etc.) remain identical to baseline.

---

## Expected Behaviour

- At training start: `log_scale_out` is zero-initialised → `log_s = 0` → `s = 1` → Laplace NLL = `log(2) + |μ - y|`. The gradient w.r.t. `μ` is `sign(μ-y)`, identical to L1. Training starts identically to baseline.
- After a few hundred steps: joints with consistently large errors will have `log_s` driven negative (s < 1), amplifying their gradient; joints with small errors will have `log_s` driven positive (s > 1), dampening their gradient. The network focuses learning on correctable errors.
- The entropy term `w_ent * log(2s)` with `w_ent = 1.0` prevents scale collapse (s→0) and scale explosion (s→∞).
- `log_scale_out` adds `256 * 1 + 1 = 257` parameters — negligible.
- `pred['log_scale']` in the forward output dict does not affect `predict()` or the metric.

---

## Constraints and Edge Cases

1. **AMP / float16**: `log_s.clamp(-10, 5)` before `torch.exp()` is mandatory. Without clamping, float16 can overflow at `exp(88)`. The clamp gives `s ∈ [4.5e-5, 148.4]`.
2. **Scale clamp after exp**: `s.clamp(min=1e-4)` prevents `log(2 * 0) = -inf` in the NLL.
3. **Broadcast correctness**: `s` shape `(B, 22, 1)` broadcast against `abs_err` shape `(B, 22, 3)` is valid in PyTorch and intentional.
4. **Baseline fallback**: when `use_per_joint_uncertainty=False` (the default), the original `loss_joints_module` path is taken exactly — zero regression risk.
5. **No changes to `pelvis_utils.py`**: this file is not modified.
6. **MMEngine config compliance**: all new config values are bool/int/float/str literals. No Python `import` statements in config.
7. **Metric invariance**: `BedlamMPJPEMetric` receives only `pred['joints']` (shape unchanged). `log_scale` never reaches the metric.

# Design 003 — Per-Joint Scalar Laplace NLL + Entropy Weight Annealing

**Design Description:** Same as Design A (shared scalar per joint) but with linear entropy weight annealing from 0.1 to 1.0 over 500 gradient steps, allowing scale to grow freely early in training before progressively forcing commitment to tight uncertainty estimates.

**Starting Point:** `baseline/`

---

## Overview

Design C extends Design A by introducing entropy weight annealing for the `log(2s)` term.

**Algorithm summary:** Same as Design A (per-joint shared-scalar Laplace NLL), but the entropy coefficient `w_ent` is computed as: `w_ent = w_start + min(1.0, step / anneal_steps) * (w_end - w_start)` where `w_start=0.1`, `w_end=1.0`, `anneal_steps=500`, and `step` is the total number of `loss()` calls. The algorithm tracks `self._loss_call_count` (incremented each `loss()` call). For steps 0–500, `w_ent` ramps from ~0.1 to 1.0; for steps >500, `w_ent` stays at 1.0. The NLL is: `mean(w_ent * log(2s) + |pred_joint - gt_joint| / s)`. Low `w_ent` early in training allows `s` to adapt freely without destabilising the loss; full `w_ent` later enforces tight uncertainty calibration. Early in training (first ~500 gradient steps), the entropy penalty weight starts at 0.1, allowing the model to freely grow `s` without strong penalisation. This avoids the destabilisation that can occur when the entropy term dominates and large initial prediction errors drive `log_s` negative (s < 1), amplifying already-noisy gradients. Over 500 steps the penalty linearly ramps to 1.0, after which training behaves identically to Design A.

The 500-step annealing window covers approximately the first 5 epochs (based on ~100 gradient steps/epoch at batch=4, accum=8, effective_batch=32 on ~500 training samples).

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

`log_scale_out_features=1` for Design C (shared scalar per joint, same as Design A).

### 2. `_init_head_weights` — no change

The existing `_init_head_weights` initialises `joints_out`, `depth_out`, `uv_out`. The `log_scale_out` is zero-initialised immediately after creation in `__init__`, so it does not need to be added to `_init_head_weights`.

### 3. `forward` — add log_scale to output dict

After the existing output projections, add:

```python
if self.use_per_joint_uncertainty:
    # Apply log_scale_out to each body query token independently
    body_decoded = decoded[:, :22, :]  # (B, 22, hidden_dim)
    log_scale = self.log_scale_out(body_decoded)  # (B, 22, 1)
    pred['log_scale'] = log_scale
```

This is identical to Design A. `log_scale_out_features=1`, so output is `(B, 22, 1)`.

### 4. `loss` — Laplace NLL with entropy weight annealing

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
    # Entropy weight annealing: Design C uses laplace_entropy_anneal_steps=500
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

Key details for Design C:
- `laplace_entropy_anneal_steps=500` → the `if self.laplace_entropy_anneal_steps > 0` branch is taken.
- `self._loss_call_count` starts at 0 (set in `__init__`) and increments each time `loss()` is called.
- `progress = min(1.0, step / 500)` goes from 0.0 at step 1 to 1.0 at step 500.
- `w_ent = 0.1 + progress * (1.0 - 0.1)`:
  - At step 0 (before first call): progress=0.0 → w_ent=0.1 (but `_loss_call_count` is incremented at start of loss, so first call gives progress=1/500=0.002 → w_ent≈0.1018).
  - At step 500: progress=1.0 → w_ent=1.0.
  - After step 500: progress stays at 1.0 → w_ent stays at 1.0.
- At `w_ent=0.1`, the entropy term is `0.1 * log(2s)`. With `s=1` at init, this is `0.1 * log(2) ≈ 0.069`, while the L1 term `abs_err / s = |μ - y|` is numerically dominant. The model can grow `s` early without large entropy penalty.
- At `w_ent=1.0` (after step 500), behaviour is identical to Design A.

### 5. `predict` — no change

`predict()` reads only `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']`. No modification needed.

### 6. Invariants preserved

- `_BODY = list(range(0, 22))` unchanged.
- Pelvis depth and UV losses unchanged.
- `_train_mpjpe` and `_train_mpjpe_abs` unchanged.
- `predict()` output structure unchanged.
- Baseline behaviour when `use_per_joint_uncertainty=False`.

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
    # Design C: per-joint shared-scalar Laplace NLL + entropy annealing
    use_per_joint_uncertainty=True,
    per_joint_uncertainty_mode='shared_scalar',
    log_scale_out_features=1,
    laplace_entropy_weight_start=0.1,
    laplace_entropy_weight_end=1.0,
    laplace_entropy_anneal_steps=500,
),
```

Note: `laplace_entropy_weight` (the static weight used when `anneal_steps=0`) is not set here because the annealing branch is active. The `__init__` default of `laplace_entropy_weight=1.0` applies but is never used since `laplace_entropy_anneal_steps=500 > 0`.

All other config values (optimizer, LR schedule, data pipeline, hooks, etc.) remain identical to baseline.

---

## Expected Behaviour

- **Steps 0–500 (approx. epochs 0–5)**: entropy weight ramps from ~0.1 to 1.0. The model can freely adjust `s` without being dominated by the entropy penalty. This prevents the scenario where large initial errors drive `log_s` sharply negative (making `s` small), which would amplify already-noisy gradients and destabilise early training.
- **Steps 500+ (approx. epochs 5–20)**: entropy weight stays at 1.0. Behaviour is identical to Design A. The model has already adapted `s` to its typical prediction errors, so full entropy penalty now refines the uncertainty estimates while continuing to route gradients adaptively.
- **Expected training dynamics**: smoother loss curve in epochs 0–5 compared to Design A, potentially converging to a lower final loss by avoiding early instability.
- `log_scale_out` adds `256 * 1 + 1 = 257` parameters — negligible.

---

## Constraints and Edge Cases

1. **AMP / float16**: `log_s.clamp(-10, 5)` before `torch.exp()` is mandatory.
2. **Scale clamp after exp**: `s.clamp(min=1e-4)` prevents `log(2 * 0) = -inf`.
3. **Broadcast**: `s` shape `(B, 22, 1)` broadcasts with `abs_err` shape `(B, 22, 3)` correctly.
4. **Counter initialisation**: `self._loss_call_count = 0` is set in `__init__` when `use_per_joint_uncertainty=True`. It is not reset between epochs — it counts total gradient steps across the entire training run. After 500 steps, the annealing completes and `w_ent` stays at 1.0 for the remaining training.
5. **Annealing schedule arithmetic**: `self._loss_call_count` is incremented BEFORE `progress` is computed, so the first call gives `progress = 1/500 = 0.002` (not 0.0). This is correct — the entropy weight is never exactly 0.1 (it starts at 0.1018 on the first call), but this difference is negligible.
6. **Resume safety**: if training is preempted and resumed from checkpoint, `self._loss_call_count` is reset to 0 (since it is a Python attribute, not a registered buffer). This means the annealing restarts from the beginning after resume. This is acceptable: if the run resumes at epoch 5+ (after 500 steps), the model has already adapted `s` values, and restarting the annealing for another ~500 steps at `w_ent=0.1` provides a gentle re-warm rather than a harmful restart. If the Builder or Reviewer considers this a concern, `_loss_call_count` could be registered as a non-persistent buffer, but this is not required for correctness.
7. **No changes to `pelvis_utils.py`**.
8. **MMEngine config compliance**: all new config values are bool/int/float/str literals. No Python `import` statements in config.
9. **Metric invariance**: `BedlamMPJPEMetric` receives only `pred['joints']` (unchanged shape). `log_scale` never reaches the metric.
10. **Implementation note**: Design A, B, and C share the same `__init__` signature. The Builder should implement all three designs' head code in a single unified `pose3d_transformer_head.py` (i.e., the `__init__` handles all three via the config kwargs). Each design's `config.py` sets the appropriate combination of kwargs. This is the recommended approach.

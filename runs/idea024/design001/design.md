**Design Description:** EMA per-joint difficulty weighting (alpha=0.5, linear normalisation) — mild focusing that redistributes body-joint gradient toward harder joints while preserving total gradient scale.

**Starting Point:** `baseline/`

---

## Overview

Add an online difficulty-tracking mechanism to `Pose3dTransformerHead` that maintains an exponential moving average (EMA) of the per-joint MPJPE across training batches, then uses those difficulty estimates to compute per-joint loss weights. This design uses **alpha=0.5** (square-root focusing, mild) with **linear normalisation** — the conservative diagnostic that tests whether per-joint difficulty signal helps at all.

## Algorithm

The core algorithm is per-joint EMA difficulty weighting:

1. After each forward pass, compute per-joint batch-mean MPJPE (in mm) for the 22 body joints.
2. Update an EMA buffer: `ema[j] ← β * ema[j] + (1−β) * batch_err[j]` with `β=0.99`.
3. Derive per-joint weights: `w[j] = (ema[j] / mean(ema))^alpha`, then renormalise: `w ← 22 * w / sum(w)`.
4. Apply weights to per-joint residuals before smooth-L1: `loss = mean(smooth_l1(pred−gt) * w)`.
5. With `alpha=0.5` (this design): harder joints get `sqrt`-scaled upweighting; the total gradient magnitude is preserved (sum of weights = 22).

No architectural changes, no data pipeline changes, no pelvis-path changes. All changes are in `pose3d_transformer_head.py` and `config.py`.

---

## Files to Change

1. `pose3d_transformer_head.py`
2. `config.py`

`pelvis_utils.py` — unchanged.

---

## `pose3d_transformer_head.py` — Detailed Changes

### 1. `__init__` — new parameters and buffers

Add these keyword arguments to `Pose3dTransformerHead.__init__` **after** the existing `loss_weight_uv` parameter and **before** `init_cfg`:

```python
per_joint_difficulty_weighting: bool = False,
ema_alpha: float = 0.5,
ema_momentum: float = 0.99,
```

Store them as instance attributes immediately after `super().__init__`:

```python
self.per_joint_difficulty_weighting = per_joint_difficulty_weighting
self.ema_alpha = ema_alpha
self.ema_momentum = ema_momentum
```

Register two non-learnable buffers **only when** `per_joint_difficulty_weighting=True`. Place this block after all `nn.Linear` / `nn.Embedding` definitions but before `self._init_head_weights()`:

```python
if self.per_joint_difficulty_weighting:
    self.register_buffer('joint_err_ema', torch.ones(22))
    self.register_buffer('_train_iter', torch.zeros(1, dtype=torch.long))
```

When `per_joint_difficulty_weighting=False` (default / baseline), these buffers are never created and the head behaves exactly as before.

### 2. `_get_adaptive_weights` — new method

Add this method to the class (place between `_get_pos_enc` and `forward`):

```python
def _get_adaptive_weights(self) -> torch.Tensor:
    """Compute per-joint adaptive loss weights from EMA joint error.

    Uses linear proportional normalisation raised to power ema_alpha,
    then re-normalises so weights sum to 22 (preserving gradient scale).

    Returns:
        Tensor of shape (22,), dtype float32, on the same device as
        self.joint_err_ema. Requires per_joint_difficulty_weighting=True.
    """
    ema = self.joint_err_ema.detach()           # (22,) — no gradient flow
    normalised = ema / (ema.mean() + 1e-6)      # (22,) mean ~ 1.0
    raw_w = normalised ** self.ema_alpha         # (22,) alpha=0.5 → sqrt
    w = 22.0 * raw_w / (raw_w.sum() + 1e-6)    # (22,) sum = 22
    return w
```

### 3. `loss()` — replace body joint loss computation

In the `loss()` method, locate the existing joint loss line:

```python
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])
```

Replace it with the following block:

```python
_BODY = list(range(0, 22))

if self.per_joint_difficulty_weighting:
    # --- EMA update (no gradient) ---
    with torch.no_grad():
        per_joint_err = (
            pred['joints'][:, _BODY] - gt_joints[:, _BODY]
        ).norm(dim=-1).mean(dim=0) * 1000.0       # (22,) in mm
        self.joint_err_ema = (
            self.ema_momentum * self.joint_err_ema
            + (1.0 - self.ema_momentum) * per_joint_err
        )
        self._train_iter += 1

    # --- Adaptive per-joint weights ---
    w = self._get_adaptive_weights()              # (22,)

    # --- Weighted Smooth-L1 (manual, beta=0.05 matching baseline) ---
    # SoftWeightSmoothL1Loss.forward(output, target, target_weight) only
    # supports per-sample scalar weights, not per-joint weights.
    # We therefore compute the smooth-L1 manually.
    pred_j = pred['joints'][:, _BODY]            # (B, 22, 3)
    gt_j   = gt_joints[:, _BODY]                 # (B, 22, 3)
    diff   = (pred_j - gt_j).abs()               # (B, 22, 3)
    beta   = 0.05
    smooth_l1 = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    )                                             # (B, 22, 3)
    # w shape (22,) → broadcast over B and 3
    weighted_loss = (smooth_l1 * w.view(1, 22, 1)).mean()
    losses['loss/joints/train'] = weighted_loss
else:
    losses['loss/joints/train'] = self.loss_joints_module(
        pred['joints'][:, _BODY], gt_joints[:, _BODY])
```

Note: `_BODY = list(range(0, 22))` is already defined earlier in `loss()` in the baseline. **Remove the duplicate definition** (keep only one). The existing `_BODY` definition at the top of the loss block is fine; simply add the `if self.per_joint_difficulty_weighting` branch after it.

### 4. Edge-case invariants the Builder must preserve

- The `_train_mpjpe` and `_train_mpjpe_abs` attribute assignments in `loss()` must remain **unchanged** — they use `pred['joints'][:, _BODY]` and `gt_joints[:, _BODY]` without any weighting.
- The `depth` and `uv` loss lines must remain **unchanged**.
- The `predict()` method must remain **unchanged**.
- When `per_joint_difficulty_weighting=False` (default), the behaviour of `loss()` must be **bit-identical** to the baseline (same `loss_joints_module` call, same inputs).
- Buffer `joint_err_ema` is saved and restored by `CheckpointHook` automatically because it is registered via `register_buffer`. On SLURM preemption + resume, the EMA continues from its saved state.
- `_train_iter` must use `dtype=torch.long` to avoid floating-point counter issues.

---

## `config.py` — Changes

In the `model.head` dict, add the following three key-value pairs:

```python
per_joint_difficulty_weighting=True,
ema_alpha=0.5,
ema_momentum=0.99,
```

All are literals (bool, float, float) — fully MMEngine-compliant, no import statements needed.

The full updated `head` dict in `config.py`:

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
    per_joint_difficulty_weighting=True,
    ema_alpha=0.5,
    ema_momentum=0.99,
),
```

Everything else in `config.py` is identical to baseline.

---

## Expected Behaviour

- At step 0: `joint_err_ema = [1.0] * 22` → `w = [1.0] * 22` → weighted smooth-L1 = baseline smooth-L1. Training starts identically to baseline.
- After ~100 steps: EMA begins to reflect true per-joint difficulty. Harder distal joints (wrists idx 9, 16; ankles idx 20, 21) will have higher EMA, receiving weights > 1.0. Easy spine/torso joints (idx 0–4) will receive weights < 1.0.
- `alpha=0.5` (square-root) gives mild focusing: if a joint has 2× mean difficulty, its weight = sqrt(2) ≈ 1.41×. Not aggressive — diagnostic level.
- Total gradient scale from joint loss is preserved (sum(w) = 22 always).
- `composite_val` target: < 340 at stage-1. `mpjpe_body_val` target: < 190 at stage-1.

---

## Summary of Changes

| File | Change |
|---|---|
| `pose3d_transformer_head.py` | Add 3 new `__init__` params; register 2 buffers; add `_get_adaptive_weights()` method; replace joint loss block with conditional adaptive-weighted smooth-L1 |
| `config.py` | Add 3 literal kwargs to `model.head` dict |

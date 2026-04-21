**Design Description:** EMA per-joint difficulty weighting (alpha=1.0, linear, group-normalised upper/lower body + 5-epoch warmup ramp) — most controlled variant preventing any body region from dominating the gradient budget.

**Starting Point:** `baseline/`

---

## Overview

This design extends the per-joint EMA difficulty weighting to use **two separate EMA buffers**

## Algorithm

The core algorithm is group-normalised per-joint EMA difficulty weighting with warmup:

1. Maintain two separate EMA buffers: `upper_err_ema` (joints 0–12) and `lower_err_ema` (joints 13–21), each updated as `ema[j] ← β * ema[j] + (1−β) * batch_err[j]` with `β=0.99`.
2. Within each group, compute proportional weights: `w_group = n_joints * (ema / mean(ema))^alpha / sum(...)`, with `alpha=1.0`. Upper group weights sum to 13; lower group weights sum to 9.
3. Concatenate group weights: `w = [w_upper, w_lower]` (22 values, sum=22).
4. Apply linear warmup ramp over 5 epochs: `w_final = (1 − ramp) * ones(22) + ramp * w`, where `ramp` linearly goes from 0 to 1 over `5 * 328 = 1640` iterations.
5. Apply `w_final` to per-joint residuals before smooth-L1: `loss = mean(smooth_l1(pred−gt) * w_final)`.
6. Group normalisation ensures each body region (upper/lower) retains its proportional share of gradient budget, preventing lower body from dominating. — one for upper body joints (indices 0–12, count=13) and one for lower body joints (indices 13–21, count=9) — with weights normalised **within each group** before combining. This prevents the lower body group (which has more consistently hard joints: knees, ankles) from completely dominating the gradient budget relative to upper body.

Additionally, a **5-epoch linear warmup ramp** blends from uniform weights (baseline) to fully difficulty-weighted training, guarding against noisy early-training EMA estimates destabilising the first epochs.

No architectural changes, no data pipeline changes, no pelvis-path changes. All changes are in `pose3d_transformer_head.py` and `config.py`.

---

## Joint Group Definitions

```
Upper body (indices 0–12, 13 joints):
  0: pelvis_smpl, 1: left_hip_smpl, 2: right_hip_smpl, 3: spine1_smpl,
  4: left_knee, 5: right_knee, 6: spine2_smpl, 7: left_ankle,
  8: right_ankle, 9: spine3_smpl, 10: left_foot, 11: right_foot,
  12: neck_smpl

  (Note: anatomical "upper" vs "lower" labelling in the idea.md text is
  approximate. The implementation must use EXACTLY these index ranges:
  upper = indices 0..12 inclusive, count=13;
  lower = indices 13..21 inclusive, count=9.
  This matches the literal split described in the idea.md: "0–12" and "13–21".)

Lower body (indices 13–21, 9 joints):
  13: left_collar, 14: right_collar, 15: head_smpl, 16: left_shoulder,
  17: right_shoulder, 18: left_elbow, 19: right_elbow, 20: left_wrist,
  21: right_wrist
```

The anatomical assignment of joint names to indices is handled by the existing dataset constants. The Builder must use the index ranges exactly as given (upper=0..12, lower=13..21) — do not reorder or re-assign.

---

## Files to Change

1. `pose3d_transformer_head.py`
2. `config.py`

`pelvis_utils.py` — unchanged.

---

## `pose3d_transformer_head.py` — Detailed Changes

### 1. Add top-level constant (module level, not inside class)

After all existing module-level imports, add:

```python
_UPPER_IDX = list(range(0, 13))   # 13 joints
_LOWER_IDX = list(range(13, 22))  # 9 joints
```

These are used in both `__init__` and `loss()` to correctly slice the EMA buffers and the joint tensor.

### 2. `__init__` — new parameters and buffers

Add these keyword arguments to `Pose3dTransformerHead.__init__` **after** the existing `loss_weight_uv` parameter and **before** `init_cfg`:

```python
per_joint_difficulty_weighting: bool = False,
ema_alpha: float = 0.5,
ema_momentum: float = 0.99,
weight_norm: str = 'linear',
weight_temperature: float = 1.0,
group_normalise: bool = False,
ema_warmup_epochs: int = 0,
```

Store all as instance attributes immediately after `super().__init__`:

```python
self.per_joint_difficulty_weighting = per_joint_difficulty_weighting
self.ema_alpha = ema_alpha
self.ema_momentum = ema_momentum
self.weight_norm = weight_norm
self.weight_temperature = weight_temperature
self.group_normalise = group_normalise
self.ema_warmup_epochs = ema_warmup_epochs
```

Register non-learnable buffers **only when** `per_joint_difficulty_weighting=True`. Place this block after all `nn.Linear` / `nn.Embedding` definitions but before `self._init_head_weights()`:

```python
if self.per_joint_difficulty_weighting:
    if self.group_normalise:
        # Separate EMA buffers per group
        self.register_buffer('upper_err_ema', torch.ones(13))
        self.register_buffer('lower_err_ema', torch.ones(9))
    else:
        # Single joint-level EMA buffer (designs 1 and 2 compatibility)
        self.register_buffer('joint_err_ema', torch.ones(22))
    self.register_buffer('_train_iter', torch.zeros(1, dtype=torch.long))
```

When `per_joint_difficulty_weighting=False` (default / baseline), no buffers are created and the head behaves exactly as before.

### 3. `_get_adaptive_weights` — new method

Add this method to the class (place between `_get_pos_enc` and `forward`):

```python
def _get_adaptive_weights(self) -> torch.Tensor:
    """Compute per-joint adaptive loss weights from EMA joint error.

    When group_normalise=True: normalise difficulty weights separately
    within upper body (indices 0-12, count=13) and lower body
    (indices 13-21, count=9), then concatenate. This preserves the
    relative gradient budget between the two groups (upper body sum=13,
    lower body sum=9, total sum=22).

    When group_normalise=False: normalise over all 22 joints at once
    (designs 1 and 2 behaviour via joint_err_ema buffer).

    Optionally applies a linear warmup ramp from uniform weights to
    fully difficulty-weighted (controlled by ema_warmup_epochs and
    _train_iter buffer).

    Returns:
        Tensor of shape (22,), dtype float32, on the device of the EMA
        buffer. Requires per_joint_difficulty_weighting=True.
    """
    # Approximate iterations per epoch for train100 (~350 seqs, batch=4,
    # accum=8). Empirically ~328 gradient steps per epoch on train100.
    ITERS_PER_EPOCH = 328

    cur_iter = int(self._train_iter.item())

    if self.group_normalise:
        upper_ema = self.upper_err_ema.detach()   # (13,)
        lower_ema = self.lower_err_ema.detach()   # (9,)

        # Linear proportional within each group, power alpha
        def _group_weights(ema, n_joints):
            norm = ema / (ema.mean() + 1e-6)
            raw_w = norm ** self.ema_alpha
            return float(n_joints) * raw_w / (raw_w.sum() + 1e-6)

        w_upper = _group_weights(upper_ema, 13)   # (13,) sum=13
        w_lower = _group_weights(lower_ema, 9)    # (9,)  sum=9
        w = torch.cat([w_upper, w_lower], dim=0)  # (22,) sum=22
    else:
        # Single-buffer path (compatibility with designs 1 and 2)
        ema = self.joint_err_ema.detach()
        if self.weight_norm == 'softmax':
            import torch.nn.functional as F
            w = 22.0 * F.softmax(ema / self.weight_temperature, dim=0)
        else:
            norm = ema / (ema.mean() + 1e-6)
            raw_w = norm ** self.ema_alpha
            w = 22.0 * raw_w / (raw_w.sum() + 1e-6)

    # Warmup ramp: linearly blend from uniform (1.0) to difficulty weights
    if self.ema_warmup_epochs > 0:
        ramp_iters = self.ema_warmup_epochs * ITERS_PER_EPOCH
        ramp = min(1.0, float(cur_iter) / max(ramp_iters, 1))
        device = w.device
        uniform = torch.ones(22, device=device)
        w = (1.0 - ramp) * uniform + ramp * w

    return w
```

### 4. `loss()` — replace body joint loss computation

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

        if self.group_normalise:
            self.upper_err_ema = (
                self.ema_momentum * self.upper_err_ema
                + (1.0 - self.ema_momentum) * per_joint_err[_UPPER_IDX]
            )
            self.lower_err_ema = (
                self.ema_momentum * self.lower_err_ema
                + (1.0 - self.ema_momentum) * per_joint_err[_LOWER_IDX]
            )
        else:
            self.joint_err_ema = (
                self.ema_momentum * self.joint_err_ema
                + (1.0 - self.ema_momentum) * per_joint_err
            )

        self._train_iter += 1

    # --- Adaptive per-joint weights (group-normalised + warmup ramp) ---
    w = self._get_adaptive_weights()              # (22,)

    # --- Weighted Smooth-L1 (manual, beta=0.05 matching baseline) ---
    # SoftWeightSmoothL1Loss.forward: (output, target, target_weight)
    # target_weight is a per-sample scalar mask, not per-joint weights.
    # Compute smooth-L1 manually to apply per-joint weights.
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

Note: `_UPPER_IDX` and `_LOWER_IDX` are the module-level constants added in step 1. They must be accessible inside `loss()` without any additional import.

Note: `_BODY = list(range(0, 22))` is already defined in the baseline's `loss()`. Keep only one definition.

### 5. Edge-case invariants the Builder must preserve

- `_train_mpjpe` and `_train_mpjpe_abs` attribute assignments must remain **unchanged**.
- `depth` and `uv` loss lines must remain **unchanged**.
- `predict()` must remain **unchanged**.
- When `per_joint_difficulty_weighting=False`, behaviour is **bit-identical** to baseline.
- When `group_normalise=True`, only `upper_err_ema` and `lower_err_ema` are registered (not `joint_err_ema`). When `group_normalise=False`, only `joint_err_ema` is registered. The `_get_adaptive_weights` method uses the correct buffer depending on `self.group_normalise`.
- All buffers (`upper_err_ema`, `lower_err_ema`, `_train_iter`) are saved/restored by `CheckpointHook`. SLURM preempt/resume works correctly.
- `_train_iter` must use `dtype=torch.long`.
- The `_UPPER_IDX` and `_LOWER_IDX` constants must be module-level (not inside the class or `__init__`) so they are available in `loss()` without `self.` prefix.

---

## `config.py` — Changes

In the `model.head` dict, add the following seven key-value pairs:

```python
per_joint_difficulty_weighting=True,
ema_alpha=1.0,
ema_momentum=0.99,
weight_norm='linear',
group_normalise=True,
ema_warmup_epochs=5,
```

(`weight_temperature` is not needed because `weight_norm='linear'` — the temperature parameter only applies in the `'softmax'` branch. Do not include `weight_temperature` in config to keep it minimal.)

All are literals (bool, float, float, str, bool, int) — fully MMEngine-compliant.

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
    ema_alpha=1.0,
    ema_momentum=0.99,
    weight_norm='linear',
    group_normalise=True,
    ema_warmup_epochs=5,
),
```

Everything else in `config.py` is identical to baseline.

---

## Warmup Ramp Behaviour

- At iter 0 (start of training): `ramp = 0.0` → `w = uniform = [1.0]*22` → exactly baseline.
- At iter `5 * 328 = 1640` (end of epoch 5): `ramp = 1.0` → `w = full difficulty-weighted`.
- In between: linear interpolation. At epoch 2 (iter ~656): `ramp ≈ 0.4`.
- Stage-2 starts from scratch (same pretrained backbone, epoch counter resets, `_train_iter` buffer resets to 0 at checkpoint deletion after stage-1 completion — but actually `_train_iter` is saved in the stage-1 checkpoint which is deleted before stage-2. Stage-2 training creates a new model object; `_train_iter` buffer starts at 0 again. This is correct — the warmup ramp restarts from 0 for stage-2.

Note on `ITERS_PER_EPOCH = 328`: this is an approximation for train100.txt. The exact value depends on the number of valid frames in train100.txt and batch size. If the actual value differs by ±50 steps, the warmup length shifts by ±15% — acceptable tolerance. The Builder must use 328 as the hardcoded constant.

---

## Group Normalisation Behaviour

With two EMA buffers:
- `upper_err_ema` (13 joints, indices 0–12): weights normalised so sum = 13.
- `lower_err_ema` (9 joints, indices 13–21): weights normalised so sum = 9.
- Combined `w` (22 joints): sum = 22 (preserves total gradient scale).
- Within each group, harder joints get higher weight; between groups, the budget is fixed at 13:9 ratio (same as if all joints had identical difficulty between groups).

This prevents the situation where lower body joints (typically harder) would crowd out upper body gradient if normalised globally. Each body region is guaranteed its proportional share of the gradient budget.

---

## Expected Behaviour

- Stage-1 target: `composite_val < 328` at epoch 20, `mpjpe_body_val < 185`.
- The warmup ramp produces a cleaner loss curve with lower variance in the first few epochs.
- Group normalisation produces a more balanced distribution of weights than global normalisation, which may help if upper-body joints also have high difficulty variation.
- `composite_val` improvement expected to be similar to design002 (alpha=1.0) but with more stable training dynamics.

---

## Summary of Changes

| File | Change |
|---|---|
| `pose3d_transformer_head.py` | Add module-level `_UPPER_IDX`/`_LOWER_IDX` constants; add 7 new `__init__` params; register group EMA buffers + `_train_iter`; add `_get_adaptive_weights()` with group-normalise + warmup branch; replace joint loss block |
| `config.py` | Add 6 literal kwargs to `model.head` dict |

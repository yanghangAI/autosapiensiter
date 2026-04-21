**Design Description:** EMA per-joint difficulty weighting (alpha=1.0, softmax temperature normalisation T=1.0) — full-proportional focusing with stable softmax normalisation, the primary bet.

**Starting Point:** `baseline/`

---

## Overview

Extend design001's EMA per-joint difficulty weighting to **alpha=1.0** (full proportional focusing) and switch the normalisation from linear-power to **temperature-scaled softmax**.

## Algorithm

The core algorithm is per-joint EMA difficulty weighting with softmax normalisation:

1. After each forward pass, compute per-joint batch-mean MPJPE (in mm) for the 22 body joints.
2. Update an EMA buffer: `ema[j] ← β * ema[j] + (1−β) * batch_err[j]` with `β=0.99`.
3. Derive per-joint weights via softmax: `w = 22 * softmax(ema / T)` with temperature `T=1.0`.
4. Apply weights to per-joint residuals before smooth-L1: `loss = mean(smooth_l1(pred−gt) * w)`.
5. With `alpha=1.0` and softmax normalisation: harder joints receive proportionally more gradient; softmax ensures all weights are positive and the distribution is smooth. Total gradient magnitude is preserved (weights sum to 22). Softmax normalisation is strictly positive, smooth, and avoids the `1e-6` epsilon guards needed by linear normalisation. At T=1.0 and typical EMA values in the 50–400 mm range, softmax produces well-calibrated weights without concentrating excessively on a single joint.

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
weight_norm: str = 'linear',
weight_temperature: float = 1.0,
```

Store them as instance attributes immediately after `super().__init__`:

```python
self.per_joint_difficulty_weighting = per_joint_difficulty_weighting
self.ema_alpha = ema_alpha
self.ema_momentum = ema_momentum
self.weight_norm = weight_norm
self.weight_temperature = weight_temperature
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

    Supports two normalisation modes controlled by self.weight_norm:
      - 'linear': power-law proportional, w = (ema / mean(ema))^alpha,
                  re-normalised to sum=22.
      - 'softmax': temperature-scaled softmax, w = 22 * softmax(ema / T),
                   naturally sums to 22 after multiplication by 22.

    Returns:
        Tensor of shape (22,), dtype float32, on the same device as
        self.joint_err_ema. Requires per_joint_difficulty_weighting=True.
    """
    import torch.nn.functional as F
    ema = self.joint_err_ema.detach()           # (22,) — no gradient

    if self.weight_norm == 'softmax':
        raw = ema / self.weight_temperature     # (22,) scale by temperature
        w = 22.0 * F.softmax(raw, dim=0)       # (22,) sums to 22
    else:
        # Linear proportional (fallback, same as design001)
        normalised = ema / (ema.mean() + 1e-6)
        raw_w = normalised ** self.ema_alpha
        w = 22.0 * raw_w / (raw_w.sum() + 1e-6)

    return w
```

Note: `import torch.nn.functional as F` inside the method is acceptable since it is a standard library import at runtime (not a config-level import statement). Alternatively, add `import torch.nn.functional as F` at the top of the file alongside the other `import torch` statements.

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

    # --- Adaptive per-joint weights (softmax normalisation for this design) ---
    w = self._get_adaptive_weights()              # (22,)

    # --- Weighted Smooth-L1 (manual, beta=0.05 matching baseline) ---
    # SoftWeightSmoothL1Loss.forward signature: (output, target, target_weight)
    # where target_weight is a per-sample scalar mask — does not support
    # per-joint tensor weights. Compute smooth-L1 manually.
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

Note: `_BODY = list(range(0, 22))` is already defined in the baseline's `loss()` block. Keep only one definition.

### 4. Edge-case invariants the Builder must preserve

- The `_train_mpjpe` and `_train_mpjpe_abs` attribute assignments in `loss()` must remain **unchanged**.
- The `depth` and `uv` loss lines must remain **unchanged**.
- The `predict()` method must remain **unchanged**.
- When `per_joint_difficulty_weighting=False`, behaviour is **bit-identical** to baseline.
- Buffer `joint_err_ema` is saved/restored by `CheckpointHook`. SLURM preempt/resume works correctly without EMA cold-start.
- `_train_iter` must use `dtype=torch.long`.
- The `import torch.nn.functional as F` must be available when `_get_adaptive_weights` is called. Either add it at the module top-level alongside `import torch`, or keep it as a local import inside the method — both are correct.

---

## `config.py` — Changes

In the `model.head` dict, add the following five key-value pairs:

```python
per_joint_difficulty_weighting=True,
ema_alpha=1.0,
ema_momentum=0.99,
weight_norm='softmax',
weight_temperature=1.0,
```

All are literals (bool, float, float, str, float) — fully MMEngine-compliant.

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
    weight_norm='softmax',
    weight_temperature=1.0,
),
```

Everything else in `config.py` is identical to baseline.

---

## Softmax Normalisation Behaviour

With `T=1.0` and typical EMA values in the range 50–400 mm across 22 joints:

- `softmax(ema / 1.0)` uses the raw mm values. Since the range is ~350 mm, the softmax is not overly concentrated: the hardest joint (say 400 mm) gets weight ≈ 22 × softmax_max ≈ a few units above 1.0, while the easiest (say 50 mm) gets weight ≈ near-zero.
- In early training when all joints have similar EMA ≈ 1.0 (initialisation), `softmax([1.0]*22) = [1/22]*22`, so `w = [1.0]*22` — exactly the baseline uniform weights.
- Compared to design001's linear normalisation, softmax is smoother and cannot produce negative or zero weights for any joint. It naturally saturates concentration when differences are large.

At alpha=1.0 with linear normalisation, a joint with 2× mean difficulty would get weight 2.0×. With softmax at T=1.0, the concentration is EMA-scale-dependent — at typical mm values (mean ~200 mm, std ~80 mm), the maximum weight is roughly 3–5× and minimum roughly 0.1×. This is a stronger but well-bounded focusing than design001.

---

## Expected Behaviour

- Early training (first ~100 steps): uniform weights ≈ baseline.
- After EMA convergence: harder distal joints receive significantly higher weights (2–4×); easy spine joints receive lower weights.
- `alpha=1.0` with softmax: the primary "full-strength" design. Expected to outperform design001.
- `composite_val` target: < 330 at stage-1. `mpjpe_body_val` target: < 185 at stage-1.

---

## Summary of Changes

| File | Change |
|---|---|
| `pose3d_transformer_head.py` | Add 5 new `__init__` params; register 2 buffers; add `_get_adaptive_weights()` with softmax branch; replace joint loss block |
| `config.py` | Add 5 literal kwargs to `model.head` dict |

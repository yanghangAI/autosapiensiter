# Design 002 — Body-only decoder with linear hand recovery + auxiliary loss (weight 0.1)

**Design Description:** 22-query body decoder; recover hand predictions (B,48,3) via `Linear(22*hidden_dim, 48*3)`; auxiliary hand loss weight 0.1.

**Starting Point:** `baseline/`

---

## Overview

Same 22-query decoder algorithm as Design 001. Instead of zero-padding hand joints, a single linear layer projects the flattened body query features `(B, 22*hidden_dim)` to hand predictions `(B, 48, 3)`. A small auxiliary hand loss (weight 0.1, same `SoftWeightSmoothL1Loss` as body joints) keeps the projection anchored in pose-space and provides auxiliary gradient to the body decoder. The composite metric is unaffected (hand joints not evaluated), but the auxiliary gradient regularises the body decoder algorithm.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

**Constructor (`__init__`):**

New kwargs after `dropout`:
- `num_body_queries: int = 22`
- `hand_aux_loss_weight: float = 0.1`

Store both: `self.num_body_queries = num_body_queries`, `self.hand_aux_loss_weight = hand_aux_loss_weight`.

Change joint query embedding:
```python
self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)
```

Add hand projection layer after `self.uv_out`:
```python
self.hand_proj = nn.Linear(num_body_queries * hidden_dim, (num_joints - num_body_queries) * 3)
```
Where `num_joints - num_body_queries = 70 - 22 = 48`. This is `Linear(22 * 256, 48 * 3)` = `Linear(5632, 144)`.

**Weight initialisation** — add to `_init_head_weights`:
```python
nn.init.trunc_normal_(self.hand_proj.weight, std=0.02)
nn.init.zeros_(self.hand_proj.bias)
```

**`forward()` method:**

After decoding:
```python
decoded = self.decoder_layer(queries, spatial)  # (B, 22, hidden_dim)
body_joints = self.joints_out(decoded)           # (B, 22, 3)

# Hand recovery via linear projection
body_flat = decoded.reshape(B, self.num_body_queries * self.hidden_dim)  # (B, 22*256)
num_hand = self.num_joints - self.num_body_queries  # 48
hand_joints = self.hand_proj(body_flat).reshape(B, num_hand, 3)          # (B, 48, 3)

joints = torch.cat([body_joints, hand_joints], dim=1)  # (B, 70, 3)
```

`pelvis_token = decoded[:, 0, :]` — unchanged.

Return dict unchanged: `{'joints': joints, 'pelvis_depth': pelvis_depth, 'pelvis_uv': pelvis_uv}` with `joints.shape == (B, 70, 3)`.

**`loss()` method:**

After existing body/depth/UV losses, add:
```python
# Auxiliary hand loss — does not contribute to composite metric
_HAND = list(range(22, 70))
losses['loss/hand_aux/train'] = self.hand_aux_loss_weight * self.loss_joints_module(
    pred['joints'][:, _HAND], gt_joints[:, _HAND])
```

The `loss_joints_module` (SoftWeightSmoothL1Loss) is reused — no new loss module needed.

`self._train_mpjpe` and `self._train_mpjpe_abs` computations are unchanged.

**Full constructor signature after change:**
```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_body_queries: int = 22,
    num_heads: int = 8,
    dropout: float = 0.1,
    hand_aux_loss_weight: float = 0.1,
    loss_joints: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                   beta=0.05, loss_weight=1.0),
    loss_depth: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                                  beta=0.05, loss_weight=1.0),
    loss_uv: ConfigType = dict(type='SoftWeightSmoothL1Loss',
                               beta=0.05, loss_weight=1.0),
    loss_weight_depth: float = 1.0,
    loss_weight_uv: float = 1.0,
    init_cfg: OptConfigType = None,
):
```

### 2. `config.py`

In the `model.head` dict, add `num_body_queries=22` and `hand_aux_loss_weight=0.1` as literal values:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_body_queries=22,
    num_heads=8,
    dropout=0.1,
    hand_aux_loss_weight=0.1,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                     loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                    loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05,
                 loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config values (optimizer, LR schedule, data pipeline, hooks, backbone) are **identical to baseline**.

### 3. `pelvis_utils.py`

No changes.

---

## Parameter Budget

- `hand_proj`: `Linear(5632, 144)` = 5632 × 144 + 144 = 810,432 + 144 = **810,576 parameters**.
- No extra attention computation vs. Design 001.
- Net decoder compute still lower than baseline (22 queries vs. 70).

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. `self.num_joints = 70` must remain so `predict()` produces correct shape.
3. Output `joints` shape must be `(B, 70, 3)` from `forward()`.
4. `hand_proj` projects from `22 * hidden_dim = 5632` → `48 * 3 = 144`. These values must be computed dynamically as `num_body_queries * hidden_dim` and `(num_joints - num_body_queries) * 3` in `__init__` to be robust to any kwarg.
5. Auxiliary hand loss uses the **same** `self.loss_joints_module` instance already built — do not create a new loss module.
6. `_HAND = list(range(22, 70))` — the auxiliary loss covers exactly indices 22–69.
7. `_BODY = list(range(0, 22))` for body joint loss — unchanged.
8. MMEngine config: `num_body_queries=22` (int literal), `hand_aux_loss_weight=0.1` (float literal). No imports required. Compliant.
9. Pelvis token is `decoded[:, 0, :]` — unchanged.
10. `_DecoderLayer` is unchanged.
11. Backbone, data preprocessor, metric, transforms are invariant.
12. Seed `2026`, batch size `4`, accumulation `8` — do not change.

---

## Expected Behavior After Change

- Body decoder operates identically to Design 001 (22 queries, clean self-attention).
- Hand joints are predicted via linear projection of body features, providing structurally consistent (but not high-fidelity) hand output.
- `loss/hand_aux/train` appears in the training log at roughly 0.1 × (hand joint SmoothL1 loss).
- Auxiliary gradient from hand loss flows back through `hand_proj` into `decoded` (body query features), providing additional regularisation signal to the body decoder.
- Body MPJPE expected to decrease 10–18 mm vs. baseline; possibly better than Design 001 due to aux regularisation.
- `composite_val` target: < 158.
- All downstream metric code expecting shape `(B, 70, 3)` receives correct shape.

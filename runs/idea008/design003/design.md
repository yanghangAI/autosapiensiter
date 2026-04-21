# Design 003 — Body-only decoder with 2-layer MLP hand recovery + auxiliary loss (weight 0.3)

**Design Description:** 22-query body decoder; recover hand predictions via 2-layer MLP `Linear(22*hidden_dim, hidden_dim) → GELU → Linear(hidden_dim, 48*3)`; auxiliary hand loss weight 0.3.

**Starting Point:** `baseline/`

---

## Overview

Same 22-query body-only decoder algorithm as Designs 001 and 002. Hand recovery is upgraded from a single linear layer to a 2-layer bottleneck MLP with GELU activation — a nonlinear algorithm for mapping body features to hand predictions. The auxiliary hand loss weight is increased from 0.1 to 0.3, providing a stronger regularisation signal to the body decoder. The motivation is that a single linear cannot capture the nonlinear relationship between body pose and hand pose; the MLP bottleneck models this better and provides richer gradient.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

**Constructor (`__init__`):**

New kwargs after `dropout`:
- `num_body_queries: int = 22`
- `hand_aux_loss_weight: float = 0.3`

Store both: `self.num_body_queries = num_body_queries`, `self.hand_aux_loss_weight = hand_aux_loss_weight`.

Change joint query embedding:
```python
self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)
```

Add 2-layer MLP for hand recovery after `self.uv_out`:
```python
num_hand = num_joints - num_body_queries  # 48
self.hand_proj = nn.Sequential(
    nn.Linear(num_body_queries * hidden_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, num_hand * 3),
)
```
Concretely: `Linear(5632, 256) → GELU → Linear(256, 144)`.

**Weight initialisation** — add to `_init_head_weights`:
```python
for layer in self.hand_proj:
    if isinstance(layer, nn.Linear):
        nn.init.trunc_normal_(layer.weight, std=0.02)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
```

**`forward()` method:**

After decoding:
```python
decoded = self.decoder_layer(queries, spatial)  # (B, 22, hidden_dim)
body_joints = self.joints_out(decoded)           # (B, 22, 3)

# Hand recovery via 2-layer MLP
body_flat = decoded.reshape(B, self.num_body_queries * self.hidden_dim)  # (B, 5632)
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
    hand_aux_loss_weight: float = 0.3,
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

In the `model.head` dict, add `num_body_queries=22` and `hand_aux_loss_weight=0.3` as literal values:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_body_queries=22,
    num_heads=8,
    dropout=0.1,
    hand_aux_loss_weight=0.3,
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

- `hand_proj` layer 1: `Linear(5632, 256)` = 5632 × 256 + 256 = 1,441,792 + 256 = 1,442,048 parameters.
- `hand_proj` layer 2: `Linear(256, 144)` = 256 × 144 + 144 = 36,864 + 144 = 37,008 parameters.
- Total hand MLP: **1,479,056 parameters** (~1.47M). Within 1080 Ti budget.
- No extra attention computation vs. Design 001 / Design 002.
- Net decoder compute still lower than baseline.

---

## Constraints and Invariants to Preserve

1. `persistent_workers=False` in both dataloaders — do not change.
2. `self.num_joints = 70` must remain so `predict()` produces correct shape.
3. Output `joints` shape must be `(B, 70, 3)` from `forward()`.
4. The MLP input dimension is `num_body_queries * hidden_dim = 22 * 256 = 5632`; output is `(num_joints - num_body_queries) * 3 = 48 * 3 = 144`. Compute both dynamically from kwargs in `__init__` so the module is not hard-coded to specific values.
5. The bottleneck dimension for the MLP intermediate layer is `hidden_dim` (256) — not a new hyperparameter, reuses existing `hidden_dim`.
6. Activation is `nn.GELU()` — not ReLU.
7. Auxiliary hand loss uses the **same** `self.loss_joints_module` instance already built — do not create a new loss module.
8. `_HAND = list(range(22, 70))` — the auxiliary loss covers exactly indices 22–69.
9. `_BODY = list(range(0, 22))` for body joint loss — unchanged.
10. MMEngine config: `num_body_queries=22` (int literal), `hand_aux_loss_weight=0.3` (float literal). No imports required. Compliant.
11. Pelvis token is `decoded[:, 0, :]` — unchanged.
12. `_DecoderLayer` is unchanged.
13. Backbone, data preprocessor, metric, transforms are invariant.
14. Seed `2026`, batch size `4`, accumulation `8` — do not change.

---

## Expected Behavior After Change

- Body decoder operates identically to Designs 001 and 002 (22 queries, clean self-attention).
- Hand joints predicted via 2-layer MLP from body features; richer nonlinear relationship modelled vs. Design 002's single linear.
- `loss/hand_aux/train` appears in training log at roughly 0.3 × (hand joint SmoothL1 loss) — 3× the auxiliary weight of Design 002.
- Stronger auxiliary gradient from hand loss flows back through MLP layers into `decoded`, providing stronger regularisation to the body decoder.
- Body MPJPE expected to show highest potential improvement among the three designs if the MLP auxiliary gradient provides useful regularisation.
- `composite_val` target: < 158.
- All downstream metric code expecting shape `(B, 70, 3)` receives correct shape.

---

## Risk Notes

- The stronger hand aux loss weight (0.3) could compete with the body joint loss (effective weight 1.0) if hand GT is noisy. The `SoftWeightSmoothL1Loss` with `beta=0.05` provides robustness to outliers. Monitor `loss/hand_aux/train` at early epochs; if it dominates the total loss magnitude unexpectedly, reduce weight to 0.2 in a follow-up.
- The MLP adds ~1.47M parameters that do not contribute to the composite metric but consume memory for activations during backprop. At batch size 4 and hidden_dim 256 this is negligible on 1080 Ti (body flat tensor: 4 × 5632 × 4 bytes ≈ 88 KB).

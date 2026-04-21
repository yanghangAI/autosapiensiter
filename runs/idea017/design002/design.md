# Design 002 — 22-query body decoder, 2 layers, intermediate body supervision (weight 0.4)

**Design Description:** 22-query 2-layer body decoder with auxiliary body joint loss at layer-1 output (weight 0.4); linear hand recovery weight 0.1; forces intermediate pose quality.

**Starting Point:** `baseline/`

---

## Overview

Extends Design 001 by adding intermediate body joint supervision on the output of decoder layer 1 (the first of two layers). The core algorithm is identical to Design 001 — iterative query refinement via 2 sequential `_DecoderLayer` passes over 22 body-only queries — but with an additional auxiliary body joint loss computed from the layer-1 intermediate representation. The intermediate supervision loss uses the same `joints_out` projection applied to the layer-1 output, weighted at 0.4 vs. the final body loss at 1.0. This mirrors idea001/design002's treatment but applied to 22 body-only queries.

Rationale: forcing decoder layer 1 to produce a usable pose estimate prevents gradient vanishing in early training, gives layer 2 a meaningful starting point to refine from, and historically improved stage-1 convergence in idea001 (design002 composite_val 338.78 at stage-1 — competitive with design001's stage-1). Over the already-clean 22-query self-attention, this supervision pressure should further accelerate convergence.

All architectural choices from Design 001 are preserved; the only addition is `aux_body_loss_weight=0.4`.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

Identical to Design 001 except `aux_body_loss_weight` defaults to 0.4 (and the config passes 0.4). All code paths that reference `self.aux_body_loss_weight` are already written in the Design 001 spec — the only runtime difference is that the `if self.aux_body_loss_weight > 0.0:` branch is now entered during training.

**Constructor signature — full updated version:**

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_body_queries: int = 22,
    num_decoder_layers: int = 2,
    num_heads: int = 8,
    dropout: float = 0.1,
    hand_aux_loss_weight: float = 0.1,
    aux_body_loss_weight: float = 0.0,
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

**`__init__` body — identical to Design 001:**

Store attributes:
```python
self.num_body_queries = num_body_queries
self.num_decoder_layers = num_decoder_layers
self.hand_aux_loss_weight = hand_aux_loss_weight
self.aux_body_loss_weight = aux_body_loss_weight  # 0.4 for Design B
```

Replace joint query embedding:
```python
self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)
```

Replace single decoder with ModuleList:
```python
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
```

Add hand projection:
```python
num_hand_joints = num_joints - num_body_queries  # 48
self.hand_proj = nn.Linear(num_body_queries * hidden_dim, num_hand_joints * 3)
# Linear(5632, 144)
```

**`_init_head_weights()` — identical to Design 001:**
```python
nn.init.trunc_normal_(self.hand_proj.weight, std=0.02)
nn.init.zeros_(self.hand_proj.bias)
```

**`forward()` — identical to Design 001:**
```python
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 22, hidden_dim)

intermediate_outputs = []
for layer in self.decoder_layers:
    queries = layer(queries, spatial)
    intermediate_outputs.append(queries)

body_joints = self.joints_out(queries)  # (B, 22, 3)

body_flat = queries.reshape(B, self.num_body_queries * self.hidden_dim)  # (B, 5632)
num_hand = self.num_joints - self.num_body_queries  # 48
hand_joints = self.hand_proj(body_flat).reshape(B, num_hand, 3)  # (B, 48, 3)

joints = torch.cat([body_joints, hand_joints], dim=1)  # (B, 70, 3)

pelvis_token = queries[:, 0, :]
pelvis_depth = self.depth_out(pelvis_token)
pelvis_uv = self.uv_out(pelvis_token)

self._intermediate_outputs = intermediate_outputs
```

**`loss()` changes — intermediate branch now active:**

After the existing three main losses, add:

```python
# Auxiliary intermediate body joint loss — active when aux_body_loss_weight > 0.0
if self.aux_body_loss_weight > 0.0:
    _BODY = list(range(0, 22))
    for i, inter_decoded in enumerate(self._intermediate_outputs[:-1]):
        # For 2-layer decoder: i=0 → layer-1 output (the only intermediate)
        inter_body = self.joints_out(inter_decoded)  # (B, 22, 3)
        losses[f'loss/joints_aux_{i}/train'] = (
            self.aux_body_loss_weight * self.loss_joints_module(
                inter_body[:, _BODY], gt_joints[:, _BODY]))

# Auxiliary hand loss
if self.hand_aux_loss_weight > 0.0:
    _HAND = list(range(22, 70))
    losses['loss/hand_aux/train'] = (
        self.hand_aux_loss_weight * self.loss_joints_module(
            pred['joints'][:, _HAND], gt_joints[:, _HAND]))
```

With `num_decoder_layers=2`, `self._intermediate_outputs[:-1]` has exactly 1 element (layer-1 output). The loss key emitted is `loss/joints_aux_0/train` at weight 0.4.

Gradient flow: from `loss/joints_aux_0/train` → `joints_out` (shared) → `inter_decoded` (layer-1 output) → layer-1 weights → `joint_queries` embedding → (via layer-1 cross-attn) → backbone features. This is the same gradient path as idea001/design002 and is known to train stably.

`self._train_mpjpe` and `self._train_mpjpe_abs` computations remain unchanged.

**`predict()` — no changes.**

---

### 2. `config.py`

Replace the `head=dict(...)` block:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_body_queries=22,
    num_decoder_layers=2,
    num_heads=8,
    dropout=0.1,
    hand_aux_loss_weight=0.1,
    aux_body_loss_weight=0.4,
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

Difference from Design 001 config: `aux_body_loss_weight=0.4` (was `0.0`).

All other config values identical to baseline (optimizer, LR schedule, data pipeline, hooks, backbone, seed, batch size).

### 3. `pelvis_utils.py`

No changes.

---

## Constraints and Invariants

1–12: All constraints from Design 001 apply unchanged.

Additionally:
13. `self._intermediate_outputs[:-1]` must be used (not `self._intermediate_outputs`) so the final layer output is NOT double-counted in the intermediate loss. With 2 layers, `[:-1]` gives exactly 1 element at index 0 (layer-1 output).
14. `joints_out` (Linear(256, 3)) is shared — the same weights produce both intermediate and final body joint predictions. This is intentional: it forces the intermediate representation to already be in a useful pose-prediction space.
15. The loss key `loss/joints_aux_0/train` must not collide with any existing baseline loss keys. The baseline only has `loss/joints/train`, `loss/depth/train`, `loss/uv/train`.
16. Auxiliary intermediate body loss weight = 0.4 (float literal in config). Final body loss weight = 1.0 (unchanged, implicit in `loss_joints_module`).

---

## Expected Behavior After Change

- Training losses logged: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/hand_aux/train`, `loss/joints_aux_0/train`.
- `loss/joints_aux_0/train` is 0.4 × SoftWeightSmoothL1Loss on layer-1 body joint predictions vs. GT.
- Layer-1 receives gradient from both the aux body loss (weight 0.4 path) and the final body loss (weight 1.0 path through layer-2's residual).
- Expected faster convergence vs. Design 001 in stage-1 (first 20 epochs), due to supervision pressure on layer-1 output.
- Expected stage-1 composite_val: competitive with or slightly better than Design 001; target < 325.
- Expected stage-2 composite_val target: < 215 (same as Design 001 — the intermediate supervision may or may not improve stage-2 final result vs. Design 001, as idea001's results showed design001 > design002 at stage-2).

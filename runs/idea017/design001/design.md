# Design 001 — 22-query body decoder, 2 layers, no intermediate supervision

**Design Description:** Replace single 70-query decoder with 2-layer `nn.ModuleList` over 22 body-only queries; recover hands via `Linear(22*256, 48*3)`; auxiliary hand loss weight 0.1; no intermediate body supervision.

**Starting Point:** `baseline/`

---

## Overview

Combine idea008/design002 (22-query body decoder + linear hand recovery) with idea001/design001 (2-layer stacked decoder, no aux loss). The core algorithm is iterative query refinement: each decoder layer refines 22 body-query embeddings by attending to the backbone spatial feature map, then residually updating the query representations. After 2 such algorithm passes, the final queries are projected to body joint coordinates via `joints_out` and to hand joint coordinates via the linear `hand_proj`. This is the minimal controlled combination: decoder depth increase without auxiliary supervision on intermediate layers.

Self-attention in both decoder layers spans only 22×22=484 elements (vs. 70×70=4,900 in baseline), eliminating hand-query contamination from all self-attention steps. Body query features refined over 2 clean decoder passes are used to compute body joints directly and hand joints via linear projection.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

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

**`__init__` body changes:**

Store new attributes:
```python
self.num_body_queries = num_body_queries
self.num_decoder_layers = num_decoder_layers
self.hand_aux_loss_weight = hand_aux_loss_weight
self.aux_body_loss_weight = aux_body_loss_weight  # 0.0 for Design A — kept for interface compat
```

Replace the joint query embedding (use `num_body_queries` not `num_joints`):
```python
self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)
```

Replace the single `self.decoder_layer` with a `nn.ModuleList`:
```python
# Remove: self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
```

Add hand projection layer after `self.uv_out`:
```python
num_hand_joints = num_joints - num_body_queries   # 70 - 22 = 48
self.hand_proj = nn.Linear(num_body_queries * hidden_dim, num_hand_joints * 3)
# Linear(22*256=5632, 48*3=144)
```

**`_init_head_weights()` changes:**

Append after the existing loop for `joints_out`, `depth_out`, `uv_out`:
```python
nn.init.trunc_normal_(self.hand_proj.weight, std=0.02)
nn.init.zeros_(self.hand_proj.bias)
```

**`forward()` changes:**

Replace:
```python
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)
decoded = self.decoder_layer(queries, spatial)
joints = self.joints_out(decoded)
```
With:
```python
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 22, hidden_dim)

intermediate_outputs = []
for layer in self.decoder_layers:
    queries = layer(queries, spatial)
    intermediate_outputs.append(queries)
# queries is now the final decoded state: (B, 22, hidden_dim)

body_joints = self.joints_out(queries)  # (B, 22, 3)

body_flat = queries.reshape(B, self.num_body_queries * self.hidden_dim)  # (B, 5632)
num_hand = self.num_joints - self.num_body_queries  # 48
hand_joints = self.hand_proj(body_flat).reshape(B, num_hand, 3)  # (B, 48, 3)

joints = torch.cat([body_joints, hand_joints], dim=1)  # (B, 70, 3)

pelvis_token = queries[:, 0, :]  # (B, hidden_dim) — still query index 0
pelvis_depth = self.depth_out(pelvis_token)  # (B, 1)
pelvis_uv = self.uv_out(pelvis_token)        # (B, 2)
```

Store `intermediate_outputs` on `self` so `loss()` can access it if `aux_body_loss_weight > 0`:
```python
self._intermediate_outputs = intermediate_outputs
```

Return dict is unchanged: `{'joints': joints, 'pelvis_depth': pelvis_depth, 'pelvis_uv': pelvis_uv}`.

**`loss()` changes:**

After the existing `losses['loss/joints/train']`, `losses['loss/depth/train']`, `losses['loss/uv/train']` assignments, add:

```python
# Auxiliary intermediate body joint loss (Design A: aux_body_loss_weight=0.0 → skipped)
if self.aux_body_loss_weight > 0.0:
    _BODY = list(range(0, 22))
    for i, inter_decoded in enumerate(self._intermediate_outputs[:-1]):
        inter_body = self.joints_out(inter_decoded)  # (B, 22, 3)
        losses[f'loss/joints_aux_{i}/train'] = (
            self.aux_body_loss_weight * self.loss_joints_module(
                inter_body[:, _BODY], gt_joints[:, _BODY]))

# Auxiliary hand loss (weight 0.1 for Design A)
if self.hand_aux_loss_weight > 0.0:
    _HAND = list(range(22, 70))
    losses['loss/hand_aux/train'] = (
        self.hand_aux_loss_weight * self.loss_joints_module(
            pred['joints'][:, _HAND], gt_joints[:, _HAND]))
```

Note: `_BODY = list(range(0, 22))` is already defined at the top of `loss()` in the baseline. The `joints_out` layer is shared across all decoder layer outputs (same Linear(256, 3)). The `self._intermediate_outputs` list holds the output after each decoder layer; `[:-1]` skips the final layer (which drives the main body loss).

`self._train_mpjpe` and `self._train_mpjpe_abs` computations remain unchanged — they operate on `pred['joints']` and `gt_joints` with shape `(B, 70, 3)` and `(B, 22, 3)` body-sliced.

**`predict()` — no changes.** Reads `pred['joints']` which is already `(B, 70, 3)`.

---

### 2. `config.py`

Replace the `head=dict(...)` block inside `model = dict(...)`:

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
    aux_body_loss_weight=0.0,
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

All other config values are identical to baseline:
- `optimizer`: AdamW, `lr=1e-4`, `betas=(0.9, 0.999)`, `weight_decay=0.03`
- `backbone lr_mult=0.1` (effective lr=1e-5), `clip_grad max_norm=1.0`
- `accumulative_counts=8`, batch size=4, effective batch=32
- LR schedule: LinearLR warmup 3 epochs (start_factor=0.333), then CosineAnnealingLR to epoch 20/10
- `convert_to_iter_based=True` on all schedulers
- Seed: 2026, `num_workers=2`, `persistent_workers=False`

### 3. `pelvis_utils.py`

No changes.

---

## Parameter Budget

- 2 decoder layers × ~3.2M params each = +3.2M params vs. baseline (1 extra layer)
- `hand_proj`: `Linear(5632, 144)` = 810,576 parameters
- `joint_queries`: 22×256 = 5,632 (vs. baseline 70×256 = 17,920 → saves 12,288)
- Net: approximately +3.2M vs. baseline decoder capacity; well within 2080 Ti VRAM

---

## Constraints and Invariants

1. `self.num_joints = 70` must remain — `predict()` uses it for `keypoint_scores` shape.
2. Output `joints` from `forward()` must be `(B, 70, 3)` — enforced by `cat([body_joints, hand_joints], dim=1)`.
3. `pelvis_token = queries[:, 0, :]` — query index 0 is the pelvis token in both baseline and this design.
4. `_DecoderLayer` class body is unchanged — only the number of instances changes.
5. `persistent_workers=False` — do not change.
6. No Python `import` statements in `config.py`. `num_body_queries=22`, `num_decoder_layers=2`, `hand_aux_loss_weight=0.1`, `aux_body_loss_weight=0.0` are all int/float literals.
7. `self._intermediate_outputs` is set during `forward()` and consumed in `loss()`. Since MMEngine calls `loss()` immediately after or as part of the same forward+loss step during training, this is safe. (For Design A, `aux_body_loss_weight=0.0` so the intermediate outputs list is populated but never consumed.)
8. Auxiliary hand loss uses the same `self.loss_joints_module` — no new loss module.
9. `joints_out` is a single `Linear(hidden_dim, 3)` shared for both final and intermediate body joint predictions.
10. `_BODY = list(range(0, 22))` for the primary body loss — unchanged from baseline.
11. `_HAND = list(range(22, 70))` for the auxiliary hand loss — 48 joints.
12. Backbone, data preprocessor, metric, transforms, `train.py`, infra hooks are invariant.

---

## Expected Behavior After Change

- `joint_queries.weight` shape: `(22, 256)` — 22 body-only learnable query vectors.
- `decoder_layers`: `nn.ModuleList` of 2 independent `_DecoderLayer(256, 8, 0.1)` instances.
- `hand_proj`: `Linear(5632, 144)`.
- Forward pass: queries → layer0 → layer1 → `joints_out` (body) + `hand_proj` (hand) → concatenate → `(B, 70, 3)`.
- Training losses logged: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/hand_aux/train`.
- `loss/joints_aux_*/train` keys do NOT appear (aux_body_loss_weight=0.0).
- Self-attention in each decoder layer: 22×22 attention maps (vs. 70×70 in baseline).
- Cross-attention in each decoder layer: 22×960 (vs. 70×960 in baseline), where 960 = H'×W' of backbone feature map.
- Expected stage-1 composite_val: below baseline (target < 325); stage-2 composite_val target < 215.

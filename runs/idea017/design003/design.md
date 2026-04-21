# Design 003 — 22-query body decoder, 3 layers, intermediate supervision at layers 1 and 2

**Design Description:** 22-query 3-layer body decoder; auxiliary body joint losses at layer-1 (weight 0.4) and layer-2 (weight 0.6) outputs; final body loss at layer-3 (weight 1.0); linear hand recovery weight 0.1.

**Starting Point:** `baseline/`

---

## Overview

Pushes decoder depth to 3 layers using the same iterative query-refinement algorithm as Designs 001/002, enabled by the VRAM savings from 22-query self-attention. With 22 queries, 3 decoder layers use approximately 3×(22/70)=94% of the cross-attention VRAM and 3×(22²/70²)=29% of the self-attention VRAM relative to the single-layer 70-query baseline, making 3 layers feasible within the 2080 Ti 8 GB budget.

Intermediate body joint losses are added at both layer-1 (weight 0.4) and layer-2 (weight 0.6) outputs. Increasing weights (0.4 → 0.6 → 1.0) provide a curriculum: the first intermediate layer is supervised softly (preserving flexibility), the second more strongly (anchoring the representation before the final refinement), and the final layer carries the full signal. This matches the standard practice in multi-layer pose decoders.

Architecture summary:
- Queries: `nn.Embedding(22, 256)`
- Decoder: `nn.ModuleList` of 3 × `_DecoderLayer(256, 8, 0.1)`
- Losses: `loss/joints_aux_0/train` (w=0.4), `loss/joints_aux_1/train` (w=0.6), `loss/joints/train` (w=1.0), `loss/depth/train`, `loss/uv/train`, `loss/hand_aux/train` (w=0.1)

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
    num_decoder_layers: int = 3,
    num_heads: int = 8,
    dropout: float = 0.1,
    hand_aux_loss_weight: float = 0.1,
    aux_body_loss_weight: float = 0.4,
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

**`__init__` body — same pattern as Designs 001/002:**

```python
self.num_body_queries = num_body_queries
self.num_decoder_layers = num_decoder_layers
self.hand_aux_loss_weight = hand_aux_loss_weight
self.aux_body_loss_weight = aux_body_loss_weight  # base weight for intermediate losses

self.joint_queries = nn.Embedding(num_body_queries, hidden_dim)

self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)   # 3 layers
])

num_hand_joints = num_joints - num_body_queries  # 48
self.hand_proj = nn.Linear(num_body_queries * hidden_dim, num_hand_joints * 3)
# Linear(5632, 144)
```

**`_init_head_weights()` — identical to Designs 001/002:**
```python
nn.init.trunc_normal_(self.hand_proj.weight, std=0.02)
nn.init.zeros_(self.hand_proj.bias)
```

**`forward()` — identical to Designs 001/002:**
```python
queries = self.joint_queries.weight.unsqueeze(0).expand(B, -1, -1)  # (B, 22, hidden_dim)

intermediate_outputs = []
for layer in self.decoder_layers:
    queries = layer(queries, spatial)
    intermediate_outputs.append(queries)
# intermediate_outputs: [layer1_out, layer2_out, layer3_out]

body_joints = self.joints_out(queries)  # (B, 22, 3) — uses layer3_out

body_flat = queries.reshape(B, self.num_body_queries * self.hidden_dim)  # (B, 5632)
num_hand = self.num_joints - self.num_body_queries  # 48
hand_joints = self.hand_proj(body_flat).reshape(B, num_hand, 3)  # (B, 48, 3)

joints = torch.cat([body_joints, hand_joints], dim=1)  # (B, 70, 3)

pelvis_token = queries[:, 0, :]
pelvis_depth = self.depth_out(pelvis_token)
pelvis_uv = self.uv_out(pelvis_token)

self._intermediate_outputs = intermediate_outputs
```

**`loss()` changes:**

After the existing three main losses, add the intermediate supervision block with escalating weights:

```python
# Auxiliary intermediate body joint losses (active when aux_body_loss_weight > 0.0)
if self.aux_body_loss_weight > 0.0:
    _BODY = list(range(0, 22))
    # For 3-layer decoder: intermediate_outputs[:-1] = [layer1_out, layer2_out]
    # Weights: layer1 → aux_body_loss_weight (0.4), layer2 → aux_body_loss_weight * 1.5 (0.6)
    intermediate_weights = [self.aux_body_loss_weight, self.aux_body_loss_weight * 1.5]
    for i, inter_decoded in enumerate(self._intermediate_outputs[:-1]):
        inter_body = self.joints_out(inter_decoded)  # (B, 22, 3)
        losses[f'loss/joints_aux_{i}/train'] = (
            intermediate_weights[i] * self.loss_joints_module(
                inter_body[:, _BODY], gt_joints[:, _BODY]))

# Auxiliary hand loss
if self.hand_aux_loss_weight > 0.0:
    _HAND = list(range(22, 70))
    losses['loss/hand_aux/train'] = (
        self.hand_aux_loss_weight * self.loss_joints_module(
            pred['joints'][:, _HAND], gt_joints[:, _HAND]))
```

Concretely with `aux_body_loss_weight=0.4`:
- `loss/joints_aux_0/train`: weight 0.4 (layer-1 output)
- `loss/joints_aux_1/train`: weight 0.6 (layer-2 output, = 0.4 × 1.5)
- `loss/joints/train`: weight 1.0 (layer-3 output, unchanged final body loss)
- `loss/hand_aux/train`: weight 0.1

The `intermediate_weights` list has exactly `num_decoder_layers - 1 = 2` elements, matching `self._intermediate_outputs[:-1]`. The Builder must verify this alignment: `len(self._intermediate_outputs[:-1]) == len(intermediate_weights)` holds when `num_decoder_layers=3`.

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
    num_decoder_layers=3,
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

Key differences from Design 002: `num_decoder_layers=3` (was 2).

All other config values identical to baseline.

### 3. `pelvis_utils.py`

No changes.

---

## Parameter Budget

- 3 decoder layers × ~3.2M params each = +6.4M params vs. baseline single-layer decoder
- `hand_proj`: `Linear(5632, 144)` = 810,576 parameters
- `joint_queries`: 22×256 = 5,632 (saves 12,288 vs. baseline)
- FFN compute: 3 × 22 = 66 FFN applications vs. baseline 1 × 70 = 70 → still slightly cheaper
- Cross-attention: 3 × (22×960) vs. 1 × (70×960) → 63,360 vs. 67,200 → comparable
- Self-attention: 3 × (22×22) vs. 1 × (70×70) → 1,452 vs. 4,900 → 70% cheaper despite 3 layers

VRAM estimate: net cross/self-attention burden is slightly less than baseline despite 3× decoder depth. Memory risk: low.

---

## Constraints and Invariants

1–12: All constraints from Design 001 apply unchanged.

Additionally:
13. `self._intermediate_outputs[:-1]` contains exactly `num_decoder_layers - 1 = 2` elements for Design 003. The `intermediate_weights` list must have exactly 2 elements: `[0.4, 0.6]`. Builder must not hard-code these as `[0.4, 0.6]` but compute them as `[self.aux_body_loss_weight, self.aux_body_loss_weight * 1.5]` to keep the logic parameterized.
14. The weight escalation factor 1.5 is implicit in the `aux_body_loss_weight * 1.5` computation. If a future design changes `aux_body_loss_weight`, the escalation scales automatically. The absolute values at `aux_body_loss_weight=0.4` are: layer-1 → 0.4, layer-2 → 0.6, layer-3 → 1.0 (final).
15. For `num_decoder_layers=2` (Designs 001/002), the same code `intermediate_weights = [self.aux_body_loss_weight, self.aux_body_loss_weight * 1.5]` would generate 2 weights but only 1 element in `[:-1]` — a length mismatch. The Builder must either: (a) make the weights list length equal to `len(self._intermediate_outputs[:-1])` dynamically, or (b) hard-code separate loss() logic for Design 003. **Preferred approach**: make `intermediate_weights` a list of length `len(self._intermediate_outputs[:-1])`, computed via a linspace or enumeration pattern. For Design 003 specifically, the values are [0.4, 0.6]. For Design 002, the value is [0.4]. The cleanest implementation:

```python
n_inter = len(self._intermediate_outputs) - 1  # num intermediate layers
if n_inter > 0:
    # Linearly scale from aux_body_loss_weight to (1.0 - small gap), evenly spaced
    # For n_inter=1: [0.4]
    # For n_inter=2: [0.4, 0.6]
    import_step = (1.0 - self.aux_body_loss_weight) / (n_inter + 1)
    intermediate_weights = [
        self.aux_body_loss_weight + k * import_step
        for k in range(n_inter)
    ]
    # Equivalently for n_inter=2, aux=0.4: step=(1.0-0.4)/3=0.2 → [0.4+0.2, 0.4+0.4] = [0.6, 0.8]
    # Wait — this does not give [0.4, 0.6]. Use explicit formula instead.
```

**Simpler and correct explicit formula for Design 003:**
```python
# n_inter=2: weights = [0.4, 0.6] hard-coded for this design
# Builder: hard-code as [self.aux_body_loss_weight, self.aux_body_loss_weight * 1.5]
# This yields [0.4, 0.6] when aux_body_loss_weight=0.4.
# For safety, use min(len(...), len(intermediate_weights)) in the loop.
intermediate_weights = [self.aux_body_loss_weight * (1 + 0.5 * k) for k in range(1, n_inter + 1)]
# k=1: 0.4*1.5=0.6 — WRONG for layer-1. Fix: use 0-indexed:
intermediate_weights = [self.aux_body_loss_weight * (1.0 + 0.5 * k) for k in range(n_inter)]
# k=0: 0.4*1.0=0.4; k=1: 0.4*1.5=0.6 ✓
```

**Final specified formula** (Builder must use this exactly):
```python
n_inter = len(self._intermediate_outputs) - 1
intermediate_weights = [self.aux_body_loss_weight * (1.0 + 0.5 * k) for k in range(n_inter)]
# n_inter=1 (Design B): [0.4*1.0] = [0.4] ✓
# n_inter=2 (Design C): [0.4*1.0, 0.4*1.5] = [0.4, 0.6] ✓
```

16. Loss key naming: `loss/joints_aux_0/train` (layer-1), `loss/joints_aux_1/train` (layer-2). These keys must be unique and not overlap with existing baseline keys.
17. `joints_out` Linear(256, 3) is shared for all 3 output levels (intermediate and final). This is intentional — the same coordinate-prediction head is applied at each stage.

---

## Expected Behavior After Change

- Training losses logged: `loss/joints/train`, `loss/depth/train`, `loss/uv/train`, `loss/hand_aux/train`, `loss/joints_aux_0/train`, `loss/joints_aux_1/train`.
- Layer-1 output receives gradient from `loss/joints_aux_0/train` (w=0.4) + cascading gradients from layers 2 and 3.
- Layer-2 output receives gradient from `loss/joints_aux_1/train` (w=0.6) + cascading gradient from layer 3.
- Layer-3 output receives gradient from `loss/joints/train` (w=1.0) — the primary body loss.
- Expected forward-pass computation: slightly cheaper than baseline in total attention ops (66 vs. 70 FFN applications; 1,452 vs. 4,900 self-attention elements).
- Expected stage-1 composite_val: < 325 (target), potentially better than Designs 001/002 if 3-layer depth compounds the benefit.
- Expected stage-2 composite_val target: < 215.
- If OOM occurs (unlikely per VRAM analysis): reduce to `num_decoder_layers=2` in config — the code is fully parameterized. Report OOM to Orchestrator rather than silently downgrading.

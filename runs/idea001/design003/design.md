**Design Description:** Stack 4 decoder layers with auxiliary joint loss (weight 0.4) and a shared output projection across all layers; pelvis losses on final layer only.

**Starting Point:** `baseline/`

---

## Overview

**Algorithm:** Same DETR-style iterative refinement as Design B (3-layer + intermediate supervision), extended to 4 layers. The key algorithmic addition is a **shared output projection**: a single `Linear(hidden_dim, 3)` is applied at every decoder layer's output (including intermediates). This forces all layers to map into a common 3D pose coordinate space, acting as a representational regulariser — unlike independent per-layer heads that can diverge in their output geometry.

Extend Design B to 4 decoder layers and enforce a **single shared** `joints_out` Linear projection used at every layer output (including intermediates). In Design B the sharing is implicit (same module called multiple times in `forward`). This design makes the sharing explicit and intentional — the same `Linear(hidden_dim, 3)` maps every layer's token representation to pose space, which acts as a regulariser forcing a common pose-space geometry to emerge across refinement stages. Pelvis (`depth_out`, `uv_out`) outputs are from the final layer only.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

**New constructor signature** (add `num_decoder_layers` and `aux_loss_weight` after `dropout`, before `loss_joints`):

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_decoder_layers: int = 4,
    aux_loss_weight: float = 0.4,
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

Store new params:
```python
self.num_decoder_layers = num_decoder_layers
self.aux_loss_weight = aux_loss_weight
```

Replace single decoder layer with ModuleList of 4:
```python
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
```

Remove: `self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)`

Output projections — **`joints_out` is defined once and shared across all layers** (this is the key difference from Design B, which also shares in practice but does not document this as the design intent). `depth_out` and `uv_out` are also defined once (final-layer only, same as baseline):
```python
# Shared output projection (used at every decoder layer for pose)
self.joints_out = nn.Linear(hidden_dim, 3)
# Final-layer-only projections
self.depth_out = nn.Linear(hidden_dim, 1)
self.uv_out = nn.Linear(hidden_dim, 2)
```

These definitions are **identical in code to baseline** — no structural change. The sharing is enforced by the forward loop below calling `self.joints_out` multiple times.

**`forward` method changes:**

Replace:
```python
decoded = self.decoder_layer(queries, spatial)
```
With:
```python
decoded = queries
layer_outputs = []
for layer in self.decoder_layers:
    decoded = layer(decoded, spatial)
    layer_outputs.append(decoded)
```

Return dict:
```python
final = layer_outputs[-1]  # (B, num_joints, hidden_dim)
return {
    'joints': self.joints_out(final),                     # (B, num_joints, 3)
    'pelvis_depth': self.depth_out(final[:, 0, :]),       # (B, 1)
    'pelvis_uv': self.uv_out(final[:, 0, :]),             # (B, 2)
    'intermediate_joints': [
        self.joints_out(h) for h in layer_outputs[:-1]
    ],  # list of (B, num_joints, 3), length = num_decoder_layers - 1 = 3
}
```

**`loss` method changes:**

After computing `pred = self.forward(feats)` and building ground-truth tensors (unchanged from baseline), add auxiliary losses:

```python
_BODY = list(range(0, 22))
losses = dict()

# Final layer losses (weight 1.0)
losses['loss/joints/train'] = self.loss_joints_module(
    pred['joints'][:, _BODY], gt_joints[:, _BODY])
losses['loss/depth/train'] = self.loss_weight_depth * self.loss_depth_module(
    pred['pelvis_depth'], gt_depth)
losses['loss/uv/train'] = self.loss_weight_uv * self.loss_uv_module(
    pred['pelvis_uv'], gt_uv)

# Intermediate auxiliary losses (weight aux_loss_weight each)
for layer_idx, inter_joints in enumerate(pred['intermediate_joints']):
    losses[f'loss/joints_aux{layer_idx}/train'] = (
        self.aux_loss_weight * self.loss_joints_module(
            inter_joints[:, _BODY], gt_joints[:, _BODY])
    )
```

With `num_decoder_layers=4`, `pred['intermediate_joints']` has 3 entries:
- `loss/joints_aux0/train` — layer 0 output, weight 0.4
- `loss/joints_aux1/train` — layer 1 output, weight 0.4
- `loss/joints_aux2/train` — layer 2 output, weight 0.4

**`predict` method:** No changes. Accesses only `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']`.

**`_train_mpjpe` computation in `loss()`:** Unchanged — uses `pred['joints']` (final layer only).

### 2. `config.py`

Update head config dict:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    num_decoder_layers=4,
    aux_loss_weight=0.4,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config values remain identical to baseline:
- LR: 1e-4 (head), 1e-5 (backbone)
- Weight decay: 0.03
- Warmup: 3 epochs linear (start_factor=0.333), then cosine to epoch 20
- Batch: 4, accum: 8, effective batch: 32
- Seed: 2026

---

## Exact Loss Computation Summary

| Loss key | Layer | Weight |
|---|---|---|
| `loss/joints/train` | Final (layer 3) | 1.0 |
| `loss/depth/train` | Final (layer 3) | `loss_weight_depth` × module weight |
| `loss/uv/train` | Final (layer 3) | `loss_weight_uv` × module weight |
| `loss/joints_aux0/train` | Intermediate (layer 0) | 0.4 |
| `loss/joints_aux1/train` | Intermediate (layer 1) | 0.4 |
| `loss/joints_aux2/train` | Intermediate (layer 2) | 0.4 |

All joint losses apply `_BODY = list(range(0, 22))` masking — indices 0–21 only.

---

## Constraints and Invariants to Preserve

1. **Body-joint-only loss:** indices 0–21 for all joint losses (final and auxiliary).
2. **Pelvis losses on final layer only** — no auxiliary pelvis depth or UV losses.
3. **`self.joints_out` is the single shared Linear(hidden_dim, 3)** — called N times in forward (once per layer). `depth_out` and `uv_out` are called only once (final layer). Do not create per-layer output heads.
4. **`persistent_workers=False`** in both dataloaders — do not change.
5. **No Python `import` statements in `config.py`** — use `__import__()` or hardcode literals.
6. **Absolute imports in `pose3d_transformer_head.py`** — keep as-is.
7. **`_DecoderLayer` class is not modified** — 4 independent instances with separate weights.
8. **`predict()` must remain backward-compatible** — accesses only `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']`.
9. **OOM mitigation:** If OOM occurs during training with batch_size=4, reduce `hidden_dim` to 192 in both `config.py` (head dict) and note that `_build_2d_sincos_pos_enc` requires `embed_dim % 4 == 0` (192 % 4 == 0 — valid). Do not change batch size below 4.
10. **No changes** to `pelvis_utils.py`, `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `train.py`, `infra/`.

---

## Expected Behavior After Change

- Model: 4 sequential decoder layers with independent weights, one shared `joints_out`.
- Forward: queries → layer0 → layer1 → layer2 → layer3 → final output.
- Intermediate outputs at layers 0, 1, 2 all projected through the same `joints_out`.
- Gradient from 3 auxiliary losses + 1 final joint loss all flow back through `self.joints_out.weight`.
- Memory overhead on 1080 Ti: ~300–350 MB above baseline — marginal; monitor for OOM.
- Expected composite_val: comparable to Design B or slightly better due to regularisation effect of shared head; model size is smaller than 4 independent output heads.

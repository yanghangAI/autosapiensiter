**Design Description:** Stack 3 decoder layers with auxiliary joint loss (weight 0.4) at each intermediate layer; pelvis losses on final layer only.

**Starting Point:** `baseline/`

---

## Overview

**Algorithm:** DETR-style multi-layer iterative refinement with intermediate supervision. Each decoder layer refines query embeddings by attending to spatial backbone features via cross-attention. Auxiliary losses at intermediate layers provide direct gradient signal to early layers, preventing vanishing gradients and specialising each layer's attention to progressively finer pose structure.

Stack 3 `_DecoderLayer` instances. After each of the two intermediate layers (layer 0 and layer 1), apply an auxiliary joint coordinate loss (restricted to body joints 0–21) weighted at 0.4 × the final joint loss. The final layer (layer 2) carries the full joint loss (weight 1.0) plus the pelvis depth and UV losses. This forces gradient signal into early layers and encourages progressive pose refinement.

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
    num_decoder_layers: int = 3,
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

Replace single decoder layer with ModuleList:
```python
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
```

Remove: `self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)`

**`forward` method changes:**

Replace:
```python
decoded = self.decoder_layer(queries, spatial)
```
With a loop that collects **all** intermediate outputs (needed by `loss()`):
```python
decoded = queries
intermediate_outputs = []
for layer in self.decoder_layers:
    decoded = layer(decoded, spatial)
    intermediate_outputs.append(decoded)
```

Return dict must be extended to include intermediate outputs so `loss()` can supervise them:
```python
return {
    'joints': self.joints_out(intermediate_outputs[-1]),          # (B, num_joints, 3)
    'pelvis_depth': self.depth_out(intermediate_outputs[-1][:, 0, :]),  # (B, 1)
    'pelvis_uv': self.uv_out(intermediate_outputs[-1][:, 0, :]),        # (B, 2)
    'intermediate_joints': [
        self.joints_out(h) for h in intermediate_outputs[:-1]
    ],  # list of (B, num_joints, 3), length = num_decoder_layers - 1
}
```

Note: `self.joints_out` is **shared** across all calls here. This is intentional — the same linear head is reused for auxiliary projections, which acts as mild parameter sharing and is consistent with DETR-style per-layer prediction.

**`loss` method changes:**

After computing `pred = self.forward(feats)` and building ground-truth tensors (unchanged), add auxiliary losses:

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

# Intermediate auxiliary losses (weight aux_loss_weight per layer)
for layer_idx, inter_joints in enumerate(pred['intermediate_joints']):
    losses[f'loss/joints_aux{layer_idx}/train'] = (
        self.aux_loss_weight * self.loss_joints_module(
            inter_joints[:, _BODY], gt_joints[:, _BODY])
    )
```

Key rule: **no pelvis depth/UV auxiliary losses** — intermediate layers do not supervise the pelvis head. Only `loss/depth/train` and `loss/uv/train` from the final layer.

**`predict` method:** No changes. It calls `self.forward(feats)` and accesses `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']` — all still present in the returned dict.

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
    num_decoder_layers=3,
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
| `loss/joints/train` | Final (layer 2) | 1.0 |
| `loss/depth/train` | Final (layer 2) | `loss_weight_depth` × module weight |
| `loss/uv/train` | Final (layer 2) | `loss_weight_uv` × module weight |
| `loss/joints_aux0/train` | Intermediate (layer 0) | 0.4 |
| `loss/joints_aux1/train` | Intermediate (layer 1) | 0.4 |

All joint losses apply `_BODY = list(range(0, 22))` masking — indices 0–21 only.

---

## Constraints and Invariants to Preserve

1. **Body-joint-only loss:** indices 0–21 for all joint losses (final and auxiliary).
2. **Pelvis losses on final layer only** — no auxiliary pelvis depth or UV losses.
3. **`self.joints_out` is reused** for all auxiliary projections — single shared Linear(hidden_dim, 3).
4. **`persistent_workers=False`** in both dataloaders — do not change.
5. **No Python `import` statements in `config.py`** — use `__import__()` or hardcode literals.
6. **Absolute imports in `pose3d_transformer_head.py`** — keep as-is.
7. **`_DecoderLayer` class is not modified** — only instantiation and loop changes.
8. **`predict()` must remain backward-compatible** — it only accesses `pred['joints']`, `pred['pelvis_depth']`, `pred['pelvis_uv']`. The `'intermediate_joints'` key is ignored at inference time.
9. **No changes** to `pelvis_utils.py`, `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `train.py`, `infra/`.

---

## Expected Behavior After Change

- Model: 3 sequential decoder layers, independent weights.
- Forward: queries → layer0 → layer1 → layer2 → final output.
- Auxiliary outputs collected after layer0 and layer1, projected via shared `joints_out`.
- Loss: final joint + aux joint at 0.4× per intermediate + pelvis losses (final only).
- Memory overhead on 1080 Ti: ~200–250 MB above baseline — within 8 GB budget with batch 4.
- Expected composite_val: 5–10% improvement over baseline (primary bet from idea.md).

**Design Description:** Stack 2 decoder layers (no auxiliary loss) — capacity ablation.

**Starting Point:** `baseline/`

---

## Overview

Replace the single `_DecoderLayer` in `Pose3dTransformerHead` with a `nn.ModuleList` of 2 identical `_DecoderLayer` instances. The decoder loop passes queries through both layers in sequence; only the final layer's output drives the joint, depth, and UV losses. No auxiliary losses.

**Algorithm:** Standard iterative refinement — each decoder layer refines the query representations by attending to the spatial feature map, producing progressively better pose estimates. This ablation tests whether additional capacity (deeper decoder) alone improves performance, isolating the capacity effect from intermediate supervision.

---

## Files to Change

### 1. `pose3d_transformer_head.py`

**`__init__` changes:**

Replace:
```python
# Transformer decoder (1 layer)
self.decoder_layer = _DecoderLayer(hidden_dim, num_heads, dropout)
```
With:
```python
# Transformer decoder (N layers)
self.decoder_layers = nn.ModuleList([
    _DecoderLayer(hidden_dim, num_heads, dropout)
    for _ in range(num_decoder_layers)
])
```

Add `num_decoder_layers: int = 2` as a constructor parameter (after `dropout`, before `loss_joints`):
```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    num_decoder_layers: int = 2,
    loss_joints: ConfigType = ...,
    ...
):
```
Store it: `self.num_decoder_layers = num_decoder_layers`

**`forward` changes:**

Replace:
```python
decoded = self.decoder_layer(queries, spatial)
```
With:
```python
decoded = queries
for layer in self.decoder_layers:
    decoded = layer(decoded, spatial)
```

All downstream output projections (`joints_out`, `depth_out`, `uv_out`) remain applied to the final `decoded` — no change needed there.

**`_init_head_weights` — no changes required.** Output projection init is unchanged.

**No changes to `loss()` or `predict()`.** Both already consume `pred = self.forward(feats)` and operate on the single final output.

### 2. `config.py`

Add `num_decoder_layers=2` and `aux_loss_weight=0.0` to the head config dict (aux_loss_weight is unused in Design A but included for consistency so all designs share the same head kwargs pattern):

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    num_decoder_layers=2,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
),
```

All other config values remain identical to baseline:
- LR: 1e-4 (head), 1e-5 (backbone, via lr_mult=0.1)
- Weight decay: 0.03
- Warmup: 3 epochs linear (start_factor=0.333), then cosine to epoch 20
- Batch: 4, accum: 8, effective batch: 32
- Seed: 2026

---

## Constraints and Invariants to Preserve

1. **Joint loss scope:** restrict to `_BODY = list(range(0, 22))` — indices 0–21 only. Unchanged from baseline.
2. **Pelvis token:** still `decoded[:, 0, :]` from the final decoder layer output.
3. **`persistent_workers=False`** in both dataloaders — do not change.
4. **No Python `import` statements in `config.py`** — use `__import__()` or hardcode literals.
5. **Absolute imports in `pose3d_transformer_head.py`** (e.g., `from mmpose.models.heads.base_head import BaseHead`) — keep as-is.
6. **`_DecoderLayer` class itself is not modified** — only instantiation count changes.
7. **`default_init_cfg`** returns `[]` — do not change.
8. **No changes** to `pelvis_utils.py`, `bedlam_metric.py`, `bedlam2_dataset.py`, `bedlam2_transforms.py`, `sapiens_rgbd.py`, `train.py`, `infra/constants.py`, `infra/metrics_csv_hook.py`.

---

## Expected Behavior After Change

- Model has two sequential decoder layers (same architecture, independent weights).
- Forward pass: queries → layer0 → layer1 → output projections.
- Loss computation is identical to baseline — single joint/depth/UV loss from final layer only.
- Parameter count increases by ~1× the single decoder layer (~3.2M extra params at hidden_dim=256, num_heads=8).
- Memory overhead on 1080 Ti (8 GB) estimated ~100–150 MB — within budget.
- Expected composite_val improvement: marginal to moderate (tests pure depth effect without supervision pressure on intermediate layers).

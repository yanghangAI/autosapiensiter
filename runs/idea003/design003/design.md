**Design Description:** Two-layer bottleneck MLP global conditioning on joint queries with LayerNorm applied to per-joint offsets before addition.

**Starting Point:** `baseline/`

---

## Overview

The algorithm change is the same bottleneck MLP as design002 (hidden_dim → 128 → num_joints*hidden_dim, GELU), extended with a `nn.LayerNorm(hidden_dim)` applied to each per-joint offset vector before addition to the static query. The LayerNorm normalises the offset magnitude so the static query always controls the scale of the combined embedding, preventing large offsets from overwhelming the static component during early training. This is the most stable variant.

---

## Files to Modify

1. `pose3d_transformer_head.py` — add `query_cond_net` MLP and `query_cond_norm` LayerNorm, update `__init__` and `forward`.
2. `config.py` — add `query_cond_type='mlp_norm'` kwarg to the head config dict.

`pelvis_utils.py` is **not** modified.

---

## Exact Changes — `pose3d_transformer_head.py`

### `__init__` additions

Add `query_cond_type: str = 'mlp_norm'` as a new parameter to `__init__` (after `init_cfg`). The full updated signature is:

```python
def __init__(
    self,
    in_channels: int,
    hidden_dim: int = 256,
    num_joints: int = 70,
    num_heads: int = 8,
    dropout: float = 0.1,
    loss_joints: ConfigType = dict(...),
    loss_depth: ConfigType = dict(...),
    loss_uv: ConfigType = dict(...),
    loss_weight_depth: float = 1.0,
    loss_weight_uv: float = 1.0,
    query_cond_type: str = 'mlp_norm',
    init_cfg: OptConfigType = None,
):
```

After the line that creates `self.decoder_layer`, add:

```python
# Content-adaptive query conditioning (Design C: bottleneck MLP + LayerNorm on offsets)
self.query_cond_type = query_cond_type
bottleneck_dim = hidden_dim // 2  # 128 when hidden_dim=256
if query_cond_type == 'mlp_norm':
    self.query_cond_net = nn.Sequential(
        nn.Linear(hidden_dim, bottleneck_dim),
        nn.GELU(),
        nn.Linear(bottleneck_dim, num_joints * hidden_dim),
    )
    # LayerNorm applied per-joint (normalises each hidden_dim vector independently)
    self.query_cond_norm = nn.LayerNorm(hidden_dim)
    # Initialise MLP weights with trunc_normal std=0.02, zero biases
    for layer in self.query_cond_net:
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
    # LayerNorm uses default init (weight=1, bias=0 — PyTorch default)
else:
    raise ValueError(f'Unknown query_cond_type: {query_cond_type}')
```

Exact bottleneck dimensions (same as design002):
- Layer 1: `nn.Linear(256, 128)` — weight shape `(128, 256)`, 32,768 params + 128 bias.
- Layer 2: `nn.Linear(128, 17920)` — weight shape `(17920, 128)`, 2,293,760 params + 17,920 bias.
- `query_cond_norm`: `nn.LayerNorm(256)` — 512 params (weight + bias).
- Total additional params: ~2.34 M (negligibly more than design002).

`_init_head_weights` is **not** modified.

### `forward` changes

Replace the block that constructs `queries` and calls `self.decoder_layer`:

**Before (baseline):**
```python
# Broadcast joint queries to batch
queries = self.joint_queries.weight.unsqueeze(0).expand(
    B, -1, -1)  # (B, num_joints, hidden_dim)

# Decoder
decoded = self.decoder_layer(queries, spatial)
```

**After:**
```python
# Static joint query embeddings, broadcast to batch
static_q = self.joint_queries.weight.unsqueeze(0).expand(
    B, -1, -1)  # (B, num_joints, hidden_dim)

# Content-adaptive offset: mean-pool spatial tokens -> per-joint delta
global_feat = spatial.mean(dim=1)  # (B, hidden_dim); spatial has pos_enc added
offsets = self.query_cond_net(global_feat)           # (B, num_joints * hidden_dim)
offsets = offsets.reshape(B, self.num_joints, self.hidden_dim)  # (B, num_joints, hidden_dim)
# Normalise each per-joint offset vector to unit-ish magnitude
offsets = self.query_cond_norm(offsets)              # (B, num_joints, hidden_dim)
queries = static_q + offsets                         # (B, num_joints, hidden_dim)

# Decoder
decoded = self.decoder_layer(queries, spatial)
```

The `global_feat` mean-pool is taken **after** `spatial = spatial + pos_enc`.

The LayerNorm is applied to the reshaped offsets tensor `(B, num_joints, hidden_dim)`. `nn.LayerNorm(hidden_dim)` normalises over the last dimension (hidden_dim=256), independently for each (batch, joint) pair. This is a standard use of LayerNorm on a 3D tensor.

Everything after `decoded = ...` (output projections, loss, predict) is **unchanged**.

---

## Exact Changes — `config.py`

In the `head` dict inside `model`, add `query_cond_type='mlp_norm'`:

```python
head=dict(
    type='Pose3dTransformerHead',
    in_channels=embed_dim,
    hidden_dim=256,
    num_joints=num_joints,
    num_heads=8,
    dropout=0.1,
    loss_joints=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_depth=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_uv=dict(type='SoftWeightSmoothL1Loss', beta=0.05, loss_weight=1.0),
    loss_weight_depth=1.0,
    loss_weight_uv=1.0,
    query_cond_type='mlp_norm',
),
```

All other config values (LR, weight decay, batch size, data pipeline, hooks, schedule) are **identical to baseline**.

---

## Constraints and Invariants

1. **Loss restriction**: joint loss remains on indices 0-21 only (`_BODY = list(range(0, 22))`). Do not change.
2. **Pelvis pathway**: `pelvis_depth` and `pelvis_uv` still read from `decoded[:, 0, :]`. Do not change.
3. **Positional encoding**: `global_feat = spatial.mean(dim=1)` must be computed after `spatial = spatial + pos_enc`.
4. **LayerNorm placement**: `self.query_cond_norm` must be applied to the **reshaped** offsets `(B, num_joints, hidden_dim)`, not to the flat `(B, num_joints * hidden_dim)` output. Normalising over `hidden_dim=256` (last dim) is correct.
5. **LayerNorm default init**: PyTorch initialises `nn.LayerNorm` with `weight=1.0` and `bias=0.0` by default. Do **not** override this — the default init lets the offset direction matter while keeping magnitude at ~1.
6. **Zero-bias init for MLP**: all `nn.Linear` biases in `query_cond_net` must be initialised to zeros.
7. **trunc_normal init for MLP**: all `nn.Linear` weights in `query_cond_net` use `trunc_normal_(std=0.02)`.
8. **Bottleneck dim**: `hidden_dim // 2 = 128`. Proportional if `hidden_dim` changes.
9. **GELU activation**: use `nn.GELU()`. No dropout inside `query_cond_net`.
10. **`nn.Sequential` iteration**: init loop uses `isinstance(layer, nn.Linear)` to skip `nn.GELU()`.
11. **persistent_workers=False**: unchanged.
12. **Seed**: `randomness = dict(seed=2026)` unchanged.
13. **MMEngine config**: `query_cond_type='mlp_norm'` is a plain string literal.

---

## Expected Behavior

- At initialization: `query_cond_net` outputs near-zero offsets (zero bias) → after LayerNorm, offsets may not be exactly zero (LayerNorm normalises to unit variance). The static queries still dominate early because the normalized offsets have unit magnitude (on average) whereas the static queries have been learned over epochs. In practice the LayerNorm effect at init is small because `trunc_normal_(std=0.02)` weights produce small pre-norm offsets.
- During training: LayerNorm prevents runaway growth of the adaptive component. The static query scale is preserved and the offset direction is learned stably.
- Body MPJPE: expected improvement of −10 to −20 mm relative to baseline (~168 mm) by epoch 20. More stable training curve than design002, especially in early epochs.
- Pelvis metrics: neutral to slight improvement.
- Composite target: aim for composite_val < 160 (baseline ~170.5).
- Training instability risk: lowest of the three designs due to normalised offsets.

**Design Description:** Two-layer bottleneck MLP global conditioning on joint queries (hidden_dim → 128 → num_joints*hidden_dim, no LayerNorm on offsets).

**Starting Point:** `baseline/`

---

## Overview

The algorithm change replaces static joint-query embeddings with image-conditioned queries, using a two-layer bottleneck MLP that maps globally-averaged spatial tokens to per-joint additive offsets. The bottleneck (dim 128) reduces parameters versus a single wide linear while adding a nonlinearity (GELU) that lets the network compose global features before projecting to query space. All other architecture, losses, and training infrastructure are unchanged.

---

## Files to Modify

1. `pose3d_transformer_head.py` — add `query_cond_net` MLP, update `__init__` and `forward`.
2. `config.py` — add `query_cond_type='mlp'` kwarg to the head config dict.

`pelvis_utils.py` is **not** modified.

---

## Exact Changes — `pose3d_transformer_head.py`

### `__init__` additions

Add `query_cond_type: str = 'mlp'` as a new parameter to `__init__` (after `init_cfg`). The full updated signature is:

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
    query_cond_type: str = 'mlp',
    init_cfg: OptConfigType = None,
):
```

After the line that creates `self.decoder_layer`, add:

```python
# Content-adaptive query conditioning (Design B: bottleneck MLP)
self.query_cond_type = query_cond_type
bottleneck_dim = hidden_dim // 2  # 128 when hidden_dim=256
if query_cond_type == 'mlp':
    self.query_cond_net = nn.Sequential(
        nn.Linear(hidden_dim, bottleneck_dim),
        nn.GELU(),
        nn.Linear(bottleneck_dim, num_joints * hidden_dim),
    )
    # Initialise weights with trunc_normal std=0.02, zero biases
    for layer in self.query_cond_net:
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)
else:
    raise ValueError(f'Unknown query_cond_type: {query_cond_type}')
```

Exact bottleneck dimensions:
- Layer 1: `nn.Linear(256, 128)` — weight shape `(128, 256)`, 32,768 params + 128 bias.
- Layer 2: `nn.Linear(128, 70*256)` = `nn.Linear(128, 17920)` — weight shape `(17920, 128)`, 2,293,760 params + 17,920 bias.
- Total additional params: ~2.34 M (vs. ~4.61 M for design001's single linear).

`_init_head_weights` is **not** modified (it handles only `joint_queries` and output projections).

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
queries = static_q + offsets                         # (B, num_joints, hidden_dim)

# Decoder
decoded = self.decoder_layer(queries, spatial)
```

The `global_feat` mean-pool is taken **after** `spatial = spatial + pos_enc` (after positional encoding has been added).

Everything after `decoded = ...` (output projections, loss, predict) is **unchanged**.

---

## Exact Changes — `config.py`

In the `head` dict inside `model`, add `query_cond_type='mlp'`:

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
    query_cond_type='mlp',
),
```

All other config values (LR, weight decay, batch size, data pipeline, hooks, schedule) are **identical to baseline**.

---

## Constraints and Invariants

1. **Loss restriction**: joint loss remains on indices 0-21 only (`_BODY = list(range(0, 22))`). Do not change.
2. **Pelvis pathway**: `pelvis_depth` and `pelvis_uv` still read from `decoded[:, 0, :]`. Do not change.
3. **Positional encoding**: `global_feat = spatial.mean(dim=1)` must be computed after `spatial = spatial + pos_enc`.
4. **Zero-bias init**: all `nn.Linear` biases in `query_cond_net` must be initialised to zeros.
5. **trunc_normal init**: all `nn.Linear` weights in `query_cond_net` use `trunc_normal_(std=0.02)`.
6. **Bottleneck dim**: `hidden_dim // 2 = 128` (integer floor division). If `hidden_dim` is ever changed, this remains proportional automatically.
7. **GELU activation**: use `nn.GELU()` (not ReLU). No dropout inside `query_cond_net`.
8. **No LayerNorm on offsets**: this design intentionally omits LayerNorm (that is reserved for design003).
9. **`nn.Sequential` iteration**: the init loop uses `isinstance(layer, nn.Linear)` to skip the `nn.GELU()` module.
10. **persistent_workers=False**: unchanged.
11. **Seed**: `randomness = dict(seed=2026)` unchanged.
12. **MMEngine config**: `query_cond_type='mlp'` is a plain string literal.

---

## Expected Behavior

- At initialization: `query_cond_net` outputs near-zero offsets due to zero-bias init → model is functionally near-equivalent to baseline.
- During training: the nonlinear bottleneck learns a compact scene representation (body scale, global orientation) before projecting to joint-query offsets.
- Body MPJPE: expected improvement of −10 to −20 mm relative to baseline (~168 mm) by epoch 20. The bottleneck compression may yield slightly better generalisation than design001's single large linear.
- Pelvis metrics: neutral to slight improvement.
- Composite target: aim for composite_val < 160 (baseline ~170.5).
- Fewer parameters than design001 (~2.34 M vs ~4.61 M), which may reduce overfitting risk.

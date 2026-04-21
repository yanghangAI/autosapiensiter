**Design Description:** Single-linear global conditioning on joint queries (minimal additive offset from mean-pooled spatial tokens).

**Starting Point:** `baseline/`

---

## Overview

Replace purely static joint-query embeddings in the baseline decoder with image-conditioned queries. The core algorithm change is a single `nn.Linear(hidden_dim, num_joints * hidden_dim)` that projects globally-averaged spatial tokens into per-joint additive offsets, which are added to the static `joint_queries.weight` before the decoder layer. All other architecture, losses, and training infrastructure are unchanged.

---

## Files to Modify

1. `pose3d_transformer_head.py` — add `query_cond_net` linear, update `__init__` and `forward`.
2. `config.py` — add `query_cond_type='linear'` kwarg to the head config dict.

`pelvis_utils.py` is **not** modified.

---

## Exact Changes — `pose3d_transformer_head.py`

### `__init__` additions

After the line that creates `self.decoder_layer`, add:

```python
# Content-adaptive query conditioning (Design A: single linear)
self.query_cond_type = query_cond_type  # stored for forward dispatch
if query_cond_type == 'linear':
    self.query_cond_net = nn.Linear(hidden_dim, num_joints * hidden_dim)
    # Initialise weights with trunc_normal std=0.02, zero bias
    nn.init.trunc_normal_(self.query_cond_net.weight, std=0.02)
    nn.init.zeros_(self.query_cond_net.bias)
else:
    raise ValueError(f'Unknown query_cond_type: {query_cond_type}')
```

Add `query_cond_type: str = 'linear'` as a new parameter to `__init__` (after `init_cfg`). The full updated signature is:

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
    query_cond_type: str = 'linear',
    init_cfg: OptConfigType = None,
):
```

`_init_head_weights` must **not** initialise `query_cond_net` (it is initialised inline above). No other changes to `_init_head_weights`.

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
global_feat = spatial.mean(dim=1)  # (B, hidden_dim); spatial already has pos_enc added
offsets = self.query_cond_net(global_feat)           # (B, num_joints * hidden_dim)
offsets = offsets.reshape(B, self.num_joints, self.hidden_dim)  # (B, num_joints, hidden_dim)
queries = static_q + offsets                         # (B, num_joints, hidden_dim)

# Decoder
decoded = self.decoder_layer(queries, spatial)
```

The `global_feat` mean-pool is taken **after** `spatial = spatial + pos_enc` (i.e., after positional encoding has been added), so the global representation includes positional information.

Everything after `decoded = ...` (output projections, loss, predict) is **unchanged**.

---

## Exact Changes — `config.py`

In the `head` dict inside `model`, add `query_cond_type='linear'`:

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
    query_cond_type='linear',
),
```

All other config values (LR, weight decay, batch size, data pipeline, hooks, schedule) are **identical to baseline**.

---

## Parameter Count

- `query_cond_net`: weight `(256, 70*256) = (256, 17920)` → 4,587,520 params + 17,920 bias = ~4.61 M additional params.
- ~5% overhead on top of the head.

---

## Constraints and Invariants

1. **Loss restriction**: joint loss remains on indices 0-21 only (`_BODY = list(range(0, 22))`). Do not change.
2. **Pelvis pathway**: `pelvis_depth` and `pelvis_uv` still read from `decoded[:, 0, :]`. Do not change.
3. **Positional encoding**: `global_feat = spatial.mean(dim=1)` must be computed after `spatial = spatial + pos_enc`, not before.
4. **Zero-bias init**: `query_cond_net.bias` must be initialised to zeros so that at epoch 0 the offsets start near zero, matching baseline queries.
5. **trunc_normal init**: `query_cond_net.weight` uses `trunc_normal_(std=0.02)` — same as static query init.
6. **No new imports needed**: `nn.Linear` is already available via `torch.nn`.
7. **MMEngine config**: `query_cond_type='linear'` is a plain string literal — no import needed.
8. **`expand` vs `repeat`**: the static queries use `.expand(B, -1, -1)` (no memory copy). The offset is a freshly computed tensor, so adding to an expanded view is safe (PyTorch broadcasts correctly).
9. **persistent_workers=False**: unchanged.
10. **Seed**: `randomness = dict(seed=2026)` unchanged.

---

## Expected Behavior

- At initialization: `query_cond_net` outputs near-zero offsets → model is functionally equivalent to baseline.
- During training: the MLP learns to shift queries toward image-relevant regions, giving cross-attention a warm start.
- Body MPJPE: expected improvement of −10 to −20 mm relative to baseline (~168 mm) by epoch 20.
- Pelvis metrics: neutral to slight improvement (global scale context may help depth regression marginally).
- Composite target: aim for composite_val < 160 (baseline ~170.5).
- No memory regression on 1080 Ti: one extra Linear forward per step, negligible cost.
